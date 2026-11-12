"""The one failure-classification table every adapter shares, with no vendor import.

`classify_core` names the `Failure` a call ended with, from the exception it raised (if any) and
the facts the wire showed (`WireFacts`). The order is fixed, and the first match wins:

1. `extra` classifiers, in order: gateway-specific knowledge lives there, not in this table.
2. An exception of a `gated` type anywhere in the chain is `gated`; an HTTP client's
   `InvalidURL` or `UnsupportedProtocol` anywhere in the chain is `internal` (`CALLER_CONFIG_ERRORS`:
   the caller's URL could not be sent, so no request left and no backend is at fault).
3. `asyncio.CancelledError` is never classified: it is re-raised, before even step 1 sees it.
4. The framework's own categories (`categorize`, supplied by the adapter that knows them).
5. A provider error code or message that names a context overflow is `context_overflow`, whatever
   the status (an OpenAI-compatible server sends a bare 400).
6. The HTTP status alone.
7. A transport error (httpx, httpx2, the OpenAI SDK's connection errors, the built-in
   `TimeoutError` and `ConnectionError`) is `timeout` or `transport`, with its phase.
8. No exception: a refusal or a `content_filter` finish is `refused`; a `length` finish is
   `truncated`, whether or not the output parsed; anything else is success (None).
9. An exception raised while a provider client was reading a response that ARRIVED (a status is
   known, and its innermost owning frame is in `PROVIDER_MODULES`; the standard library and
   pydantic are skipped as neutral) is
   `malformed_response`, under `unavailable` and retryable: a 200 with `choices: []` is the
   provider's fault. Anything else is `internal`, under `request_fault`: a bug in the caller's
   code is not an outage, so it is never retried, failed over, or counted against a backend.

An exception group (a fallback chain's `FallbackExceptionGroup`) is classified member by member
through this same order, and the worst member names the call (`SEVERITY`).

This module is shared by the adapters that classify (the pydantic-ai generator and embedder, the
HTTP reranker, the LiteLLM gateway reader), so it sits directly under `adapters/`, outside the
adapter-independence contract, and imports no vendor: transport errors are recognised by their
class names along the MRO, which httpx and httpx2 share.

Invariants:
    - Pure: no I/O, never raises except to re-raise a cancellation.
    - A `reason` never carries a response body.

Rationale: ADR-0020.
"""

from __future__ import annotations

import asyncio
import email.utils
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from llm_ports.ports import Failure, FailureKind, Phase, UnifiedFinish, failure
from llm_ports.ports.outcomes import REASON_MAX, exception_chain, failure_reason

#: The longest provider error message `WireFacts` keeps, in characters.
ERROR_MESSAGE_MAX = 200

#: Provider error codes and types that name a context overflow.
OVERFLOW_CODES = frozenset({"context_length_exceeded", "exceed_context_size_error", "context_window_exceeded"})
#: Lower-case fragments of a provider error message that name a context overflow.
OVERFLOW_MESSAGES = ("maximum context length", "prompt is too long", "input token count", "context window")

#: HTTP status to kind, for a status no earlier step explained (step 6).
STATUS_KINDS: Mapping[int, FailureKind] = {
    400: "bad_request",
    401: "auth",
    403: "auth",
    404: "not_found",
    408: "timeout",
    413: "bad_request",
    422: "bad_request",
    429: "rate_limited",
    503: "overloaded",
    529: "overloaded",
}

#: Modules whose code reads a provider's response: an exception raised there is the response's
#: fault (step 9), not the caller's.
PROVIDER_MODULES = ("pydantic_ai.models", "openai", "httpx", "httpx2", "httpcore")

#: Every kind, worst first: the order an exception group's members are ranked by. A failure no
#: retry fixes outranks one a retry might; among those, the caller's own fault outranks the
#: backend's.
SEVERITY: tuple[FailureKind, ...] = (
    "internal",
    "bad_request",
    "context_overflow",
    "unsupported",
    "auth",
    "not_found",
    "quota",
    "caller_limit",
    "gated",
    "refused",
    "invalid_output",
    "truncated",
    "malformed_response",
    "server",
    "overloaded",
    "rate_limited",
    "timeout",
    "transport",
)

_TRANSPORT_MODULES = ("httpx", "httpx2", "httpcore", "openai")
#: HTTP-client errors raised before any byte leaves because the caller's URL cannot be sent: the
#: caller's configuration, never a backend's (step 2). Matched by class name along the MRO.
CALLER_CONFIG_ERRORS = ("InvalidURL", "UnsupportedProtocol")
#: Packages a frame is skipped through when finding who raised: they are called by both sides.
_NEUTRAL = frozenset({"pydantic", "pydantic_core", "typing_extensions", "anyio", "sniffio"})
_PHASES: Mapping[str, Phase] = {"Connect": "connect", "Read": "read", "Write": "write", "Pool": "pool"}


@dataclass(frozen=True, slots=True)
class WireFacts:
    """What the wire showed of a call's last response, with no body.

    Attributes:
        status: The HTTP status, when a response arrived.
        headers: The response headers, names lower-cased.
        finish_raw: The reply's finish reason, verbatim.
        error_code: The provider's error code, verbatim.
        error_type: The provider's error type, verbatim.
        error_message: The provider's error text, at most `ERROR_MESSAGE_MAX` characters. Never the
            prompt or the completion.
        refusal: True when the reply carried an explicit refusal field.
    """

    status: int | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    finish_raw: str | None = None
    error_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    refusal: bool = False


class Classifier(Protocol):
    """An `extra` classifier: knowledge a gateway or provider adds ahead of the shared table."""

    def __call__(self, exc: BaseException | None, response: WireFacts, /) -> Failure | None:
        """Return the failure `exc` and `response` mean, or None to leave them to the table."""
        ...


#: The framework step (4): the exception chain, the wire facts and the unified finish, to a
#: `Failure` or None. The pydantic-ai adapter supplies one; the reranker has none.
Categorize = Callable[[Sequence[BaseException], WireFacts, UnifiedFinish | None], Failure | None]


def retry_after_s(headers: Mapping[str, str]) -> float | None:
    """Return the server's wait hint in seconds, from `retry-after-ms` or `Retry-After`.

    Pure. `Retry-After` may be seconds or an HTTP date. A hint in the past, unreadable or not
    finite is None.
    """
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            v = float(ms) / 1000
            return v if math.isfinite(v) and v >= 0 else None
        except ValueError:
            pass
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        v = float(raw)
        return v if math.isfinite(v) and v >= 0 else None
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    left = when.timestamp() - time.time()
    return left if left >= 0 else None


def is_context_overflow(codes: Sequence[str | None], messages: Sequence[str | None]) -> bool:
    """True when a provider code or message names a context overflow (step 5). Pure."""
    if any(c is not None and c.lower() in OVERFLOW_CODES for c in codes):
        return True
    return any(m is not None and any(frag in m.lower() for frag in OVERFLOW_MESSAGES) for m in messages)


def _status_of(chain: Sequence[BaseException], facts: WireFacts) -> tuple[int | None, BaseException | None]:
    """The HTTP status of the call, and the exception that carried it (None when the wire did)."""
    for e in chain:
        code = getattr(e, "status_code", None)
        if isinstance(code, int):
            return code, e
        resp = getattr(e, "response", None)
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code, e
    return facts.status, None


def _codes(chain: Sequence[BaseException], facts: WireFacts) -> list[str | None]:
    out: list[str | None] = [facts.error_code, facts.error_type]
    for e in chain:
        for attr in ("provider_error_code", "provider_error_type", "code", "type"):
            v = getattr(e, attr, None)
            if isinstance(v, str):
                out.append(v)
    return out


def _provider_messages(chain: Sequence[BaseException]) -> list[str]:
    """The text of each exception in the chain that came from a provider (it carries a status or a body)."""
    return [str(e) for e in chain if hasattr(e, "status_code") or hasattr(e, "body")]


def _caller_config(chain: Sequence[BaseException]) -> Failure | None:
    """Step 2: an HTTP client refused the caller's URL before sending anything."""
    for e in chain:
        for cls in type(e).__mro__:
            if cls.__name__ in CALLER_CONFIG_ERRORS and cls.__module__.split(".")[0] in _TRANSPORT_MODULES:
                return failure("internal", reason=failure_reason(e, with_cause=True))
    return None


def _transport(chain: Sequence[BaseException]) -> Failure | None:
    """Step 7: a transport or timeout error anywhere in the chain."""
    for e in chain:
        for cls in type(e).__mro__:
            if cls.__module__.split(".")[0] not in _TRANSPORT_MODULES:
                continue
            name = cls.__name__
            if name in ("TimeoutException", "APITimeoutError"):
                return failure("timeout", phase=_phase(type(e).__name__), reason=failure_reason(e))
            if name in ("TransportError", "NetworkError", "RequestError", "APIConnectionError"):
                return failure("transport", phase=_phase(type(e).__name__), reason=failure_reason(e))
    for e in chain:
        if isinstance(e, TimeoutError):
            return failure("timeout", reason=failure_reason(e))
        if isinstance(e, ConnectionError):
            return failure("transport", phase="connect", reason=failure_reason(e))
    return None


def raised_in_provider(e: BaseException) -> bool:
    """True when `e` was raised by code in `PROVIDER_MODULES` (step 9). Pure.

    The owner is the innermost frame of `e`'s traceback whose package is neither the standard
    library nor a neutral helper (pydantic, anyio): a `KeyError` from `json` inside a caller's
    validator is owned by the validator; an `IndexError` in `pydantic_ai.models.openai` by the
    provider client.
    """
    tb = e.__traceback__
    owners: list[str] = []
    while tb is not None:
        owners.append(str(tb.tb_frame.f_globals.get("__name__", "")))
        tb = tb.tb_next
    for mod in reversed(owners):
        top = mod.split(".")[0]
        if top in sys.stdlib_module_names or top in _NEUTRAL or not top:
            continue
        return any(mod == p or mod.startswith(p + ".") for p in PROVIDER_MODULES)
    return False


def worst(failures: Sequence[Failure]) -> Failure:
    """Return the worst of `failures` by `SEVERITY`; the first of equals. Pure.

    Raises:
        ValueError: `failures` is empty.
    """
    if not failures:
        raise ValueError("worst() of no failures")
    return min(failures, key=lambda f: SEVERITY.index(f.kind))


def _phase(class_name: str) -> Phase | None:
    for prefix, phase in _PHASES.items():
        if class_name.startswith(prefix):
            return phase
    return None


def kind_for_status(status: int) -> FailureKind:
    """Return the kind an HTTP error status maps to (step 6). Pure.

    A status not in `STATUS_KINDS` is `server` (outcome `unavailable`), as 1.x mapped it: only the
    listed 4xx say the input itself was refused.
    """
    return STATUS_KINDS.get(status, "server")


def classify_core(
    exc: BaseException | None,
    facts: WireFacts,
    *,
    finish: UnifiedFinish | None = None,
    extra: Sequence[Classifier] = (),
    gated: tuple[type[BaseException], ...] = (),
    categorize: Categorize | None = None,
) -> Failure | None:
    """Return the failure a call ended with, or None for a success; the module docstring gives the order.

    Args:
        exc: The exception the call raised; None when it returned.
        facts: What the wire showed of the last response.
        finish: The unified reading of `facts.finish_raw`, from the adapter's own mapping.
        extra: Classifiers consulted first, in order.
        gated: Exception types that mean the caller's gate refused the call.
        categorize: The framework's own categories (step 4).

    Raises:
        asyncio.CancelledError: Re-raised when `exc` is one; a cancellation is never an outcome.

    Rationale: ADR-0020.
    """
    if isinstance(exc, asyncio.CancelledError):
        raise exc
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        members = [
            classify_core(m, facts, finish=finish, extra=extra, gated=gated, categorize=categorize)
            for m in exc.exceptions
        ]
        named = worst([m for m in members if m is not None])
        return replace(named, reason=f"{type(exc).__name__}: {named.reason}"[:REASON_MAX])
    for c in extra:
        got = c(exc, facts)
        if got is not None:
            return got
    chain = list(exception_chain(exc)) if exc is not None else []
    for e in chain:
        if gated and isinstance(e, gated):
            return failure("gated", reason=str(e)[:REASON_MAX] or type(e).__name__)
    got = _caller_config(chain)
    if got is not None:
        return got
    if chain and categorize is not None:
        got = categorize(chain, facts, finish)
        if got is not None:
            return got
    status, carrier = _status_of(chain, facts)
    failed = exc is not None or (status is not None and status >= 400)
    if failed and is_context_overflow(_codes(chain, facts), [facts.error_message, *_provider_messages(chain)]):
        return failure(
            "context_overflow",
            status=status,
            provider_code=facts.error_code or facts.error_type,
            reason=failure_reason(exc) if exc is not None else f"HTTP {status}: context overflow",
        )
    if status is not None and status >= 400:
        hint = retry_after_s(facts.headers)
        if hint is None and carrier is not None:
            got_hint = getattr(carrier, "retry_after", None)
            hint = float(got_hint) if isinstance(got_hint, int | float) else None
        return failure(
            kind_for_status(status),
            status=status,
            retry_after_s=hint,
            provider_code=facts.error_code or facts.error_type,
            reason=failure_reason(carrier) if carrier is not None else f"HTTP {status}",
        )
    if chain:
        got = _transport(chain)
        if got is not None:
            return got
        for e in chain:
            if status is not None and raised_in_provider(e):
                return failure("malformed_response", status=status, reason=failure_reason(e))
        return failure("internal", reason=failure_reason(chain[0], with_cause=True))
    if facts.refusal or finish == "content_filter":
        return failure("refused", reason=f"the model declined (finish_reason={facts.finish_raw})")
    if finish == "length":
        return failure("truncated", reason="finish_reason=length: the token ceiling cut the reply")
    return None


__all__ = [
    "CALLER_CONFIG_ERRORS",
    "ERROR_MESSAGE_MAX",
    "OVERFLOW_CODES",
    "OVERFLOW_MESSAGES",
    "PROVIDER_MODULES",
    "SEVERITY",
    "STATUS_KINDS",
    "Categorize",
    "Classifier",
    "WireFacts",
    "classify_core",
    "is_context_overflow",
    "kind_for_status",
    "raised_in_provider",
    "retry_after_s",
    "worst",
]
