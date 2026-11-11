"""Failure classification for the pydantic-ai adapter: its error categories over the shared table.

`classify_failure` runs the shared order (`adapters._classify.classify_core`) with pydantic-ai's
own categories as step 4, walking the whole exception chain:

| pydantic-ai | kind |
|---|---|
| `ModelRateLimitError` | `rate_limited` |
| `ModelQuotaExceededError` | `quota` |
| `ModelOverloadedError` | `overloaded` |
| `ModelContextWindowExceededError` | `context_overflow` |
| `ModelAuthenticationError` | `auth` |
| `ModelNotFoundError` | `not_found` |
| `ModelTimeoutError`, `ModelRequestTimeout`, `RunTimedOut` | `timeout`, with its phase |
| `RunCancelled` (the application cancelled the run), `UsageLimitExceeded` | `caller_limit` |
| `ConcurrencyLimitExceeded` (the agent's queue is full) | `overloaded` |
| `ModelConnectionError` | `transport`, with its phase |
| `ModelServerError`, `ModelUnavailableError` | `server` |
| `ContentFilterError` | `refused` |
| `ModelRequestRejectedError` (other) | `bad_request`, or `context_overflow` when its code says so |
| `UnexpectedModelBehavior` (other) | `refused` on a refusal, `truncated` on a length finish, else `invalid_output` |
| `ModelAPIError` with no category and no status | `server` |

`openai_finish` is the one mapping from an OpenAI-wire `finish_reason` to `Finish.unified`.

Invariants:
    - Pure: no I/O; never raises except to re-raise a cancellation.

Rationale: ADR-0020.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from llm_ports._extras import extra
from llm_ports.adapters._classify import Classifier, WireFacts, classify_core, is_context_overflow
from llm_ports.ports import Failure, FailureKind, Outcome, Phase, UnifiedFinish, failure
from llm_ports.ports.outcomes import failure_reason

with extra("pydantic-ai"):
    from pydantic_ai.exceptions import (
        ConcurrencyLimitExceeded,
        ContentFilterError,
        ModelAPIError,
        ModelAuthenticationError,
        ModelConnectionError,
        ModelContextWindowExceededError,
        ModelHTTPError,
        ModelNotFoundError,
        ModelOverloadedError,
        ModelQuotaExceededError,
        ModelRateLimitError,
        ModelRequestRejectedError,
        ModelRequestTimeout,
        ModelServerError,
        ModelTimeoutError,
        ModelUnavailableError,
        RunCancelled,
        RunTimedOut,
        UnexpectedModelBehavior,
        UsageLimitExceeded,
    )

#: The OpenAI-wire finish reasons and what each means; anything else is `other`. `tool_call` is
#: pydantic-ai's own reading, met when a response came through the framework rather than the wire.
OPENAI_FINISH: Mapping[str, UnifiedFinish] = {
    "stop": "stop",
    "tool_call": "tool_call",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "error": "error",
}

#: pydantic-ai's error categories, most specific first (a subclass precedes its base).
_CATEGORIES: tuple[tuple[type[BaseException], FailureKind], ...] = (
    (ModelRateLimitError, "rate_limited"),
    (ModelQuotaExceededError, "quota"),
    (ModelOverloadedError, "overloaded"),
    (ModelContextWindowExceededError, "context_overflow"),
    (ModelAuthenticationError, "auth"),
    (ModelNotFoundError, "not_found"),
    (ModelTimeoutError, "timeout"),
    (ModelRequestTimeout, "timeout"),
    (RunTimedOut, "timeout"),
    (RunCancelled, "caller_limit"),
    (UsageLimitExceeded, "caller_limit"),
    (ConcurrencyLimitExceeded, "overloaded"),
    (ModelConnectionError, "transport"),
    (ModelServerError, "server"),
    (ModelUnavailableError, "server"),
    (ContentFilterError, "refused"),
    (ModelRequestRejectedError, "bad_request"),
)


def openai_finish(raw: str | None) -> UnifiedFinish | None:
    """Return the unified reading of an OpenAI-wire `finish_reason`; None when there was none. Pure."""
    if raw is None:
        return None
    return OPENAI_FINISH.get(raw, "other")


def _phase(e: BaseException) -> Phase | None:
    if isinstance(e, RunTimedOut):
        return "total"
    if isinstance(e, ModelRequestTimeout):
        return "first_event" if getattr(e, "setting", None) == "first_event_timeout" else "total"
    phase = getattr(e, "phase", None)
    return phase if phase in ("pool", "connect", "write", "read") else None


def _from_category(e: BaseException, kind: FailureKind, facts: WireFacts) -> Failure:
    status = e.status_code if isinstance(e, ModelHTTPError) else facts.status
    code = getattr(e, "provider_error_code", None) or getattr(e, "provider_error_type", None)
    if kind == "bad_request" and is_context_overflow([code, facts.error_code, facts.error_type], [facts.error_message]):
        kind = "context_overflow"
    hint = getattr(e, "retry_after", None)
    return failure(
        kind,
        status=status,
        provider_code=code or facts.error_code or facts.error_type,
        retry_after_s=float(hint) if isinstance(hint, int | float) else None,
        phase=_phase(e) if kind in ("timeout", "transport") else None,
        in_stream=bool(getattr(e, "in_stream", False)),
        reason=failure_reason(e),
    )


def categorize(chain: Sequence[BaseException], facts: WireFacts, finish: UnifiedFinish | None) -> Failure | None:
    """Step 4 of the shared order: pydantic-ai's error categories, walking the whole chain. Pure."""
    for e in chain:
        for cls, kind in _CATEGORIES:
            if isinstance(e, cls):
                return _from_category(e, kind, facts)
    for e in chain:
        if isinstance(e, UnexpectedModelBehavior):
            if facts.refusal or finish == "content_filter":
                return failure("refused", reason=failure_reason(e))
            if finish == "length":
                return failure("truncated", reason="finish_reason=length: the token ceiling cut the reply")
            return failure("invalid_output", reason=failure_reason(e))
    for e in chain:
        if isinstance(e, ModelAPIError) and not isinstance(e, ModelHTTPError):
            return _from_category(e, "server", facts)
    return None


def classify_failure(
    exc: BaseException | None,
    facts: WireFacts,
    *,
    extra: Sequence[Classifier] = (),
    gated: tuple[type[BaseException], ...] = (),
) -> Failure | None:
    """Return the failure a pydantic-ai call ended with, or None when it succeeded.

    Args:
        exc: The exception the run raised; None when it returned.
        facts: What the wire showed of the last response (its finish reason, status, error code).
        extra: Classifiers consulted first, in order (e.g. `adapters.litellm_proxy.classify_gateway`).
        gated: Exception types that mean the caller's gate refused the call.

    Raises:
        asyncio.CancelledError: Re-raised when `exc` is one.

    Rationale: ADR-0020.
    """
    return classify_core(
        exc, facts, finish=openai_finish(facts.finish_raw), extra=extra, gated=gated, categorize=categorize
    )


def classify(
    exc: BaseException, finish_reason: str | None = None, *, gated: tuple[type[BaseException], ...] = ()
) -> tuple[Outcome, str]:
    """Name the outcome a failed run maps to, with a one-line reason (the 1.x signature).

    A thin wrapper over `classify_failure`, which also returns the kind and its detail. Pure.

    Args:
        exc: The exception the run raised.
        finish_reason: The last reply's `finish_reason`, when the wire reported one.
        gated: Exception types that mean the caller's gate refused the call.

    Returns:
        The outcome and its reason.

    Rationale: ADR-0001, ADR-0020.
    """
    got = classify_failure(exc, WireFacts(finish_raw=finish_reason), gated=gated)
    assert got is not None, "an exception always classifies"  # step 9 catches everything
    return got.outcome, got.reason


__all__ = ["OPENAI_FINISH", "classify", "classify_failure", "openai_finish"]
