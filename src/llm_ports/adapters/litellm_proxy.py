"""What a LiteLLM proxy in front of the model says about a call, read from its response.

A LiteLLM proxy knows things the response body does not: which deployment served an alias
(`x-litellm-model-id`, `x-litellm-model-api-base`), what the call cost at that deployment's price
(`x-litellm-response-cost`), and how many retries and fallbacks it spent
(`x-litellm-attempted-retries`, `x-litellm-attempted-fallbacks`). It also refuses a call in two
ways that look alike on the wire: a 429 because its parallel-request limit is full (comes back in
seconds), and a 429 because a budget or an upstream quota is spent (does not).

The module satisfies `ports.GatewayReader` (pass the module itself as `gateway=`), and
`classify_gateway` is an `extra` classifier for the generator. It imports no LiteLLM and no HTTP
client: it reads header mappings and `WireFacts`.

Invariants:
    - Pure: no I/O.
    - A cost header that is present but not a plausible dollar amount (a finite number, 0 or
      between `COST_MIN_USD` and `COST_MAX_USD`) raises `ValueError` from `cost_from_headers`;
      the generator treats it as absent and reports it.
    - `classify_gateway` names only `quota`; a plain 429 is left to the shared table
      (`rate_limited`, retryable, with the server's `Retry-After`).

Example:
    gen = PydanticAIGenerator(model_for=openai_compatible(url, key), gateway=litellm_proxy,
                              extra=(litellm_proxy.classify_gateway,))

Rationale: ADR-0019, ADR-0020.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

from llm_ports.adapters._classify import WireFacts
from llm_ports.ports import Failure, failure
from llm_ports.ports.outcomes import exception_chain

#: The deployment id the proxy routed the call to.
MODEL_ID_HEADER = "x-litellm-model-id"
#: The upstream base URL of that deployment.
API_BASE_HEADER = "x-litellm-model-api-base"
#: The call's cost in US dollars at the deployment's price.
COST_HEADER = "x-litellm-response-cost"
#: Retries the proxy spent on the call.
RETRIES_HEADER = "x-litellm-attempted-retries"
#: Fallback deployments the proxy tried.
FALLBACKS_HEADER = "x-litellm-attempted-fallbacks"

#: The largest cost one call may report, in US dollars; at or past it the header is implausible
#: (`1e9999999` overflows any sum).
COST_MAX_USD = Decimal("1e7")
#: The smallest non-zero cost one call may report; below it the header is implausible.
COST_MIN_USD = Decimal("1e-12")

#: A provider's own retry hint past this many seconds is a spent quota, not a queue: it resets on
#: the provider's clock.
QUOTA_HINT_MAX_S = 120.0

_PER_DAY = re.compile(r"per\s*day", re.IGNORECASE)
_RETRY_IN = re.compile(r"retry in\s+(?=\d)(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", re.IGNORECASE)
_TEXT_MAX = 2000


def _get(headers: Mapping[str, str], name: str) -> str | None:
    v = headers.get(name)
    if v is None:
        v = next((val for k, val in headers.items() if k.lower() == name), None)
    v = v.strip() if v is not None else None
    return v or None


def cost_from_headers(headers: Mapping[str, str]) -> Decimal | None:
    """Return the cost the proxy reported (`x-litellm-response-cost`), or None when it reported none.

    Pure.

    A reported zero (`0`, `-0`, `0.000`) is `Decimal(0)`.

    Raises:
        ValueError: The header is present but not a finite number of dollars that is 0 or between
            `COST_MIN_USD` and `COST_MAX_USD` (exclusive of the maximum).
    """
    raw = _get(headers, COST_HEADER)
    if raw is None:
        return None
    try:
        usd = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{COST_HEADER} is not a number: {raw[:40]!r}") from exc
    if not usd.is_finite():
        raise ValueError(f"{COST_HEADER} is not a finite number: {raw[:40]!r}")
    if usd.is_zero():
        return Decimal(0)
    if usd < 0 or not COST_MIN_USD <= usd < COST_MAX_USD:
        raise ValueError(f"{COST_HEADER} is not a plausible cost in dollars: {raw[:40]!r}")
    return usd


def answered_from_headers(headers: Mapping[str, str]) -> tuple[str | None, str | None]:
    """Return `(deployment id, upstream api base)` as the proxy reported them; None for each it did not. Pure."""
    return _get(headers, MODEL_ID_HEADER), _get(headers, API_BASE_HEADER)


def _int(headers: Mapping[str, str], name: str) -> int | None:
    raw = _get(headers, name)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def attempts_from_headers(headers: Mapping[str, str]) -> tuple[int | None, int | None]:
    """Return `(retries, fallbacks)` the proxy spent on the call; None for each it did not report. Pure."""
    return _int(headers, RETRIES_HEADER), _int(headers, FALLBACKS_HEADER)


def retry_hint_s(text: str) -> float | None:
    """Return a provider's "Please retry in 8h26m23.4s" hint in seconds; None when the text has none. Pure."""
    m = _RETRY_IN.search(text)
    if not m or not any(m.groups()):
        return None
    h, mins, secs = m.groups()
    hint = int(h or 0) * 3600 + int(mins or 0) * 60 + float(secs or 0)  # zero-ok: an absent unit is 0 of it
    return hint if math.isfinite(hint) else None


def _text(exc: BaseException | None, facts: WireFacts) -> str:
    """The provider's error text: the wire's error fields and every error body in the chain."""
    parts = [facts.error_message or "", facts.error_type or "", facts.error_code or ""]
    for e in exception_chain(exc) if exc is not None else ():
        body = getattr(e, "body", None)
        if isinstance(body, str):
            parts.append(body)
        elif body is not None:
            parts.append(json.dumps(body, default=str))
    return " ".join(parts)[:_TEXT_MAX]


def _status(exc: BaseException | None, facts: WireFacts) -> int | None:
    for e in exception_chain(exc) if exc is not None else ():
        code = getattr(e, "status_code", None)
        if isinstance(code, int):
            return code
    return facts.status


def classify_gateway(exc: BaseException | None, facts: WireFacts) -> Failure | None:
    """Name a 429 that no wait will clear as `quota`; leave everything else to the shared table.

    A 429 is `quota` when its error names a budget (the proxy's budget for a key, team or
    provider), a per-day limit, or asks for a retry longer than `QUOTA_HINT_MAX_S` (an upstream
    provider's spent quota). A 429 from the proxy's parallel-request limit is left alone: the
    shared table makes it `rate_limited`, retryable, with the server's `Retry-After`. Pure.

    Rationale: ADR-0020.
    """
    if _status(exc, facts) != 429:
        return None
    text = _text(exc, facts)
    hint = retry_hint_s(text)
    if "budget" in text.lower() or _PER_DAY.search(text) or (hint is not None and hint > QUOTA_HINT_MAX_S):
        return failure(
            "quota",
            status=429,
            provider_code=facts.error_type or facts.error_code,
            reason="HTTP 429: a budget or quota is spent",
        )
    return None


__all__ = [
    "API_BASE_HEADER",
    "COST_HEADER",
    "FALLBACKS_HEADER",
    "MODEL_ID_HEADER",
    "QUOTA_HINT_MAX_S",
    "RETRIES_HEADER",
    "answered_from_headers",
    "attempts_from_headers",
    "classify_gateway",
    "cost_from_headers",
    "retry_hint_s",
]
