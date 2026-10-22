"""failure -- how a failed model call is NAMED: the one persisted reason line (pure).

A provider's error payload is a log line's business, not a ledger row's: a 300-char
`ModelHTTPError: status_code: 500, model_name: ..., body: {'message': "li...` once reached an
operator feed through a reason field, one row per tick. And a proxy error can embed a keyed URL.
`failure_reason` keeps the class, the status code and the head of the message -- never the body.
"""

from __future__ import annotations

from collections.abc import Iterator

REASON_MAX = 200


class GenerationFailed(RuntimeError):
    """The facade's non-ok, non-gated outcome, RAISED for a caller whose contract is that a failed
    generation propagates (the Citer port; the `except Exception` ledgers of the structured family
    that record `ok=false` rows). `outcome` and `reason` ride on it so the row can name both."""

    def __init__(self, outcome: str, reason: str | None) -> None:
        super().__init__(f"{outcome}: {reason or ''}".rstrip(": "))
        self.outcome = outcome
        self.reason = reason


def failure_reason(exc: BaseException) -> str:
    """Exception class, the status code when the exception carries one, and the first line of its
    message cut BEFORE any `body:` payload -- never the body."""
    cls = type(exc).__name__
    code = getattr(exc, "status_code", None)
    head = (str(exc).splitlines() or [""])[0]
    head = head.split("body:", 1)[0].split("body=", 1)[0].strip().rstrip(",")
    if code is not None:
        return f"{cls}: status {code}"[:REASON_MAX]
    return (f"{cls}: {head}" if head else cls)[:REASON_MAX]


def exception_chain(exc: BaseException, limit: int = 12) -> Iterator[BaseException]:
    """`exc`, then its `__cause__` (else `__context__`), and so on: the SDK layers (pydantic-ai ->
    openai -> httpx -> our transport) each wrap the one below, so the cause is never on top."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(seen) < limit:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


__all__ = ["REASON_MAX", "GenerationFailed", "exception_chain", "failure_reason"]
