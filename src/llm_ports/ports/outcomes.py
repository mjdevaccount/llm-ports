"""Named failure reasons for a failed model call.

`failure_reason` turns an exception into the one reason line a caller persists: the exception
class, the status code when there is one, and the head of the message. A provider's error payload
belongs in a log line, not in a persisted reason, and a proxy error can embed a keyed URL.

Invariants:
    - Pure: no I/O.
    - A reason line never carries the response body and never exceeds `REASON_MAX` characters.

Rationale: ADR-0001.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

REASON_MAX = 200


class GenerationFailed(RuntimeError):
    """A non-ok, non-gated outcome raised for a caller whose contract is that a failure propagates.

    The facade raises it for a caller that expects an exception (a `Citer`, or a caller that records
    a failed row from an `except` block).

    Attributes:
        outcome: The `Outcome` the call ended with.
        reason: The one-line reason, or None.
    """

    def __init__(self, outcome: str, reason: str | None) -> None:
        super().__init__(f"{outcome}: {reason or ''}".rstrip(": "))
        self.outcome = outcome
        self.reason = reason


def failure_reason(exc: BaseException, *, with_cause: bool = False) -> str:
    """Return the persisted reason line for `exc`.

    The line is the exception class, the status code when the exception carries one, and the
    first line of its message cut before any `body:` payload.

    Args:
        exc: The exception to name.
        with_cause: Append ` <- <the cause's reason>`, cut by the same rule. A wrapper's own line
            can name nothing ("Exceeded maximum output retries") while its cause ("Invalid JSON:
            EOF while parsing a string") names the fix.

    Returns:
        A line of at most `REASON_MAX` characters; the cause keeps up to half of it.
    """
    head = _reason(exc)
    cause = next(islice(exception_chain(exc), 1, None), None) if with_cause else None
    if cause is None:
        return head
    tail = _reason(cause)[: REASON_MAX // 2]
    return f"{head[: REASON_MAX - len(tail) - 4]} <- {tail}"


def _reason(exc: BaseException) -> str:
    cls = type(exc).__name__
    code = getattr(exc, "status_code", None)
    head = (str(exc).splitlines() or [""])[0]
    head = head.split("body:", 1)[0].split("body=", 1)[0].strip().rstrip(",")
    if code is not None:
        return f"{cls}: status {code}"[:REASON_MAX]
    return (f"{cls}: {head}" if head else cls)[:REASON_MAX]


def exception_chain(exc: BaseException, limit: int = 12) -> Iterator[BaseException]:
    """Yield `exc`, then its `__cause__` (else `__context__`), and so on down the chain.

    SDK layers (pydantic-ai, openai, httpx, a transport) each wrap the one below, so the root
    cause is never on top.

    Args:
        exc: The outermost exception.
        limit: The most exceptions to yield. A cycle ends the chain early.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(seen) < limit:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


__all__ = ["REASON_MAX", "GenerationFailed", "exception_chain", "failure_reason"]
