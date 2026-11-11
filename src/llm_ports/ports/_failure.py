"""What went wrong with a failed call: a `Failure` carries the kind under the coarse outcome.

The seven outcomes are the caller's decision surface (ADR-0001); a `Failure` answers the three
questions a caller, a retry policy and a fallback chain share: what kind, is it worth retrying,
and after how long. `kind` decides `outcome` and `retryable`; nothing sets them by hand.

Invariants:
    - Pure: no I/O.
    - `OUTCOME_FOR` is total over `FailureKind`; `RETRYABLE` is a subset of it.
    - `reason` never carries a response body and never exceeds `outcomes.REASON_MAX` characters.

Rationale: ADR-0020.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, get_args

from llm_ports.ports.outcomes import REASON_MAX, Outcome

#: Every kind of failure, each under exactly one outcome (`OUTCOME_FOR`).
FailureKind = Literal[
    "gated",
    "rate_limited",
    "quota",
    "overloaded",
    "server",
    "timeout",
    "transport",
    "auth",
    "not_found",
    "malformed_response",
    "caller_limit",
    "bad_request",
    "context_overflow",
    "unsupported",
    "internal",
    "truncated",
    "invalid_output",
    "refused",
]

#: Where a transport or deadline failure happened: httpx's four phases, the first streamed event,
#: or a bound over the whole call.
Phase = Literal["pool", "connect", "write", "read", "first_event", "total"]

#: The outcome each kind sits under.
OUTCOME_FOR: Mapping[FailureKind, Outcome] = {
    "gated": "gated",
    "rate_limited": "unavailable",
    "quota": "unavailable",
    "overloaded": "unavailable",
    "server": "unavailable",
    "timeout": "unavailable",
    "transport": "unavailable",
    "auth": "unavailable",
    "not_found": "unavailable",
    "malformed_response": "unavailable",
    "caller_limit": "unavailable",
    "bad_request": "request_fault",
    "context_overflow": "request_fault",
    "unsupported": "request_fault",
    "internal": "request_fault",
    "truncated": "truncated",
    "invalid_output": "invalid",
    "refused": "refused",
}

#: The kinds a retry can help: the server, not the request, was the problem, and it comes back in
#: seconds. `quota` and `caller_limit` are not here: a spent budget does not come back within a
#: backoff, and neither does a limit the caller set.
RETRYABLE: frozenset[FailureKind] = frozenset(
    {"rate_limited", "overloaded", "server", "timeout", "transport", "malformed_response"}
)

#: Every `FailureKind`, in declaration order.
FAILURE_KINDS: tuple[FailureKind, ...] = get_args(FailureKind)


@dataclass(frozen=True, slots=True)
class Failure:
    """One failed call's kind and the facts a retry or fallback decision reads.

    Attributes:
        kind: What went wrong.
        outcome: The coarse outcome; derived from `kind` by `OUTCOME_FOR`, never passed in.
        retryable: Whether a retry can help; derived from `kind` by `RETRYABLE`, never passed in.
        retry_after_s: The server's own wait hint, in seconds, when it gave one.
        status: The HTTP status, when there was one.
        provider_code: The provider's error code or type string, verbatim.
        phase: Where a transport or deadline failure happened, when known.
        in_stream: True when the failure came after output had started.
        reason: The one-line reason a caller persists (the 1.x `reason`).

    Invariants:
        - `outcome == OUTCOME_FOR[kind]` and `retryable == (kind in RETRYABLE)`.
        - `reason` is at most `outcomes.REASON_MAX` characters.

    Rationale: ADR-0020.
    """

    kind: FailureKind
    outcome: Outcome = field(init=False)
    retryable: bool = field(init=False)
    retry_after_s: float | None = None
    status: int | None = None
    provider_code: str | None = None
    phase: Phase | None = None
    in_stream: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        """Derive `outcome` and `retryable` from `kind`, and cap `reason`.

        Raises:
            ValueError: `kind` is not a `FailureKind`.
        """
        if self.kind not in OUTCOME_FOR:
            raise ValueError(f"unknown failure kind {self.kind!r}")
        object.__setattr__(self, "outcome", OUTCOME_FOR[self.kind])
        object.__setattr__(self, "retryable", self.kind in RETRYABLE)
        object.__setattr__(self, "reason", self.reason[:REASON_MAX])


def failure(
    kind: FailureKind,
    *,
    retry_after_s: float | None = None,
    status: int | None = None,
    provider_code: str | None = None,
    phase: Phase | None = None,
    in_stream: bool = False,
    reason: str = "",
) -> Failure:
    """Return the `Failure` of `kind` with the given detail; the one constructor `src/` uses.

    Pure. The arguments are the `Failure` attributes of the same name; `reason` defaults to the
    kind itself, so a failure always names something.

    Raises:
        ValueError: `kind` is not a `FailureKind`.
    """
    return Failure(
        kind,
        retry_after_s=retry_after_s,
        status=status,
        provider_code=provider_code,
        phase=phase,
        in_stream=in_stream,
        reason=reason or kind,
    )


__all__ = ["FAILURE_KINDS", "OUTCOME_FOR", "RETRYABLE", "Failure", "FailureKind", "Phase", "failure"]
