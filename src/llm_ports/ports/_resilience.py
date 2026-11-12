"""The wire's value types: how a backend is retried, how long a call may take, why a slot was refused.

`GenRequest` carries a `WirePolicy` and a `Deadline`, so both live with the port types; the call
layer re-exports them (`llm_ports.call.resilience`), beside the in-process `LocalLimiter`.

Invariants:
    - Pure: no I/O, no vendor import, no clock read except where a method takes `now`.
    - A `WirePolicy` names only kinds in `RETRYABLE`; `quota`, `auth` and every request-side kind
      can never be retried, whatever a caller writes.

Rationale: ADR-0021.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from llm_ports.ports._failure import RETRYABLE, FailureKind


@dataclass(frozen=True, slots=True)
class RetryRule:
    """How one failure kind is retried.

    Attributes:
        attempts: Tries of one wire request for this kind, the first included; 1 means no retry.
        wait_initial_s: The wait before the second try, in seconds.
        wait_max_s: The cap on one wait, in seconds.
        multiplier: Each later wait is the previous one times this.
        jitter: A fraction of each wait added or removed at random; 0 disables it.
        honour_retry_after: Wait what the server's `retry-after-ms` / `Retry-After` says instead of
            the computed wait.
        retry_after_max_s: A server hint longer than this ends the retry; the `Failure` keeps the
            hint so the caller can reschedule.

    Raises:
        ValueError: `attempts` is below 1, a wait is negative, the cap is below the initial wait,
            `multiplier` is below 1, or `jitter` is outside [0, 1].
    """

    attempts: int
    wait_initial_s: float
    wait_max_s: float
    multiplier: float = 2.0
    jitter: float = 0.1
    honour_retry_after: bool = True
    retry_after_max_s: float = 120.0

    def __post_init__(self) -> None:
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 1:
            raise ValueError("RetryRule: attempts must be an int >= 1")
        if not 0 <= self.wait_initial_s <= self.wait_max_s or not math.isfinite(self.wait_max_s):
            raise ValueError("RetryRule: need 0 <= wait_initial_s <= wait_max_s, both finite")
        if self.multiplier < 1 or not 0 <= self.jitter <= 1 or self.retry_after_max_s < 0:
            raise ValueError("RetryRule: need multiplier >= 1, 0 <= jitter <= 1, retry_after_max_s >= 0")

    def wait_s(self, n: int, *, spread: float = 0.0) -> float:
        """The wait after the `n`-th failed try (from 1), capped, with `spread` in [-1, 1] of jitter. Pure."""
        base = min(self.wait_max_s, self.wait_initial_s * self.multiplier ** max(0, n - 1))
        return max(0.0, base * (1 + self.jitter * spread))


#: The fields a `[policies.<name>]` rule may set, and the keys a policy may set beside its rules.
_RULE_KEYS = frozenset(RetryRule.__slots__)
_POLICY_KEYS = frozenset({"max_total_attempts"})


@dataclass(frozen=True, slots=True)
class WirePolicy:
    """Which failure kinds one wire request is retried on, and how.

    Attributes:
        rules: Failure kind to its `RetryRule`; a kind not listed is not retried.
        max_total_attempts: Wire requests one call may make across every kind and every request, a
            hard ceiling.

    Raises:
        ValueError: A rule's key is not in `RETRYABLE`, or `max_total_attempts` is below 1.

    Rationale: ADR-0021.
    """

    rules: Mapping[FailureKind, RetryRule] = field(default_factory=dict)
    max_total_attempts: int = 60

    def __post_init__(self) -> None:
        bad = sorted(k for k in self.rules if k not in RETRYABLE)
        if bad:
            raise ValueError(f"WirePolicy: {bad} cannot be retried (retryable: {sorted(RETRYABLE)})")
        if isinstance(self.max_total_attempts, bool) or self.max_total_attempts < 1:
            raise ValueError("WirePolicy: max_total_attempts must be >= 1")
        object.__setattr__(self, "rules", MappingProxyType(dict(self.rules)))

    def __hash__(self) -> int:
        """Hash by value, so a frozen `Role` or `Tier` carrying a policy stays hashable."""
        return hash((tuple(sorted(self.rules.items())), self.max_total_attempts))

    @classmethod
    def default(cls) -> WirePolicy:
        """`rate_limited` 3 tries; `overloaded`, `server`, `transport`, `timeout` 2 tries; waits 0.5-8 s."""
        busy = RetryRule(attempts=2, wait_initial_s=0.5, wait_max_s=8.0)
        return cls(
            rules={
                "rate_limited": RetryRule(attempts=3, wait_initial_s=0.5, wait_max_s=8.0),
                "overloaded": busy,
                "server": busy,
                "transport": busy,
                "timeout": busy,
            }
        )

    @classmethod
    def none(cls) -> WirePolicy:
        """No retry: every request is sent once (an interactive first-token path)."""
        return cls(rules={})

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, path: str = "policy") -> WirePolicy:
        """Read a policy written as data: `{<kind>: {<RetryRule field>: value}, max_total_attempts: n}`.

        Args:
            raw: The policy, e.g. a tier table's `[policies.<name>]` section.
            path: Where the policy sits, named in an error.

        Raises:
            ValueError: An unknown or non-retryable kind, an unknown field, or a value a
                `RetryRule` refuses; the message starts with the offending path.
        """
        rules: dict[FailureKind, RetryRule] = {}
        total = 60
        for key, value in raw.items():
            where = f"{path}.{key}"
            if key in _POLICY_KEYS:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{where}: must be an integer, not {value!r}")
                total = value
                continue
            if key not in RETRYABLE:
                raise ValueError(f"{where}: not a retryable failure kind (retryable: {sorted(RETRYABLE)})")
            if not isinstance(value, Mapping):
                raise ValueError(f"{where}: must be a table of RetryRule fields, not {type(value).__name__}")
            unknown = sorted(set(value) - _RULE_KEYS)
            if unknown:
                raise ValueError(f"{where}.{unknown[0]}: unknown field (allowed: {sorted(_RULE_KEYS)})")
            try:
                rules[key] = RetryRule(**dict(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where}: {exc}") from None
        try:
            return cls(rules=rules, max_total_attempts=total)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None


@dataclass(frozen=True, slots=True)
class Deadline:
    """How long one call may take, on the event loop's monotonic clock.

    Attributes:
        total_s: The whole call: slot wait, every wire request and re-ask, every backoff, every
            fallback tier.
        attempt_s: One wire request; None means `total_s`.
        first_event_s: Time to the first streamed event; None means not enforced.
        started: When the call started, on the loop's clock; None until `start` stamps it.

    Raises:
        ValueError: A bound is not a positive finite number.

    Rationale: ADR-0021.
    """

    total_s: float
    attempt_s: float | None = None
    first_event_s: float | None = None
    started: float | None = None

    def __post_init__(self) -> None:
        for name in ("total_s", "attempt_s", "first_event_s"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v <= 0):
                raise ValueError(f"Deadline: {name} must be a positive finite number of seconds, not {v!r}")

    def start(self, now: float) -> Deadline:
        """This deadline, started at `now`; one already started is returned as is."""
        if self.started is not None:
            return self
        return Deadline(self.total_s, self.attempt_s, self.first_event_s, now)

    @property
    def at(self) -> float:
        """When the call must be over, on the loop's clock.

        Raises:
            ValueError: The deadline was never started.
        """
        if self.started is None:
            raise ValueError("Deadline.at: start() it first")
        return self.started + self.total_s

    def remaining(self, now: float) -> float:
        """Seconds left at `now`, never below 0; `total_s` when not started."""
        if self.started is None:
            return self.total_s
        return max(0.0, self.started + self.total_s - now)

    def attempt_bound(self, now: float) -> float:
        """The bound on one wire request started at `now`: `attempt_s`, never past the deadline."""
        left = self.remaining(now)
        return left if self.attempt_s is None else min(self.attempt_s, left)


class SlotWaitExceeded(Exception):
    """Raised by a `Limiter` whose wait for a slot passed its `max_wait_s`.

    An adapter turns it into `rate_limited` with `provider_code="local_limiter"`, not retried.

    Attributes:
        group: The concurrency group.
        waited_s: How long the call waited, in seconds.
    """

    def __init__(self, group: str, waited_s: float) -> None:
        super().__init__(f"no {group!r} slot within {waited_s:.3g} s")
        self.group = group
        self.waited_s = waited_s


__all__ = ["Deadline", "RetryRule", "SlotWaitExceeded", "WirePolicy"]
