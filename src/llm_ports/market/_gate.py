"""Cached control reads: a last-known value, and the per-category gate `call.generate` admits by.

A control plane (a global work switch, a per-category switchboard with call caps) lives in the
consumer's store. Reading it on every call is too slow and a failed read must not flip a working
pipeline off, so both reads are cached for `ttl_s` and a failed read serves the last known value.

`GateCache.admit` has the signature of `call.Hooks.admit`, so a switched-off category returns
`gated` before any trace opens:

    gates = GateCache(read_gates, window="daily")
    hooks = call.Hooks(admit=gates.admit, trace=call.tracer_trace(tracer))

Invariants:
    - `LastKnown.get` and `GateCache.refresh` never raise: a failed read keeps the last known value,
      is emitted as `market_read_failed` (WARNING), and still marks the value fresh, so the next
      read waits one TTL instead of retrying in a hot loop.
    - `GateCache.check` and `admit` never read the store: they judge the last cached gates. An
      unknown category, or a cache never filled, is admitted (the gate fails open).
    - A cap is enforced against the cached use count, so a burst can overshoot it by up to one TTL.

Rationale: ADR-0018.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from llm_ports.call.roles import Role
from llm_ports.observability import EventHook, log_events

_log_event = log_events("llm_ports.market")

#: The reason prefix of a switched-off category (machine-readable: a managed skip, not an error).
CATEGORY_DISABLED = "category_disabled"
#: The reason prefix of a category over its call cap.
CATEGORY_CAPPED = "category_capped"


class LastKnown[T]:
    """A value read through `read` at most once per `ttl_s`, serving the last known one on a failure.

    Args:
        read: The store read.
        initial: The value served until a read succeeds.
        name: Names the value in `market_read_failed`.
        ttl_s: Seconds a read (or a failed read) stays fresh.
        clock: Monotonic seconds; injectable for tests.
        on_event: Receives `market_read_failed`; without one, the stdlib logger `llm_ports.market`.
    """

    __slots__ = ("_clock", "_emit", "_fresh_until", "_read", "_value", "name", "ttl_s")

    def __init__(
        self,
        read: Callable[[], Awaitable[T]],
        initial: T,
        *,
        name: str = "value",
        ttl_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventHook | None = None,
    ) -> None:
        self.name: str = name
        self.ttl_s: float = ttl_s
        self._read = read
        self._value = initial
        self._clock = clock
        self._emit = on_event or _log_event
        self._fresh_until: float | None = None

    @property
    def value(self) -> T:
        """The last known value, without a read."""
        return self._value

    @property
    def fresh(self) -> bool:
        """Whether the last read (or failed read) is younger than `ttl_s`."""
        return self._fresh_until is not None and self._clock() < self._fresh_until

    async def get(self) -> T:
        """The value: cached while fresh, else read; the last known one when the read fails. Never raises."""
        if self.fresh:
            return self._value
        try:
            self._value = await self._read()
        except Exception as exc:
            self._emit(
                logging.WARNING,
                "market_read_failed",
                {"value": self.name, "reason": f"{type(exc).__name__}: {exc}"[:200], "serving": "last_known"},
            )
        self._fresh_until = self._clock() + self.ttl_s
        return self._value

    def prime(self, value: T) -> None:
        """Set the value and mark it fresh (a warm start, a test)."""
        self._value = value
        self._fresh_until = self._clock() + self.ttl_s

    def invalidate(self) -> None:
        """Mark the value stale; the next `get` reads."""
        self._fresh_until = None


@dataclass(frozen=True)
class Gate:
    """One category's switch.

    Attributes:
        enabled: False switches the category off.
        call_cap: The most calls admitted per window; None for no cap.
        used: Calls already admitted this window, as of the read; None when unknown (the cap is
            then not enforced by the cached check).
    """

    enabled: bool = True
    call_cap: int | None = None
    used: int | None = None


_EMPTY: Mapping[str, Gate] = MappingProxyType({})


class GateCache:
    """The per-category switchboard, cached: `refresh` reads it, `check` and `admit` judge it.

    Args:
        read: Returns every seeded category's `Gate`.
        window: Names the cap's window in the refusal (`"daily"`); None leaves it out.
        ttl_s: Seconds a read stays fresh.
        clock: Monotonic seconds; injectable for tests.
        on_event: Receives `market_read_failed`; without one, the stdlib logger `llm_ports.market`.
    """

    __slots__ = ("_cached", "_window")

    def __init__(
        self,
        read: Callable[[], Awaitable[Mapping[str, Gate]]],
        *,
        window: str | None = None,
        ttl_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventHook | None = None,
    ) -> None:
        self._cached: LastKnown[Mapping[str, Gate]] = LastKnown(
            read, _EMPTY, name="gates", ttl_s=ttl_s, clock=clock, on_event=on_event
        )
        self._window = f"{window} " if window else ""

    @property
    def gates(self) -> Mapping[str, Gate]:
        """The last cached gates, without a read."""
        return self._cached.value

    async def refresh(self) -> Mapping[str, Gate]:
        """The gates, read when the cache is stale. Never raises."""
        return await self._cached.get()

    def prime(self, gates: Mapping[str, Gate]) -> None:
        """Set the cached gates and mark them fresh (a warm start, a test)."""
        self._cached.prime(dict(gates))

    def invalidate(self) -> None:
        """Mark the cache stale; the next `refresh` reads."""
        self._cached.invalidate()

    def check(self, category: str | None) -> str | None:
        """Judge `category` against the cached gates: the refusal reason, or None to admit. Never raises.

        No category, an unknown one, or an empty cache is admitted.
        """
        if not category:
            return None
        gate = self._cached.value.get(category)
        if gate is None:
            return None
        if not gate.enabled:
            return f"{CATEGORY_DISABLED}: {category} is switched off"
        if gate.call_cap is not None and gate.used is not None and gate.used >= gate.call_cap:
            return f"{CATEGORY_CAPPED}: {category} hit its {self._window}call cap ({gate.call_cap})"
        return None

    def admit(self, role: Role) -> str | None:
        """`call.Hooks.admit`: judge the role's category (`check`). Never raises."""
        return self.check(role.category)


__all__ = ["CATEGORY_CAPPED", "CATEGORY_DISABLED", "Gate", "GateCache", "LastKnown"]
