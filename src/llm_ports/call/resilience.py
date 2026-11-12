"""Retry, deadline, concurrency and skip mechanics for model calls.

One wire request: `WirePolicy` (a `RetryRule` per retryable `FailureKind`) says how an adapter
retries it; the adapter's own loop is the only retry owner (ADR-0021). One call: `Deadline` bounds
it on the clock. One process: `LocalLimiter` caps how many calls run at once per group of models.
One long run: `FailureBudget` turns "skip and go on" into "stop".

`WirePolicy`, `RetryRule` and `Deadline` are port value types (a `GenRequest` carries them),
re-exported here. `RetryPolicy` is the 1.x embedder knob, kept as an alias that builds a
`WirePolicy` (`RetryPolicy.wire`).

A run that embeds or generates over thousands of items fails in two different ways, and the two
need different answers:

- One request is slow or dropped (a timeout, a 503, a reset). Retrying THAT request after a pause
  usually lands it. `RetryPolicy` says how often and how long to wait; an adapter that takes one
  (`PydanticAIEmbedder(retry=...)`) applies it per request, to `unavailable` failures only. A
  `request_fault` or `gated` outcome fails the same way every time and is never retried.
- One item keeps failing after its retries. The run skips it, records it, and goes on; a later run
  fills the gap. `FailureBudget` is the bound that turns "skip and go on" into "stop": too many
  failures IN A ROW (the server is down, not one item bad), or too high a failure RATE once enough
  items have been tried.

The wait before a retry is long by default (30 s, doubling to 300 s): a server that timed out on
a request may still be computing it after the client has gone (llama.cpp does), so an immediate
retry lands on a server that is still busy with the first attempt.

Invariants:
    - No I/O and no vendor import. The adapter brings the retry library. `LocalLimiter` reads the
      event loop's clock and holds per-loop semaphores; nothing else holds state.

Rationale: ADR-0021; for `FailureBudget`, a long embedding fill that died on one item whose
request timed out twice, because a single failed item read as "the embedder is down".
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from llm_ports.ports import RETRYABLE, Deadline, Limiter, RetryRule, SlotWaitExceeded, WirePolicy


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How an adapter retries ONE request that failed `unavailable`. Deprecated: use `WirePolicy`.

    Kept for 1.x callers of `PydanticAIEmbedder(retry=...)`; `wire()` is the same behaviour as a
    `WirePolicy`.

    Attributes:
        attempts: Tries per request, the first included; 1 means no retry.
        wait_initial_s: The wait before the second try; each later wait doubles.
        wait_max_s: The cap on one wait.

    Raises:
        ValueError: When `attempts` is below 1, a wait is negative, or the cap is below the
            initial wait.
    """

    attempts: int = 3
    wait_initial_s: float = 30.0
    wait_max_s: float = 300.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("RetryPolicy: attempts must be >= 1")
        if self.wait_initial_s < 0 or self.wait_max_s < self.wait_initial_s:
            raise ValueError("RetryPolicy: need 0 <= wait_initial_s <= wait_max_s")

    def wire(self) -> WirePolicy:
        """This policy as a `WirePolicy`: every retryable kind, `attempts` tries, waits doubling, no jitter.

        A server's `Retry-After` is honoured up to `wait_max_s`.
        """
        rule = RetryRule(
            attempts=self.attempts,
            wait_initial_s=self.wait_initial_s,
            wait_max_s=self.wait_max_s,
            multiplier=2.0,
            jitter=0.0,
            retry_after_max_s=self.wait_max_s,
        )
        return WirePolicy(rules=dict.fromkeys(sorted(RETRYABLE), rule))


class LocalLimiter:
    """The in-process `Limiter`: at most `slots[group]` calls at once per group, per event loop.

    Args:
        slots: Group to its slot count (e.g. a tier table's `[models.<key>] slots`).
        default: Slots for a group not in `slots`; None leaves such a group unlimited.
        max_wait_s: The longest a call waits for a slot; past it, `SlotWaitExceeded`. None waits
            as long as the caller's deadline allows.

    Raises:
        ValueError: A slot count below 1.

    Invariants:
        - A slot is released when the block exits, by return, exception or cancellation.
        - Two event loops never share a semaphore.

    Rationale: ADR-0021.
    """

    def __init__(
        self, slots: Mapping[str, int], *, default: int | None = None, max_wait_s: float | None = None
    ) -> None:
        if any(n < 1 for n in slots.values()) or (default is not None and default < 1):
            raise ValueError("LocalLimiter: a slot count must be >= 1")
        self._slots = dict(slots)
        self._default = default
        self._max_wait_s = max_wait_s
        self._sems: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]] = (
            weakref.WeakKeyDictionary()
        )

    def size(self, group: str) -> int | None:
        """The slot count of `group`; None when it is unlimited."""
        return self._slots.get(group, self._default)

    def free(self, group: str) -> int | None:
        """Slots of `group` free now on the running loop; None when it is unlimited."""
        n = self.size(group)
        if n is None:
            return None
        sem = self._sems.get(asyncio.get_running_loop(), {}).get(group)
        return n if sem is None else sem._value  # asyncio exposes no public count

    @asynccontextmanager
    async def slot(self, group: str) -> AsyncIterator[float]:
        """Hold one slot of `group`; yields the seconds waited for it.

        Raises:
            SlotWaitExceeded: No slot came free within `max_wait_s`.
        """
        n = self.size(group)
        if n is None:
            yield 0.0
            return
        loop = asyncio.get_running_loop()
        sem = self._sems.setdefault(loop, {}).setdefault(group, asyncio.Semaphore(n))
        started = loop.time()
        try:
            async with asyncio.timeout(self._max_wait_s):
                await sem.acquire()
        except TimeoutError:
            raise SlotWaitExceeded(group, loop.time() - started) from None
        try:
            yield loop.time() - started
        finally:
            sem.release()

    @classmethod
    def for_models(
        cls, slots: Mapping[str, tuple[str | None, int | None]], *, max_wait_s: float | None = None
    ) -> LocalLimiter:
        """Build a limiter from per-model facts: model key to `(group, slots)`.

        A model's group is its `group`, else its key; a group's count is the smallest any of its
        models states. A model that states no slots adds nothing.
        """
        out: dict[str, int] = {}
        for key, (group, n) in slots.items():
            if n is not None:
                g = group or key
                out[g] = min(out.get(g, n), n)
        return cls(out, max_wait_s=max_wait_s)


@dataclass(slots=True)
class FailureBudget:
    """Counts a run's per-item outcomes and says when skipping has to become stopping.

    `record(ok)` once per item tried. `exhausted` turns true when `max_consecutive` items failed
    in a row, or when at least `min_attempts` items were tried and more than `max_rate` of them
    failed. A success resets the run of consecutive failures; the rate counts the whole run.

    Attributes:
        max_rate: The failure rate above which the run stops, in [0, 1].
        min_attempts: Items tried before the rate is judged, so two early failures do not stop it.
        max_consecutive: Failures in a row that stop the run.
        attempts: Items tried so far.
        failures: Items failed so far.
        consecutive: The current run of failures in a row.

    Raises:
        ValueError: When `max_rate` is outside [0, 1] or a count bound is below 1.
    """

    max_rate: float = 0.05
    min_attempts: int = 20
    max_consecutive: int = 3
    attempts: int = 0
    failures: int = 0
    consecutive: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_rate <= 1.0:
            raise ValueError("FailureBudget: max_rate must be in [0, 1]")
        if self.min_attempts < 1 or self.max_consecutive < 1:
            raise ValueError("FailureBudget: min_attempts and max_consecutive must be >= 1")

    def record(self, ok: bool) -> None:
        """Count one item tried: `ok` True when it landed."""
        self.attempts += 1
        if ok:
            self.consecutive = 0
            return
        self.failures += 1
        self.consecutive += 1

    @property
    def rate(self) -> float:
        """Failures over attempts; 0.0 before any attempt."""
        return self.failures / self.attempts if self.attempts else 0.0

    @property
    def exhausted(self) -> bool:
        """True when the run should stop rather than skip another failed item."""
        return self.reason is not None

    @property
    def reason(self) -> str | None:
        """Why the budget is exhausted, as one line; None while it is not."""
        if self.consecutive >= self.max_consecutive:
            return f"{self.consecutive} failures in a row (max {self.max_consecutive})"
        if self.attempts >= self.min_attempts and self.rate > self.max_rate:
            return f"failure rate {self.rate:.1%} of {self.attempts} > {self.max_rate:.1%}"
        return None


__all__ = [
    "Deadline",
    "FailureBudget",
    "Limiter",
    "LocalLimiter",
    "RetryPolicy",
    "RetryRule",
    "SlotWaitExceeded",
    "WirePolicy",
]
