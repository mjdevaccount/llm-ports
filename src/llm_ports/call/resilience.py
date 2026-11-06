"""Retry and skip mechanics for long runs of model calls: `RetryPolicy` and `FailureBudget`.

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
    - Pure: no I/O, no vendor import. The adapter brings the retry library.

Rationale: a 10-hour embedding fill died on one item whose request timed out twice, because a
single failed item read as "the embedder is down".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How an adapter retries ONE request that failed `unavailable`.

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


__all__ = ["FailureBudget", "RetryPolicy"]
