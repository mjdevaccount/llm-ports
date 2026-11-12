"""The one wire-retry loop every adapter shares: a `WirePolicy` keyed on `Failure.kind`.

`WireRetry.run(send, judge)` sends one wire request and re-sends it while `judge` names a failure
the policy retries. `judge` reads either the exception `send` raised or the result it returned
(an adapter whose failures are values, like the reranker, returns them), and gives back the
`Failure` and the response headers it saw. tenacity is the loop engine; the decision is here.

A failure is retried only when all of these hold:

- its kind is in the policy's `rules` (so never `quota`, `auth`, a request-side kind or a bug);
- it did not happen after output had started (`in_stream`);
- the server did not say `x-should-retry: false`;
- this kind has tries left on this request, and the call has wire requests left in total;
- a server hint (`retry-after-ms`, then `Retry-After`) is within the rule's `retry_after_max_s`;
- the wait ends before the call's deadline. A wait that would not marks the call `expired` and
  ends it at once: it does not sleep and then fail.

Every failed request that is retried becomes an `Attempt` and emits `llm_wire_retry`; an expiry
emits `llm_deadline_expired`.

Sits directly under `adapters/`, outside the adapter-independence contract, like `_classify`:
the pydantic-ai generator and embedder and the HTTP reranker share it. Both extras carry tenacity.

Invariants:
    - One retry owner: an adapter that uses this loop sets its SDK's own retries to 0 and installs
      no retrying transport, so attempts never multiply.
    - `asyncio.CancelledError` is never retried and never swallowed.

Rationale: ADR-0021.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from llm_ports._extras import extra
from llm_ports.observability import EventHook
from llm_ports.ports import Attempt, Failure, FailureKind, WirePolicy

with extra("http"):
    import tenacity

#: What `judge` returns: the failure a reply or an exception means (None for success), and the
#: response headers it saw (empty when there was no response).
Judged = tuple[Failure | None, Mapping[str, str]]
#: Reads a raised exception (`BaseException`) or a returned result (anything else).
Judge = Callable[[Any], Judged]


async def _sleep(seconds: float) -> None:
    """The backoff sleep; a test replaces it to run on a fake clock."""
    await asyncio.sleep(seconds)


def _now() -> float:
    return asyncio.get_running_loop().time()


class WireRetry:
    """One call's wire-retry state: the policy, the deadline, the attempts so far.

    Args:
        policy: Which kinds are retried, and how.
        model: The model id, named in each `Attempt` and event.
        deadline_at: When the call must be over, on the loop's clock; None for no bound.
        emit: Receives `llm_wire_retry` and `llm_deadline_expired`.
        label: The call's name in events.

    Attributes:
        attempts: Every failed wire request that was retried, in order.
        sent: Wire requests sent so far in this call.
        expired: True once a wait would have passed the deadline.
    """

    def __init__(
        self,
        policy: WirePolicy,
        *,
        model: str,
        deadline_at: float | None,
        emit: EventHook,
        label: str = "",
    ) -> None:
        self.policy = policy
        self.model = model
        self.deadline_at = deadline_at
        self.label = label
        self._emit = emit
        self.attempts: list[Attempt] = []
        self.sent = 0
        self.expired = False
        self._started: float | None = None

    async def run[T](self, send: Callable[[], Awaitable[T]], judge: Judge) -> T:
        """Send one wire request through `send`, re-sending it while the policy allows.

        Returns:
            What the last request returned (a failed result too, when the adapter returns failures).

        Raises:
            BaseException: What the last request raised, once the policy stops retrying it;
                `asyncio.CancelledError` at once.
        """
        if self._started is None:
            self._started = _now()
        per_kind: dict[FailureKind, int] = {}
        wait = [0.0]

        def decide(outcome: Any) -> bool:
            if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
                return False  # a cancellation, a KeyboardInterrupt: never retried
            got, headers = judge(outcome)
            if got is None:
                return False
            next_wait = self._next_wait(got, headers, per_kind)
            if next_wait is None:
                return False
            wait[0] = next_wait
            self.attempts.append(Attempt(len(self.attempts) + 1, None, self.model, got, next_wait))
            self._emit(
                logging.INFO,
                "llm_wire_retry",
                {
                    "label": self.label,
                    "model": self.model,
                    "kind": got.kind,
                    "n": len(self.attempts),
                    "wait_s": next_wait,
                },
            )
            return True

        retrying = tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception(decide) | tenacity.retry_if_result(decide),
            wait=lambda _state: wait[0],
            stop=tenacity.stop_never,
            sleep=_sleep,
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                self.sent += 1
                result = await send()
            outcome = attempt.retry_state.outcome
            if outcome is not None and not outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    def _next_wait(self, got: Failure, headers: Mapping[str, str], per_kind: dict[FailureKind, int]) -> float | None:
        """The wait before the next try, or None when the policy stops here."""
        rule = self.policy.rules.get(got.kind)
        if rule is None or got.in_stream or headers.get("x-should-retry", "").strip().lower() == "false":
            return None
        tries = per_kind[got.kind] = per_kind.get(got.kind, 0) + 1
        if tries >= rule.attempts or self.sent >= self.policy.max_total_attempts:
            return None
        if rule.honour_retry_after and got.retry_after_s is not None:
            if got.retry_after_s > rule.retry_after_max_s:
                return None
            wait = got.retry_after_s
        else:
            wait = rule.wait_s(tries, spread=random.uniform(-1.0, 1.0))  # jitter, not a secret
        if self.deadline_at is not None and _now() + wait >= self.deadline_at:
            self.expire("backoff")
            return None
        return wait

    def expire(self, where: str) -> None:
        """Mark the call expired and emit `llm_deadline_expired` (once)."""
        if self.expired:
            return
        self.expired = True
        elapsed = _now() - self._started if self._started is not None else None
        self._emit(
            logging.WARNING,
            "llm_deadline_expired",
            {"label": self.label, "model": self.model, "phase": "total", "where": where, "elapsed_s": elapsed},
        )


__all__: list[str] = []
