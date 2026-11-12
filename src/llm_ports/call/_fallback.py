"""The facade's fallback loop: which tier runs next, what the result says about the ones that failed.

A role's tier, then every tier its `fallback` chain reaches, in order. A tier's result ends the
call unless its failure kind is in that tier's `fallback_on`; then the next tier runs with what is
left of the deadline. A result always names the tier that produced it (`answered.tier`) and
carries, in `attempts`, every earlier tier's failure and every retried wire request, each tagged
with its tier.

Invariants:
    - Pure bookkeeping: no I/O; events go to the hook passed in.
    - The next tier never starts once the deadline is spent: the call is `timeout` / `total`.

Rationale: ADR-0021, ADR-0022.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from llm_ports.call.roles import Role, _fallback_chain
from llm_ports.observability import EventHook
from llm_ports.ports import Answered, Attempt, Deadline, Failure, Generated, failure


def chain_of(role: Role) -> Iterator[Role]:
    """The role, then the role on each tier its fallback chain reaches, in order.

    A fallback tier drops the role's own model override: the override names a model of the role's
    tier, and a fallback tier's model is its own.
    """
    yield role
    for name in _fallback_chain(role.TIERS, role.tier):
        yield replace(role, tier=name, model=None)


class Route:
    """One call's walk down its fallback chain.

    Attributes:
        result: The latest tier's result; None before the first and after an expiry.
        attempts: Every failed step so far: each retried wire request and each tier that failed over.
        depth: The index of the tier running now in the chain; -1 before the first.
        tier: The tier running now.
    """

    def __init__(self, role: Role, clocks: Deadline) -> None:
        self.role = role
        self.clocks = clocks
        self.result: Generated[Any] | None = None
        self.attempts: list[Attempt] = []
        self.depth = -1
        self.tier = role.tier
        self._failed: Failure | None = None

    def next_tier(self, tier_role: Role, now: float, emit: EventHook) -> bool:
        """Start the next tier; False when the deadline is spent (the call has expired)."""
        if self.depth >= 0:
            if self.clocks.remaining(now) <= 0:
                self.expire(now, emit)
                return False
            emit(
                logging.WARNING,
                "llm_fallback",
                {
                    "role": self.role.name,
                    "from": self.tier,
                    "to": tier_role.tier,
                    "kind": self._failed.kind if self._failed is not None else None,
                    "depth": self.depth + 1,
                },
            )
            self.result = None  # the tier that failed over is in `attempts` now
        self.depth += 1
        self.tier = tier_role.tier
        return True

    def settle(self, tier_role: Role, res: Generated[Any]) -> bool:
        """Record one tier's result; True when its failure fails over to the next tier."""
        tier = tier_role.resolved_tier
        info = tier.info
        own = tuple(replace(a, tier=a.tier or tier.name) for a in res.attempts)
        asked = res.answered or Answered(requested=res.model)
        answered = replace(
            asked, tier=tier.name, provider=asked.provider or (info.provider if info is not None else None)
        )
        res = replace(
            res, attempts=(*self.attempts, *own), answered=answered, stamp={**res.stamp, "fallback_depth": self.depth}
        )
        self.result = res
        got = res.failure
        if got is None or got.kind not in tier.fallback_on:
            return False
        self.attempts.extend(own)
        self.attempts.append(Attempt(len(self.attempts) + 1, tier.name, res.model, got, 0.0, usage=res.usage_v2))
        self._failed = got
        return True

    def expire(self, now: float, emit: EventHook) -> None:
        """The deadline ran out before a tier answered: the call is `timeout` (`timed_out`).

        Emits `llm_deadline_expired`.
        """
        self.result = None
        emit(
            logging.WARNING,
            "llm_deadline_expired",
            {
                "role": self.role.name,
                "tier": self.tier,
                "phase": "total",
                "elapsed_s": now - (self.clocks.started or now),
                "total_s": self.clocks.total_s,
            },
        )


def timed_out(role: Role, route: Route) -> Generated[Any]:
    """The `timeout` / `total` result of a call whose deadline ran out, with every attempt so far."""
    got = failure("timeout", phase="total", reason=f"deadline: the call's {route.clocks.total_s:g} s ran out")
    tier = role.TIERS[route.tier]
    return Generated(
        value=None,
        outcome=got.outcome,
        model=tier.model,
        answered=Answered(requested=tier.model, tier=tier.name),
        attempts=tuple(route.attempts),
        failure=got,
        stamp={"fallback_depth": max(route.depth, 0)},
    )


__all__: list[str] = []
