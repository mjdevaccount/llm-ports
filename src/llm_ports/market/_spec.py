"""One background action's declaration, and the rules that refuse a declaration nothing could judge.

Invariants:
    - A spec declares at least one freshness rule: `desired_ttl`, `cadence` or `max_silence_h`. A
      producer with none cannot be judged stale and would read as unknown forever.
    - Interval literals are parsed when the spec is declared, so a malformed one fails at import,
      not on the first sync.
    - Specs compare on their declared metadata (and `handler_path`), never on the callables. A
      registry therefore never treats equality as sameness: only the identical object re-adds.
    - A `desired_ttl` needs `cache_table` and `age_col`: a TTL with nothing to date the output
      by can never be judged stale.

Rationale: ADR-0018.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

#: A stale function: `(session, ttl_seconds)` to the work keys currently due. The session is the
#: consumer's own (a database session, a client, None); the market never opens one.
StaleFn = Callable[[Any, float], Awaitable[list[str]]]
#: A handler: `(session, work_key)` to its output. Its output contract is the consumer's.
Handler = Callable[..., Awaitable[Any]]

#: The clocks silence can be counted on: `wall` counts every hour, `session` only hours inside the
#: consumer's own trading or business calendar. The consumer's freshness check owns the calendar.
SILENCE_CLOCKS: tuple[str, ...] = ("wall", "session")

_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_INTERVAL = re.compile(r"\s*(\d+)\s*([smhd])\s*")


def parse_interval(literal: str) -> timedelta:
    """Parse a short interval literal: an integer and one unit of `s`, `m`, `h` or `d`.

    Args:
        literal: `"90s"`, `"30m"`, `"18h"`, `"7d"`; surrounding whitespace is ignored.

    Returns:
        The interval.

    Raises:
        ValueError: `literal` is not `<int><s|m|h|d>`.
    """
    m = _INTERVAL.fullmatch(literal)
    if m is None:
        raise ValueError(f"bad interval literal: {literal!r} (use <int><s|m|h|d>)")
    return timedelta(seconds=int(m.group(1)) * _UNIT_S[m.group(2)])


@dataclass(frozen=True)
class ActionSpec:
    """A background action: a handler, what makes it due, and how it is ranked and judged.

    Attributes:
        feature: The feature that owns the action (the first half of its key).
        action: The action's name within the feature (the second half of its key).
        handler_path: `module.qualname` of the handler, recorded as provenance.
        fn: The handler. Not compared: two imports of one module declare equal specs.
        model: The model alias the handler calls, recorded for the operator; None when it calls none.
        lane: Priority lane, 0 first; a scheduler claims a lower lane before a higher one.
        value_weight: Relative worth inside a lane; a fair-share scheduler divides by it.
        desired_ttl: How long an output stays fresh, as an interval literal (`"18h"`).
        est_cost_s: Expected seconds per dispatch; a scheduler scales its timeout and lease by it.
        cadence: How often the action is expected to run, as an interval literal.
        cache_table: Where the action's output lands, for a freshness check to read.
        age_col: The column of `cache_table` that dates an output.
        stale_fn: Yields the work keys currently due; None for an action fed only by events.
            Not compared.
        max_silence_h: Hours of output silence past which an action with no `desired_ttl` and no
            `cadence` is stale: a property of how often its input arrives.
        silence_clock: The clock silence is counted on (`SILENCE_CLOCKS`, or the registry's own).
        source: Ids of the content sources the action reads or writes; a consumer skips the action
            while any of them is switched off. Empty when it depends on none.
    """

    feature: str
    action: str
    handler_path: str
    fn: Handler = field(compare=False, repr=False)
    model: str | None = None
    lane: int = 2
    value_weight: float = 1.0
    desired_ttl: str | None = None
    est_cost_s: float = 40.0
    cadence: str | None = None
    cache_table: str | None = None
    age_col: str | None = None
    stale_fn: StaleFn | None = field(default=None, compare=False, repr=False)
    max_silence_h: float | None = None
    silence_clock: str = "wall"
    source: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        """`(feature, action)`, the registry key."""
        return (self.feature, self.action)

    @property
    def name(self) -> str:
        """`feature.action`, the name a kill switch and an operator use."""
        return f"{self.feature}.{self.action}"

    @property
    def ttl(self) -> timedelta | None:
        """`desired_ttl` parsed, or None."""
        return parse_interval(self.desired_ttl) if self.desired_ttl else None

    @property
    def cadence_interval(self) -> timedelta | None:
        """`cadence` parsed, or None."""
        return parse_interval(self.cadence) if self.cadence else None

    @property
    def limit_h(self) -> float | None:
        """The declared freshness limit in hours: `desired_ttl`, else `cadence`, else None.

        None means `max_silence_h` is the rule.
        """
        span = self.ttl or self.cadence_interval
        return span.total_seconds() / 3600.0 if span is not None else None


def refusal(
    spec: ActionSpec, clocks: Collection[str] = SILENCE_CLOCKS, sources: Collection[str] | None = None
) -> str | None:
    """Return why `spec` cannot be declared, or None when it can.

    Args:
        spec: The declaration.
        clocks: The silence clocks the registry accepts.
        sources: The source ids that exist (the consumer's lane registry); None checks none.

    Returns:
        The reason, naming the field; None for a declarable spec. Never raises.
    """
    if not spec.feature or not spec.action:
        return "feature and action must be non-empty"
    if spec.desired_ttl is None and spec.cadence is None and spec.max_silence_h is None:
        return (
            "declare desired_ttl, cadence or max_silence_h: an action with no freshness rule cannot be "
            "judged stale and reads as unknown forever"
        )
    for label, literal in (("desired_ttl", spec.desired_ttl), ("cadence", spec.cadence)):
        if literal is not None:
            try:
                parse_interval(literal)
            except ValueError as exc:
                return f"{label}: {exc}"
    if spec.desired_ttl is not None and not (spec.cache_table and spec.age_col):
        return (
            "desired_ttl needs cache_table and age_col: a TTL with no output to date can never be "
            "judged stale"
        )
    if spec.max_silence_h is not None and not spec.max_silence_h > 0:
        return f"max_silence_h must be > 0, not {spec.max_silence_h!r}"
    if spec.silence_clock not in clocks:
        return f"silence_clock {spec.silence_clock!r} not in {tuple(clocks)}"
    if sources is not None:
        unknown = [s for s in spec.source if s not in sources]
        if unknown:
            return f"source {unknown} not among the known sources"
    if isinstance(spec.lane, bool) or not isinstance(spec.lane, int) or spec.lane < 0:
        return f"lane must be an int >= 0, not {spec.lane!r}"
    if not spec.value_weight > 0:
        return f"value_weight must be > 0, not {spec.value_weight!r}"
    if not spec.est_cost_s > 0:
        return f"est_cost_s must be > 0, not {spec.est_cost_s!r}"
    return None


__all__ = ["SILENCE_CLOCKS", "ActionSpec", "Handler", "StaleFn", "parse_interval", "refusal"]
