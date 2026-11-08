"""Replaying a combiner in time order, as it would have run: predict first, learn once resolved.

- `Event`: one question the voices spoke on: `x` (`voice -> probability`), when the combined
  forecast is made (`forecast_time`), when the outcome became known (`resolve_time`) and the
  outcome (`resolution`: True / False, None while open).
- `walk_forward(combiner, events)`: replays the events in `forecast_time` order. Before each
  prediction the combiner hears every outcome whose `resolve_time` is at or before that
  `forecast_time`, so it never learns from an outcome before the outcome was known. It returns
  one probability (None where the combiner abstained) per event, in the input's order. An open
  event is predicted and never learned, so the same replay yields the shadow forecasts.
- `holdout_start(events, share)`: the `forecast_time` from which the last `share` of the resolved
  events falls, the split a tuner fits before and is judged after.
- `compare(combiners, events, since=)`: walks each combiner over the same events and scores each
  on the same subset: resolved, forecast at or after `since`, and answered by every combiner.

Invariants:
    - Pure, standard library only.
    - Never raises for a combiner that keeps the `Combiner` contract.
    - Ties in time keep the input order; an outcome resolving exactly at a forecast time is heard
      before that forecast.

Rationale: ADR-0015.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from llm_ports.forecast._types import valid_p
from llm_ports.forecast.combine import Combiner
from llm_ports.forecast.score import Score, score


@dataclass(frozen=True)
class Event:
    """One question several voices forecast.

    Attributes:
        x: `voice -> probability` of yes, for the voices that spoke.
        forecast_time: When the combined forecast is made (the newest voice's time, at earliest).
        resolve_time: When the outcome became known; None while open.
        resolution: True if it happened, False if not, None while open.
    """

    x: Mapping[str, float]
    forecast_time: datetime
    resolve_time: datetime | None = None
    resolution: bool | None = None

    @property
    def resolved(self) -> bool:
        """True when both the outcome and its time are known."""
        return self.resolution is not None and self.resolve_time is not None


def walk_forward(combiner: Combiner, events: Sequence[Event]) -> list[float | None]:
    """Replay `events` through `combiner` in time order; return each event's probability of yes.

    Args:
        combiner: A fresh combiner; it is trained in place.
        events: The events, in any order.

    Returns:
        One probability per event, aligned with `events`; None where the combiner abstained.
        Never raises for a combiner that keeps its contract.
    """
    order = sorted(range(len(events)), key=lambda i: events[i].forecast_time)
    pending: list[tuple[datetime, int]] = []
    out: list[float | None] = [None] * len(events)
    for i in order:
        e = events[i]
        while pending and pending[0][0] <= e.forecast_time:
            _, j = heapq.heappop(pending)
            combiner.learn_one(events[j].x, bool(events[j].resolution))
        p = combiner.predict_proba_one(e.x).get(True)
        out[i] = float(p) if p is not None and valid_p(p) else None
        if e.resolution is not None and e.resolve_time is not None:
            heapq.heappush(pending, (e.resolve_time, i))
    return out


def holdout_start(events: Sequence[Event], share: float = 0.2) -> datetime | None:
    """Return the `forecast_time` from which the last `share` of the resolved events falls.

    Args:
        events: The events.
        share: The holdout's share of the resolved events, in (0, 1).

    Returns:
        The first holdout event's `forecast_time`, or None when there are fewer than two resolved
        events or `share` is outside (0, 1). Never raises.
    """
    times = sorted(e.forecast_time for e in events if e.resolved)
    if len(times) < 2 or not 0.0 < share < 1.0:
        return None
    k = min(max(int(len(times) * (1.0 - share)), 1), len(times) - 1)
    return times[k]


def compare(
    combiners: Mapping[str, Combiner], events: Sequence[Event], since: datetime | None = None
) -> dict[str, Score]:
    """Walk each combiner over `events` and score all of them on the same events.

    Args:
        combiners: `name -> fresh combiner`; each is trained in place.
        events: The events, in any order.
        since: Score only events forecast at or after this time (a holdout); None scores all.

    Returns:
        `name -> Score` over the resolved events every combiner answered (each `Score.n` is the
        same). Never raises for combiners that keep their contract.
    """
    preds = {name: walk_forward(c, events) for name, c in combiners.items()}
    keep = [
        i
        for i, e in enumerate(events)
        if e.resolved and (since is None or e.forecast_time >= since) and all(p[i] is not None for p in preds.values())
    ]
    obs = [bool(events[i].resolution) for i in keep]
    return {name: score(obs, [q for i in keep if (q := p[i]) is not None]) for name, p in preds.items()}


__all__ = ["Event", "compare", "holdout_start", "walk_forward"]
