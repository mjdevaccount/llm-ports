"""forecast.walk: a combiner replayed in time order never hears an outcome before it was known."""

from __future__ import annotations

import random
from collections.abc import Mapping
from datetime import datetime, timedelta

import pytest

from llm_ports.forecast import BestVoice, Event, LogitPool, PriorPool, walk_forward
from llm_ports.forecast.walk import compare, holdout_start

T0 = datetime(2026, 1, 1)


class Recorder:
    """A combiner that answers (outcomes heard) / 10."""

    def __init__(self) -> None:
        self.heard: list[bool] = []

    def predict_proba_one(self, x: Mapping[str, float]) -> dict[bool, float]:
        del x
        p = len(self.heard) / 10
        return {False: 1 - p, True: p}

    def learn_one(self, x: Mapping[str, float], y: bool) -> None:
        del x
        self.heard.append(y)


def ev(day: int, resolve_day: int | None, y: bool | None, **x: float) -> Event:
    return Event(
        x=x,
        forecast_time=T0 + timedelta(days=day),
        resolve_time=None if resolve_day is None else T0 + timedelta(days=resolve_day),
        resolution=y,
    )


def test_an_outcome_is_heard_only_once_known() -> None:
    events = [
        ev(3, 4, True, a=0.5),
        ev(0, 3, False, a=0.5),
        ev(1, 9, True, a=0.5),
        ev(5, None, None, a=0.5),
    ]
    rec = Recorder()
    got = walk_forward(rec, events)
    # day 0 and day 1 hear nothing; day 3 hears day 0's outcome (it resolves AT day 3);
    # day 5 hears day 0 and day 3; day 1 resolves on day 9 and is never heard.
    assert got == [pytest.approx(0.1), pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.2)]
    assert rec.heard == [False, True]


def test_an_abstention_is_none_and_the_order_is_the_inputs() -> None:
    got = walk_forward(BestVoice(min_n=1), [ev(1, 2, True, a=0.8), ev(0, 1, True, a=0.6)])
    assert got == [pytest.approx(0.8), None]
    assert walk_forward(LogitPool(), []) == []


def test_holdout_start_splits_the_resolved_events() -> None:
    events = [ev(d, d + 1, True, a=0.5) for d in range(10)] + [ev(20, None, None, a=0.5)]
    assert holdout_start(events, 0.2) == T0 + timedelta(days=8)
    assert holdout_start(events[:1], 0.2) is None
    assert holdout_start(events, 1.0) is None


def test_compare_scores_every_combiner_on_the_same_events() -> None:
    rng = random.Random(3)
    events = []
    for d in range(400):
        y = rng.random() < 0.5
        events.append(ev(d, d + 1, y, sharp=0.8 if y else 0.2, noise=rng.random(), prior=0.5))
    got = compare(
        {
            "equal": LogitPool(),
            "best": BestVoice(min_n=10),
            "prior": PriorPool(inner=BestVoice(min_n=10)),
        },
        events,
        since=T0 + timedelta(days=200),
    )
    assert {s.n for s in got.values()} == {200}
    b = {k: s.brier_score for k, s in got.items()}
    assert b["best"] is not None and b["equal"] is not None and b["best"] < b["equal"]
