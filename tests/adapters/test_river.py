"""adapters.river: the online logit pool learns which voice to trust and never raises."""

from __future__ import annotations

import random
from typing import ClassVar

import pytest

pytest.importorskip("river")

from llm_ports.adapters import river as river_adapter


def test_river_pool_learns_to_trust_the_informed_voice() -> None:
    rng = random.Random(5)
    pool = river_adapter.RiverLogitPool(lr=0.1, warmup=20)
    assert pool.predict_proba_one({"sharp": 0.9}) == {}  # before warmup
    for _ in range(3000):
        truth = rng.random() < 0.5
        sharp = 0.85 if truth else 0.15
        noise = rng.random()
        ps = {"sharp": sharp, "noise": noise}
        if rng.random() < 0.2:
            del ps["noise"]  # silent sometimes
        pool.learn_one(ps, truth)
    w = pool.weights()
    assert w["voice:sharp"] > abs(w["voice:noise"])
    hi = pool.predict_proba_one({"sharp": 0.85, "noise": 0.1}).get(True)
    lo = pool.predict_proba_one({"sharp": 0.15, "noise": 0.9}).get(True)
    assert hi is not None and lo is not None and hi > 0.7 > 0.3 > lo
    assert pool.voices == {"sharp", "noise"} and pool.seen == 3000
    both = pool.predict_proba_one({"sharp": 0.85})
    assert both[False] == pytest.approx(1.0 - both[True])


def test_a_river_classifier_drops_in_as_a_combiner() -> None:
    from river import linear_model

    from llm_ports.forecast.combine import Combiner

    model: Combiner = linear_model.LogisticRegression()
    model.learn_one({"a": 0.9}, True)
    assert set(model.predict_proba_one({"a": 0.9})) == {False, True}


def test_river_pool_ignores_unusable_events_and_survives_failures() -> None:
    pool = river_adapter.RiverLogitPool(warmup=0)
    pool.learn_one({"a": float("nan")}, True)
    assert pool.seen == 0 and pool.predict_proba_one({}) == {}

    class Broken:
        weights: ClassVar[dict[str, float]] = {}

        def learn_one(self, *a: object) -> None:
            raise ValueError("bad")

        def predict_proba_one(self, *a: object) -> dict[bool, float]:
            raise ValueError("bad")

    pool._model = Broken()
    pool.learn_one({"a": 0.6}, True)
    assert pool.seen == 0 and pool.failures == 1 and pool.predict_proba_one({"a": 0.6}) == {}
