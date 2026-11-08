"""forecast.combine: PriorPool shrinks toward and falls back to a prior; BestVoice tracks the best voice."""

from __future__ import annotations

import pytest

from llm_ports.forecast import BestVoice, LogitPool, PriorPool
from llm_ports.forecast.combine import Combiner, expit, logit


def yes(pool: Combiner, x: dict[str, float]) -> float | None:
    return pool.predict_proba_one(x).get(True)


def test_prior_pool_falls_back_to_the_prior_and_shrinks_toward_it() -> None:
    silent = PriorPool(inner=BestVoice(min_n=5), prior_key="base")
    assert yes(silent, {"a": 0.9, "base": 0.6}) == pytest.approx(0.6)  # inner abstains: the prior
    assert silent.predict_proba_one({"a": 0.9}) == {}  # neither answers
    pool = PriorPool(inner=LogitPool(), prior_key="base", shrink=0.5)
    inner = expit((logit(0.8) + logit(0.6)) / 2)  # the equal pool hears the prior as a voice
    assert yes(pool, {"a": 0.8, "base": 0.6}) == pytest.approx(expit(0.5 * logit(inner) + 0.5 * logit(0.6)))
    assert yes(PriorPool(inner=LogitPool(), shrink=1.0), {"a": 0.8, "prior": 0.3}) == pytest.approx(0.3)
    assert yes(PriorPool(inner=LogitPool(), alpha=2.0), {"a": 0.7}) == pytest.approx(expit(2 * logit(0.7)))


@pytest.mark.parametrize("bad", [{"shrink": 1.5}, {"shrink": -0.1}, {"alpha": float("nan")}, {"shrink": float("inf")}])
def test_prior_pool_refuses_a_bad_setting(bad: dict[str, float]) -> None:
    assert PriorPool(inner=LogitPool(), **bad).predict_proba_one({"a": 0.7, "prior": 0.5}) == {}


def test_prior_pool_passes_outcomes_to_the_inner_combiner() -> None:
    inner = BestVoice(min_n=1)
    PriorPool(inner=inner).learn_one({"a": 0.9, "prior": 0.5}, True)
    assert inner.n == {"a": 1, "prior": 1}


def test_best_voice_answers_with_the_lowest_running_brier() -> None:
    best = BestVoice(min_n=3)
    for _ in range(3):
        best.learn_one({"sharp": 0.9, "dull": 0.6, "bad": float("nan")}, True)
    assert best.best({"sharp": 0.2, "dull": 0.5}) == "sharp"
    assert yes(best, {"sharp": 0.2, "dull": 0.5}) == pytest.approx(0.2)
    assert yes(best, {"dull": 0.55}) == pytest.approx(0.55)  # the best voice that spoke
    assert best.predict_proba_one({"new": 0.9}) == {}  # not yet eligible
    assert "bad" not in best.n
    tie = BestVoice(min_n=1)
    tie.learn_one({"b": 0.7, "a": 0.7}, True)
    assert tie.best({"b": 0.7, "a": 0.7}) == "a"
