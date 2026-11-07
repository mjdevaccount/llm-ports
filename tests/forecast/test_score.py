from __future__ import annotations

import math
import random

import pytest

from llm_ports.forecast.score import EPS, Bin, brier_score, ece, log_score, reliability, score


def test_brier_and_log_score_of_one_forecast() -> None:
    assert brier_score(True, 0.7) == pytest.approx(0.09)
    assert brier_score(False, 0.7) == pytest.approx(0.49)
    assert log_score(True, 0.5) == pytest.approx(math.log(2))
    assert log_score(False, 1.0) == pytest.approx(-math.log(EPS))  # clipped, finite


def test_score_of_an_overconfident_voice() -> None:
    # Claims 0.68 and hits 0.38: the shape that motivated this package.
    s = score([i < 38 for i in range(100)], [0.68] * 100)
    assert s.n == 100
    assert s.hit_rate == pytest.approx(0.38)
    assert s.overconfidence == pytest.approx(0.30)
    assert s.brier_score == pytest.approx(0.38 * 0.32**2 + 0.62 * 0.68**2)
    assert s.brier_skill_score is not None and s.brier_skill_score < 0  # worse than the base rate


def test_score_skips_invalid_and_leaves_the_undefined_none() -> None:
    empty = score([True, False], [float("nan"), 2.0])
    assert empty.n == 0 and empty.brier_score is None and empty.overconfidence is None
    one_sided = score([True, True], [0.9, 0.8])
    assert one_sided.n == 2 and one_sided.brier_skill_score is None


def test_unaligned_inputs_score_nothing() -> None:
    assert score([True, False], [0.4]).n == 0
    assert reliability([True], [0.4, 0.5]) == ()


def test_a_calibrated_voice_scores_better_than_a_flat_one() -> None:
    rng = random.Random(7)
    ps = [rng.random() for _ in range(5000)]
    ys = [rng.random() < p for p in ps]
    good = score(ys, ps)
    flat = score(ys, [0.5] * len(ys))
    assert good.brier_score is not None and flat.brier_score is not None and good.brier_score < flat.brier_score
    assert good.brier_skill_score is not None and good.brier_skill_score > 0.2
    assert ece(reliability(ys, ps)) == pytest.approx(0.0, abs=0.03)


def test_reliability_bins() -> None:
    curve = reliability([False, True, True, True, False], [0.05, 0.15, 0.95, 1.0, 0.9], bins=10)
    assert [(b.lo, b.n) for b in curve] == [(0.0, 1), (0.1, 1), (0.9, 3)]
    assert curve[-1].hit_rate == pytest.approx(2 / 3)
    assert reliability([True], [0.5], bins=0) == (Bin(lo=0.0, hi=1.0, n=1, mean_p=0.5, hit_rate=1.0),)
    assert ece(()) is None


def test_a_value_on_a_bin_edge_lands_in_that_bin() -> None:
    (b,) = reliability([True], [0.57], bins=100)
    assert b.lo == pytest.approx(0.57)
