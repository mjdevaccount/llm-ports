from __future__ import annotations

import math
import random

import pytest

from llm_ports.forecast.score import EPS, Bin, brier, ece, log_loss, reliability, score


def test_brier_and_log_loss_of_one_forecast() -> None:
    assert brier(0.7, True) == pytest.approx(0.09)
    assert brier(0.7, False) == pytest.approx(0.49)
    assert log_loss(0.5, True) == pytest.approx(math.log(2))
    assert log_loss(1.0, False) == pytest.approx(-math.log(EPS))  # clipped, finite


def test_score_of_an_overconfident_voice() -> None:
    # Claims 0.68 and hits 0.38: the shape that motivated this package.
    pairs = [(0.68, i < 38) for i in range(100)]
    s = score(pairs)
    assert s.n == 100
    assert s.hit_rate == pytest.approx(0.38)
    assert s.overconfidence == pytest.approx(0.30)
    assert s.brier == pytest.approx(0.38 * 0.32**2 + 0.62 * 0.68**2)
    assert s.brier_skill is not None and s.brier_skill < 0  # worse than the base rate


def test_score_skips_invalid_and_leaves_the_undefined_none() -> None:
    empty = score([(float("nan"), True), (2.0, False)])
    assert empty.n == 0 and empty.brier is None and empty.overconfidence is None
    one_sided = score([(0.9, True), (0.8, True)])
    assert one_sided.n == 2 and one_sided.brier_skill is None


def test_a_calibrated_voice_scores_better_than_a_flat_one() -> None:
    rng = random.Random(7)
    ps = [rng.random() for _ in range(5000)]
    ys = [rng.random() < p for p in ps]
    good = score(zip(ps, ys, strict=True))
    flat = score((0.5, y) for y in ys)
    assert good.brier is not None and flat.brier is not None and good.brier < flat.brier
    assert good.brier_skill is not None and good.brier_skill > 0.2
    assert ece(reliability(zip(ps, ys, strict=True))) == pytest.approx(0.0, abs=0.03)


def test_reliability_bins() -> None:
    curve = reliability([(0.05, False), (0.15, True), (0.95, True), (1.0, True), (0.9, False)], bins=10)
    assert [(b.lo, b.n) for b in curve] == [(0.0, 1), (0.1, 1), (0.9, 3)]
    assert curve[-1].hit_rate == pytest.approx(2 / 3)
    assert reliability([(0.5, True)], bins=0) == (Bin(lo=0.0, hi=1.0, n=1, mean_p=0.5, hit_rate=1.0),)
    assert ece(()) is None


def test_a_value_on_a_bin_edge_lands_in_that_bin() -> None:
    (b,) = reliability([(0.57, True)], bins=100)
    assert b.lo == pytest.approx(0.57)
