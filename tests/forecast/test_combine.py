from __future__ import annotations

import pytest

from llm_ports.forecast import LogitPool
from llm_ports.forecast.combine import Combiner, expit, logit


def test_logit_and_expit_invert_and_stay_finite() -> None:
    for p in (0.01, 0.3, 0.5, 0.97):
        assert expit(logit(p)) == pytest.approx(p)
    assert expit(-800.0) == pytest.approx(0.0) and expit(800.0) == pytest.approx(1.0)
    assert abs(logit(0.0)) < 20 and abs(logit(1.0)) < 20


def test_equal_weight_pool_and_extremizing() -> None:
    pool: Combiner = LogitPool()
    assert pool.combine({"a": 0.7, "b": 0.7}) == pytest.approx(0.7)
    assert pool.combine({"a": 0.8, "b": 0.2}) == pytest.approx(0.5)
    sharp = LogitPool(alpha=2.0).combine({"a": 0.7, "b": 0.7})
    assert sharp is not None and sharp > 0.8


def test_weights_and_unusable_voices() -> None:
    pool = LogitPool(weights={"a": 3.0, "muted": 0.0})
    p = pool.combine({"a": 0.9, "b": 0.5, "muted": 0.01, "bad": float("nan")})
    assert p is not None and p == pytest.approx(expit(0.75 * logit(0.9)))
    assert pool.combine({}) is None
    assert pool.combine({"muted": 0.9}) is None
    pool.learn({"a": 0.9}, True)  # fixed: nothing changes
    assert pool.combine({"a": 0.9}) == pytest.approx(0.9)


@pytest.mark.parametrize("pool", [LogitPool(weights={"a": float("inf")}), LogitPool(weights={"a": float("nan")})])
def test_a_non_finite_weight_is_ignored(pool: LogitPool) -> None:
    assert pool.combine({"a": 0.9, "b": 0.6}) == pytest.approx(0.6)


def test_a_non_finite_alpha_does_not_combine() -> None:
    assert LogitPool(alpha=float("nan")).combine({"a": 0.7}) is None
