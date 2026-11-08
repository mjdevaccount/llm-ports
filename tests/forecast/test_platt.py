from __future__ import annotations

import json
import math
import random

import pytest

from llm_ports.forecast import Platt, PlattCalibrator
from llm_ports.forecast.platt import EPS


def _overconfident(n: int, seed: int = 7) -> tuple[list[float], list[bool]]:
    """A voice whose stated p is too extreme: the truth is sigmoid(0.5 * logit(p))."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        p = rng.uniform(0.02, 0.98)
        truth = 1.0 / (1.0 + math.exp(-0.5 * math.log(p / (1 - p))))
        xs.append(p)
        ys.append(rng.random() < truth)
    return xs, ys


def test_fit_recovers_an_overconfident_voices_slope() -> None:
    xs, ys = _overconfident(4000)
    cal = PlattCalibrator().fit(xs, ys)
    assert cal.insufficient_ is None and cal.map_ is not None
    assert cal.map_.a == pytest.approx(0.5, abs=0.08)
    assert cal.map_.b == pytest.approx(0.0, abs=0.1)
    assert cal.map_.n == 4000
    assert cal.fitted_ == (cal.map_.a, cal.map_.b)
    rows = cal.predict_proba([0.9])
    assert rows is not None and rows[0][0] + rows[0][1] == pytest.approx(1.0)
    assert rows[0][1] < 0.9  # pulled toward 0.5


def test_an_inverted_slope_is_refused_and_kept_for_diagnosis() -> None:
    xs, ys = _overconfident(2000)
    cal = PlattCalibrator().fit(xs, [not o for o in ys])
    assert cal.map_ is None and cal.predict_proba([0.5]) is None
    assert cal.insufficient_ is not None and cal.insufficient_.reason.startswith("inverted slope")
    assert cal.fitted_ is not None and cal.fitted_[0] < 0


@pytest.mark.parametrize(
    ("X", "y", "reason"),
    [
        ([0.5] * 10, [True, False] * 5, "usable pairs"),
        ([0.3] * 80, [True] * 80, "one outcome only"),
        ([0.3, 0.4], [True], "X has 2 items"),
    ],
)
def test_too_little_data_gives_no_map_and_never_raises(X: list[float], y: list[bool], reason: str) -> None:
    cal = PlattCalibrator().fit(X, y)
    assert cal.map_ is None and cal.fitted_ is None
    assert cal.insufficient_ is not None and reason in cal.insufficient_.reason


def test_a_refit_clears_the_previous_map() -> None:
    xs, ys = _overconfident(500)
    cal = PlattCalibrator().fit(xs, ys)
    assert cal.map_ is not None
    cal.fit(xs[:5], ys[:5])
    assert cal.map_ is None and cal.insufficient_ is not None


def test_platt_map_clamps_handles_bad_input_and_round_trips() -> None:
    m = Platt(a=2.0, b=0.0, n=9)
    assert m(0.5) == pytest.approx(0.5)
    assert m(0.0) >= EPS and m(1.0) <= 1 - EPS
    assert m(float("nan")) == 0.5
    assert Platt.from_dict(json.loads(json.dumps(m.as_dict()))) == m
    assert Platt.from_dict({"a": 1, "b": 0}).n == 0
    with pytest.raises(ValueError, match="not a Platt dict"):
        Platt.from_dict({"a": 1})


@pytest.mark.parametrize(
    "kw", [{"a": 0.0, "b": 0.0}, {"a": -1.0, "b": 0.0}, {"a": math.inf, "b": 0.0}, {"a": 1.0, "b": math.nan}]
)
def test_platt_map_refuses_a_non_monotone_or_non_finite_map(kw: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="Platt"):
        Platt(**kw)


def test_params_follow_the_scikit_learn_contract() -> None:
    cal = PlattCalibrator(min_pairs=10)
    assert cal.get_params() == {"min_pairs": 10, "ridge": 1e-3, "max_iter": 50, "tol": 1e-8}
    assert cal.set_params(ridge=0.1).ridge == 0.1
    with pytest.raises(ValueError, match="invalid parameter"):
        cal.set_params(nope=1)
