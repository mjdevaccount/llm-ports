"""adapters.sklearn: the isotonic calibrator fixes an overconfident voice; a short or failed fit is Insufficient."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("sklearn")

from llm_ports.adapters.sklearn import IsotonicCalibrator
from llm_ports.forecast import Calibrator, Insufficient, PiecewiseLinear
from llm_ports.forecast.score import score


def overconfident(n: int, seed: int = 3) -> tuple[list[float], list[bool]]:
    """A voice whose stated p overclaims: the truth is p**2 (0.68 stated hits about 0.46)."""
    rng = random.Random(seed)
    ps = [rng.random() for _ in range(n)]
    return ps, [rng.random() < p * p for p in ps]


def test_isotonic_fixes_an_overconfident_voice() -> None:
    ps, ys = overconfident(4000)
    cal: Calibrator = IsotonicCalibrator()
    cal.fit(ps, ys)
    fitted = IsotonicCalibrator().fit(ps, ys).map_
    assert isinstance(fitted, PiecewiseLinear) and fitted.n == 4000
    held_p, held_y = overconfident(4000, seed=11)
    raw = score(held_y, held_p)
    cal_rows = cal.predict_proba(held_p)
    assert cal_rows is not None
    calibrated = score(held_y, [r[1] for r in cal_rows])
    assert raw.brier_score is not None and calibrated.brier_score is not None
    assert calibrated.brier_score < raw.brier_score
    assert fitted(0.7) == pytest.approx(0.49, abs=0.08)
    assert PiecewiseLinear.from_dict(fitted.as_dict()) == fitted


def test_isotonic_is_a_scikit_learn_estimator() -> None:
    from sklearn.base import clone

    cal = IsotonicCalibrator(min_pairs=5)
    assert cal.get_params() == {"min_pairs": 5}
    assert cal.predict_proba([0.5]) is None  # unfitted: no map, no exception
    twin = clone(cal.set_params(min_pairs=7))
    assert twin.get_params() == {"min_pairs": 7} and not hasattr(twin, "map_")


def test_isotonic_refuses_too_few_and_a_failed_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    short = IsotonicCalibrator().fit([0.5] * 5, [True, False] * 2 + [True])
    assert short.map_ is None and isinstance(short.insufficient_, Insufficient) and short.predict_proba([0.5]) is None

    class Broken:
        def __init__(self, **k: object) -> None:
            pass

        def fit(self, *a: object, **kw: object) -> None:
            raise ValueError("bad")

    import llm_ports.adapters.sklearn as sk

    monkeypatch.setattr(sk, "IsotonicRegression", Broken)
    ps, ys = overconfident(100)
    out = IsotonicCalibrator().fit(ps, ys)
    assert out.map_ is None and out.insufficient_ is not None and "isotonic fit failed" in out.insufficient_.reason


def test_a_good_fit_clears_an_earlier_refusal() -> None:
    cal = IsotonicCalibrator().fit([0.5], [True])
    assert cal.insufficient_ is not None
    ps, ys = overconfident(200)
    assert cal.fit(ps, ys).insufficient_ is None and cal.map_ is not None


def test_weights_pull_the_map_toward_the_heavier_pairs() -> None:
    p = [0.5] * 40
    y = [True, False] * 20
    w = [3.0, 1.0] * 20
    m = IsotonicCalibrator().fit(p, y, sample_weight=w).map_
    assert isinstance(m, PiecewiseLinear) and m(0.5) == pytest.approx(0.75)


def test_weights_drop_only_their_own_bad_pairs() -> None:
    import numpy as np

    cal = IsotonicCalibrator()
    p, y = overconfident(200)
    p[7] = float("nan")
    for w in ([1.0] * 200, np.ones(200, dtype=np.float32), np.ones(200, dtype=np.int64)):
        m = cal.fit(p, y, sample_weight=w).map_
        assert isinstance(m, PiecewiseLinear) and m.n == 199
    zero = cal.fit([0.5] * 40, [True, False] * 20, sample_weight=[0.0] * 40).insufficient_
    assert isinstance(zero, Insufficient) and zero.n == 0
    unaligned = cal.fit([0.5] * 40, [True, False] * 20, sample_weight=[1.0]).insufficient_
    assert isinstance(unaligned, Insufficient) and "not aligned" in unaligned.reason


def test_weights_walk_positions_not_labels() -> None:
    pd = pytest.importorskip("pandas")
    p, y = overconfident(200)
    idx = range(1000, 1200)
    m = (
        IsotonicCalibrator()
        .fit(pd.Series(p, index=idx), pd.Series(y, index=idx), sample_weight=pd.Series([1.0] * 200, index=idx))
        .map_
    )
    assert isinstance(m, PiecewiseLinear) and m.n == 200
