"""adapters.venn_abers: a small history is bracketed and pulled toward what happened; failures are named."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("venn_abers")

from llm_ports.adapters.venn_abers import VennAbersCalibrator
from llm_ports.forecast import Insufficient


def overconfident(n: int, seed: int = 3) -> tuple[list[float], list[bool]]:
    """A voice whose stated p overclaims: the truth is p**2 (0.68 stated hits about 0.46)."""
    rng = random.Random(seed)
    ps = [rng.random() for _ in range(n)]
    return ps, [rng.random() < p * p for p in ps]


def test_venn_abers_brackets_a_small_history() -> None:
    ps, ys = overconfident(60)
    cal = VennAbersCalibrator().fit(ps, ys)
    assert cal.n_ == 60 and cal.insufficient_ is None
    intervals = cal.predict_interval([0.8, float("nan")])
    rows = cal.predict_proba([0.8, 2.0])
    assert intervals is not None and rows is not None
    (lo, hi), neutral = intervals
    assert 0.0 <= lo <= hi <= 1.0 and neutral == (0.5, 0.5)
    p = rows[0][1]
    assert lo <= p <= hi and p < 0.8  # overclaiming is pulled down
    assert rows[0][0] == pytest.approx(1.0 - p) and rows[1] == (0.5, 0.5)


def test_venn_abers_is_a_scikit_learn_estimator() -> None:
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    cal = VennAbersCalibrator(min_pairs=12)
    assert cal.get_params() == {"min_pairs": 12}
    assert cal.predict_proba([0.5]) is None and cal.predict_interval([0.5]) is None
    assert clone(cal).get_params() == {"min_pairs": 12}


def test_venn_abers_refuses_and_survives_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    short = VennAbersCalibrator().fit([0.2, 0.8], [False, True])
    assert isinstance(short.insufficient_, Insufficient) and short.predict_proba([0.5]) is None
    ps, ys = overconfident(60)

    class Broken:
        def fit(self, *a: object) -> None:
            raise ValueError("bad")

        def predict_proba(self, *a: object) -> object:
            raise ValueError("bad")

    cal = VennAbersCalibrator().fit(ps, ys)
    cal.model_ = Broken()
    assert cal.predict_interval([0.3]) == [(0.3, 0.3)]
    import llm_ports.adapters.venn_abers as va

    monkeypatch.setattr(va, "VennAbers", Broken)
    out = VennAbersCalibrator().fit(ps, ys)
    assert out.model_ is None and out.insufficient_ is not None and "venn-abers fit failed" in out.insufficient_.reason


def test_venn_abers_degenerate_bracket_keeps_the_stated_p() -> None:
    class Degenerate:
        def predict_proba(self, *a: object) -> tuple[None, list[list[float]]]:
            return None, [[1.0, 0.0]]  # 1 - p0 + p1 == 0

    cal = VennAbersCalibrator().fit(*overconfident(60))
    cal.model_ = Degenerate()
    rows = cal.predict_proba([0.4])
    assert rows is not None and rows[0][1] == 0.4
