"""adapters.venn_abers: a small history is bracketed and pulled toward what happened; failures are named."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("venn_abers")

from llm_ports.adapters.venn_abers import VennAbersCalibrator, VennAbersFit
from llm_ports.forecast import Insufficient


def overconfident(n: int, seed: int = 3) -> tuple[list[float], list[bool]]:
    """A voice whose stated p overclaims: the truth is p**2 (0.68 stated hits about 0.46)."""
    rng = random.Random(seed)
    ps = [rng.random() for _ in range(n)]
    return ps, [rng.random() < p * p for p in ps]


def test_venn_abers_brackets_a_small_history() -> None:
    ps, ys = overconfident(60)
    fitted = VennAbersCalibrator().fit(ps, ys)
    assert isinstance(fitted, VennAbersFit) and fitted.n == 60
    lo, hi = fitted.interval(0.8)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= fitted(0.8) <= hi
    assert fitted(0.8) < 0.8  # overclaiming is pulled down
    assert fitted.interval(float("nan")) == (0.5, 0.5) and fitted(2.0) == 0.5


def test_venn_abers_refuses_and_survives_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(VennAbersCalibrator().fit([0.2, 0.8], [False, True]), Insufficient)
    ps, ys = overconfident(60)
    fitted = VennAbersCalibrator().fit(ps, ys)
    assert isinstance(fitted, VennAbersFit)

    class Broken:
        def fit(self, *a: object) -> None:
            raise ValueError("bad")

        def predict_proba(self, *a: object) -> object:
            raise ValueError("bad")

    assert VennAbersFit(model=Broken(), n=1).interval(0.3) == (0.3, 0.3)
    import llm_ports.adapters.venn_abers as va

    monkeypatch.setattr(va, "VennAbers", Broken)
    out = VennAbersCalibrator().fit(ps, ys)
    assert isinstance(out, Insufficient) and "venn-abers fit failed" in out.reason


def test_venn_abers_degenerate_bracket_keeps_the_stated_p() -> None:
    class Degenerate:
        def predict_proba(self, *a: object) -> tuple[None, list[list[float]]]:
            return None, [[1.0, 0.0]]  # 1 - p0 + p1 == 0

    assert VennAbersFit(model=Degenerate(), n=1)(0.4) == 0.4
