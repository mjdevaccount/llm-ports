"""adapters.scoringrules: agrees with the stdlib Brier; unusable input is None, never an exception."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("scoringrules")

from llm_ports.adapters import scoringrules as sr_adapter
from llm_ports.forecast.score import brier_score


def test_scoringrules_brier_matches_the_stdlib() -> None:
    ps, ys = [0.7, 0.2, 0.99], [True, False, False]
    got = sr_adapter.brier_score(ys, ps)
    assert got == pytest.approx([brier_score(y, p) for p, y in zip(ps, ys, strict=True)])


def test_interval_and_crps_scores() -> None:
    assert sr_adapter.interval_score([5.0, 7.0], [4.0, 4.0], [6.0, 6.0], 0.1) == pytest.approx([2.0, 22.0])
    crps = sr_adapter.crps_normal([0.0, 3.0], [0.0, 0.0], [1.0, 1.0])
    assert crps is not None and crps[0] < crps[1]


@pytest.mark.parametrize(
    "call",
    [
        lambda: sr_adapter.brier_score([], []),
        lambda: sr_adapter.brier_score([True], [0.1, 0.2]),
        lambda: sr_adapter.brier_score([True], [1.5]),
        lambda: sr_adapter.interval_score([1.0], [0.0], [2.0], 1.5),
        lambda: sr_adapter.interval_score([1.0], [0.0, 1.0], [2.0], 0.1),
        lambda: sr_adapter.crps_normal([1.0], [0.0], [0.0]),
        lambda: sr_adapter.crps_normal([1.0], [0.0], [float("nan")]),
    ],
)
def test_unusable_input_is_none_never_an_exception(call: Any) -> None:
    assert call() is None


def test_a_vendor_rejection_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> object:
        raise ValueError("rejected")

    for name in ("brier_score", "interval_score", "crps_normal"):
        monkeypatch.setattr(sr_adapter.sr, name, boom)
    assert sr_adapter.brier_score([True], [0.5]) is None
    assert sr_adapter.interval_score([1.0], [0.0], [2.0], 0.1) is None
    assert sr_adapter.crps_normal([1.0], [0.0], [1.0]) is None


def test_numpy_arrays_are_usable_input() -> None:
    import numpy as np

    got = sr_adapter.brier_score(np.array([True, False]), np.array([0.7, 0.2]))
    assert got is not None and got[0] == pytest.approx(brier_score(True, 0.7))
