from __future__ import annotations

import json

import pytest

from llm_ports.forecast import Calibrated, Insufficient, PiecewiseLinear
from llm_ports.forecast.calibrate import usable


def test_piecewise_linear_interpolates_and_clamps() -> None:
    m = PiecewiseLinear(xs=(0.2, 0.6), ys=(0.1, 0.5), n=40)
    assert isinstance(m, Calibrated)
    assert m(0.0) == 0.1 and m(1.0) == 0.5
    assert m(0.4) == pytest.approx(0.3)
    assert m(0.6) == 0.5
    assert m(float("nan")) == 0.5  # the middle knot


def test_piecewise_linear_round_trips_through_json() -> None:
    m = PiecewiseLinear(xs=(0.1, 0.5, 0.9), ys=(0.2, 0.2, 0.7), n=12)
    assert PiecewiseLinear.from_dict(json.loads(json.dumps(m.as_dict()))) == m
    assert PiecewiseLinear.from_dict({"xs": [0.5], "ys": [0.4]}).n == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"xs": (), "ys": ()},
        {"xs": (0.1,), "ys": (0.1, 0.2)},
        {"xs": (0.1, 1.5), "ys": (0.1, 0.2)},
        {"xs": (0.5, 0.5), "ys": (0.1, 0.2)},
        {"xs": (0.1, 0.5), "ys": (0.3, 0.2)},
    ],
)
def test_a_malformed_map_is_refused(kw: dict[str, tuple[float, ...]]) -> None:
    with pytest.raises(ValueError):
        PiecewiseLinear(**kw)


@pytest.mark.parametrize("data", [{}, {"xs": None, "ys": [0.1]}, {"xs": [0.9, 0.1], "ys": [0.1, 0.2]}])
def test_from_dict_refuses_what_as_dict_never_wrote(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PiecewiseLinear.from_dict(data)


def test_usable_drops_invalid_and_names_the_shortfall() -> None:
    assert usable([0.1, float("nan"), 0.9], [False, True, True], need=2) == ([0.1, 0.9], [False, True])
    short = usable([0.1], [True], need=2)
    assert isinstance(short, Insufficient) and short.n == 1 and "< 2" in short.reason
    one = usable([0.1, 0.2, 0.3], [True, True, True], need=2)
    assert isinstance(one, Insufficient) and one.reason == "one outcome only"
