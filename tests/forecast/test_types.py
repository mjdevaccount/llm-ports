from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from llm_ports.forecast import Forecast, Resolved, valid_p

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def make(**kw: object) -> Forecast:
    base: dict[str, object] = {
        "producer": "author-a",
        "subject": "SPY",
        "kind": "direction",
        "asserted_at": T0,
        "resolves_at": T0 + timedelta(days=1),
        "evidence": "SPY likely higher into Friday",
        "hedge": "likely",
    }
    base.update(kw)
    return Forecast(**base)  # type: ignore[arg-type]  # the test varies fields by name


def test_a_forecast_keeps_what_was_said() -> None:
    f = make(p=0.7, extracted_by="pack@abc123")
    assert (f.p, f.hedge, f.extracted_by) == (0.7, "likely", "pack@abc123")
    r = Resolved(forecast=f, outcome=True, resolved_at=f.resolves_at)
    assert r.outcome and r.forecast is f and r.resolved_at == f.resolves_at


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"producer": ""}, "producer"),
        ({"evidence": "  "}, "evidence"),
        ({"resolves_at": T0 - timedelta(seconds=1)}, "before"),
        ({"hedge": None}, "probability or a hedge"),
        ({"hedge": " "}, "probability or a hedge"),
        ({"p": 1.2}, "finite probability"),
        ({"p": float("nan")}, "finite probability"),
    ],
)
def test_a_malformed_forecast_is_refused_at_construction(kw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make(**kw)


def test_a_bare_probability_needs_no_hedge() -> None:
    assert make(hedge=None, p=0.0).p == 0.0


@pytest.mark.parametrize(
    ("v", "ok"),
    [(0.0, True), (1, True), (0.5, True), (True, False), (-0.1, False), (float("inf"), False), ("0.5", False)],
)
def test_valid_p(v: object, ok: bool) -> None:
    assert valid_p(v) is ok  # type: ignore[arg-type]  # probes non-floats on purpose
