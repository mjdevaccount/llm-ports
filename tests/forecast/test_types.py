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


def test_valid_p_reads_any_real_but_not_bool_or_decimal() -> None:
    from decimal import Decimal
    from fractions import Fraction

    assert valid_p(Fraction(1, 3))
    assert not valid_p(Decimal("0.4")) and not valid_p(True) and not valid_p("0.5")  # type: ignore[arg-type]


def test_a_naive_and_an_aware_time_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="naive"):
        make(resolves_at=datetime(2026, 1, 6))  # the naive time under test


def test_a_number_past_float_range_is_not_finite() -> None:
    from fractions import Fraction

    from llm_ports.forecast import finite_real

    assert not finite_real(10**400) and not finite_real(Fraction(10**400)) and not valid_p(10**400)
