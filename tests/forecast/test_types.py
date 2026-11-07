from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from llm_ports.forecast import Forecast, Resolved, valid_p

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def make(**kw: object) -> Forecast:
    base: dict[str, object] = {
        "author": "author-a",
        "subject": "SPY",
        "kind": "direction",
        "forecast_time": T0,
        "resolve_time": T0 + timedelta(days=1),
        "text": "SPY likely higher into Friday",
        "hedge": "likely",
    }
    base.update(kw)
    return Forecast(**base)  # type: ignore[arg-type]  # the test varies fields by name


def test_a_forecast_keeps_what_was_said() -> None:
    f = make(probability=0.7, was_generated_by="pack@abc123", used=["post-1"], resolution_criteria="close > open")
    assert (f.probability, f.hedge, f.was_generated_by) == (0.7, "likely", "pack@abc123")
    assert f.used == ("post-1",) and f.resolution_criteria == "close > open"
    r = Resolved(claim_reviewed=f, resolution="yes", actual_resolve_time=f.resolve_time)
    assert r.outcome is True and r.claim_reviewed is f and r.actual_resolve_time == f.resolve_time


def test_a_resolution_reads_as_an_outcome_or_none() -> None:
    f = make()
    assert Resolved(claim_reviewed=f, resolution="no", actual_resolve_time=T0).outcome is False
    assert Resolved(claim_reviewed=f, resolution="annulled", actual_resolve_time=T0).outcome is None
    with pytest.raises(ValueError, match="yes, no or annulled"):
        Resolved(claim_reviewed=f, resolution="true", actual_resolve_time=T0)  # type: ignore[arg-type]  # refused value


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"author": ""}, "author"),
        ({"text": "  "}, "text"),
        ({"resolve_time": T0 - timedelta(seconds=1)}, "before"),
        ({"hedge": None}, "probability or a hedge"),
        ({"hedge": " "}, "probability or a hedge"),
        ({"probability": 1.2}, "finite probability"),
        ({"probability": float("nan")}, "finite probability"),
    ],
)
def test_a_malformed_forecast_is_refused_at_construction(kw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make(**kw)


def test_a_bare_probability_needs_no_hedge() -> None:
    assert make(hedge=None, probability=0.0).probability == 0.0


@pytest.mark.parametrize(
    ("v", "ok"),
    [(0.0, True), (1, True), (0.5, True), (True, False), (-0.1, False), (float("inf"), False), ("0.5", False)],
)
def test_valid_p(v: object, ok: bool) -> None:
    assert valid_p(v) is ok


def test_valid_p_reads_any_real_but_not_bool_or_decimal() -> None:
    from decimal import Decimal
    from fractions import Fraction

    assert valid_p(Fraction(1, 3))
    assert not valid_p(Decimal("0.4")) and not valid_p(True) and not valid_p("0.5")


def test_a_naive_and_an_aware_time_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="naive"):
        make(resolve_time=datetime(2026, 1, 6))  # the naive time under test


def test_a_number_past_float_range_is_not_finite() -> None:
    from fractions import Fraction

    from llm_ports.forecast import finite_real

    assert not finite_real(10**400) and not finite_real(Fraction(10**400)) and not valid_p(10**400)
