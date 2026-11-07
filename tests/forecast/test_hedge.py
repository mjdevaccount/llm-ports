from __future__ import annotations

import json
import math
import random

import pytest

from llm_ports.forecast import HedgeMap
from llm_ports.forecast.hedge import DEFAULT_STRENGTH, MAX_STRENGTH, _tau2, normalise


def rows(source: str, hedge: str, hits: int, n: int) -> list[tuple[str, str, bool]]:
    return [(source, hedge, i < hits) for i in range(n)]


def fit(data: list[tuple[str, str, bool]], strength: float | None = None) -> HedgeMap:
    return HedgeMap(strength=strength).fit([(src, h) for src, h, _ in data], [o for _, _, o in data])


def test_normalise() -> None:
    assert normalise("  Very   LIKELY ") == "very likely"


def test_the_same_word_means_different_things_from_different_sources() -> None:
    data = rows("a", "likely", 160, 200) + rows("b", "Likely", 60, 200) + rows("c", "likely", 110, 200)
    hm = fit(data)
    pa, pb, pc = hm.probability("a", "likely"), hm.probability("b", "LIKELY"), hm.probability("c", "likely")
    assert pa is not None and pb is not None and pc is not None
    assert pa > pc > pb
    assert pa == pytest.approx(0.8, abs=0.02)  # lots of data: barely shrunk
    assert hm.strengths_["likely"] < MAX_STRENGTH


def test_a_thin_source_is_shrunk_toward_the_pool() -> None:
    data = rows("a", "could", 30, 100) + rows("b", "could", 35, 100) + rows("thin", "could", 2, 2)
    hm = fit(data)
    thin = hm.probability("thin", "could")
    assert thin is not None and thin < 0.6  # two lucky calls do not make "could" mean 1.0
    assert hm.probability("never-used-it", "could") == pytest.approx(hm.pooled_["could"])


def test_agreeing_sources_are_pooled_and_a_single_source_uses_the_fallback() -> None:
    agree = fit(rows("a", "maybe", 50, 100) + rows("b", "maybe", 50, 100))
    assert agree.strengths_["maybe"] == MAX_STRENGTH
    single = fit(rows("a", "watch for", 3, 10))
    assert single.strengths_["watch for"] == DEFAULT_STRENGTH
    fixed = fit(rows("a", "x", 1, 2) + rows("b", "x", 0, 2), strength=4.0)
    assert fixed.strengths_["x"] == 4.0


def test_an_unknown_word_has_no_probability_and_empty_hedges_are_skipped() -> None:
    hm = fit([("a", "  ", True), ("a", "likely", True)])
    assert hm.probability("a", "certain") is None
    assert [r.hedge for r in hm.rows()] == ["likely"]


def test_rows_and_json_round_trip() -> None:
    hm = fit(rows("a", "likely", 7, 10) + rows("b", "likely", 3, 10))
    [ra, rb] = hm.rows()
    assert (ra.source, ra.n, ra.hits) == ("a", 10, 7) and rb.pooled == hm.pooled_["likely"]
    back = HedgeMap.from_dict(json.loads(json.dumps(hm.as_dict())))
    assert back.probability("a", "likely") == hm.probability("a", "likely")


@pytest.mark.parametrize("data", [{}, {"cells": [["a"]], "pooled": {}}, {"cells": [], "pooled": []}])
def test_from_dict_refuses_what_as_dict_never_wrote(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HedgeMap.from_dict(data)


def test_the_estimated_strength_recovers_the_true_prior() -> None:
    rng = random.Random(7)
    m, s = 0.6, 20.0
    data: list[tuple[str, str, bool]] = []
    for j in range(200):
        rate = rng.betavariate(m * s, (1 - m) * s)
        data += [(f"v{j}", "likely", rng.random() < rate) for _ in range(30)]
    assert 14.0 < fit(data).strengths_["likely"] < 28.0


def test_sources_that_each_spoke_once_fall_back() -> None:
    hm = fit([("a", "likely", True), ("b", "likely", False), ("c", "likely", True)])
    assert hm.strengths_["likely"] == DEFAULT_STRENGTH


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_an_unusable_strength_is_refused(bad: float) -> None:
    with pytest.raises(ValueError):
        fit(rows("a", "likely", 1, 2), strength=bad)
    with pytest.raises(ValueError):
        HedgeMap.from_dict({"cells": [["a", "likely", 1, 2]], "pooled": {"likely": 0.5}, "strengths": {"likely": bad}})


def test_from_dict_refuses_a_pooled_rate_outside_0_1() -> None:
    with pytest.raises(ValueError):
        HedgeMap.from_dict({"cells": [], "pooled": {"likely": 7.0}})


def test_a_hedge_that_is_not_a_string_is_skipped() -> None:
    hm = fit([("a", None, True), ("a", "likely", True)])  # type: ignore[list-item]  # the guard under test
    assert hm.probability("a", None) is None  # type: ignore[arg-type]  # the guard under test
    assert list(hm.pooled_) == ["likely"]


def test_the_between_source_variance_is_unbiased_at_small_n() -> None:
    # Three sources, ten uses each, true tau^2 = 0.04 (m = 0.5, s = 5.25). The estimator without
    # the design-term correction averages about 0.026 here.
    rng = random.Random(11)
    m, s, ests = 0.5, 5.25, []
    for _ in range(4000):
        cells = {}
        for j in range(3):
            rate = rng.betavariate(m * s, (1 - m) * s)
            cells[str(j)] = (sum(rng.random() < rate for _ in range(10)), 10)
        t = _tau2(cells, sum(k for k, _ in cells.values()) / 30)
        assert t is not None
        ests.append(t)
    assert 0.036 < sum(ests) / len(ests) < 0.052


@pytest.mark.parametrize("cell", [["a", "likely", 0, -10], ["a", "likely", 50, 1], ["a", "likely", -1, 3]])
def test_from_dict_refuses_impossible_counts(cell: list[object]) -> None:
    with pytest.raises(ValueError):
        HedgeMap.from_dict({"cells": [cell], "pooled": {"likely": 0.5}, "strengths": {"likely": 10.0}})


def test_a_bool_is_not_a_pooled_rate_and_any_real_is_a_strength() -> None:
    with pytest.raises(ValueError):
        HedgeMap()._set({}, {"x": True}, {})
    from fractions import Fraction

    hm = fit(rows("a", "likely", 1, 2), strength=Fraction(5))  # type: ignore[arg-type]  # any Real
    assert type(hm.strengths_["likely"]) is float and json.dumps(hm.as_dict())
    direct = HedgeMap()._set({("a", "x"): (1, 2)}, {"x": Fraction(1, 2)}, {})  # type: ignore[dict-item]
    assert type(direct.pooled_["x"]) is float and json.dumps(direct.as_dict())
    with pytest.raises(ValueError):
        HedgeMap()._set({("a", "x"): ("1", "2")}, {"x": 0.5}, {})  # type: ignore[dict-item]


def test_hedge_map_is_a_scikit_learn_estimator() -> None:
    hm = HedgeMap(strength=4.0)
    assert hm.get_params() == {"strength": 4.0}
    assert hm.set_params(strength=2.0) is hm and hm.strength == 2.0
    with pytest.raises(ValueError, match="invalid parameter"):
        hm.set_params(alpha=1.0)
    assert hm.probability("a", "likely") is None and hm.rows() == () and hm.as_dict()["cells"] == []
    hm.fit([("a", "likely"), ("a", "could")], [True, False])
    [p_likely, p_unknown] = hm.predict_proba([("a", "likely"), ("a", "certain")])
    assert p_unknown is None and p_likely is not None
    assert p_likely[1] == pytest.approx(hm.probability("a", "likely")) and sum(p_likely) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="2 rows"):
        hm.fit([("a", "likely"), ("b", "likely")], [True])


def test_clone_rebuilds_an_unfitted_copy() -> None:
    base = pytest.importorskip("sklearn.base")
    hm = fit(rows("a", "likely", 1, 2), strength=3.0)
    twin = base.clone(hm)
    assert twin.get_params() == {"strength": 3.0} and not hasattr(twin, "cells_")
