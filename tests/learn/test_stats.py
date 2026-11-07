"""stats: agreement and false-discovery control."""

from __future__ import annotations

import random

from llm_ports.evals import calibration
from llm_ports.learn import stats


def test_calibration_re_exports_the_same_functions() -> None:
    assert calibration.cohen_kappa is stats.cohen_kappa
    assert calibration.gwet_ac1 is stats.gwet_ac1
    assert calibration.YES == stats.YES


def test_agreement_is_undefined_under_two_pairs() -> None:
    assert stats.cohen_kappa([("yes", "yes")]) is None
    assert stats.gwet_ac1([]) is None


class TestBhFdr:
    def test_an_empty_family_has_no_survivor(self) -> None:
        assert stats.bh_fdr([]) == []
        assert stats.bh_fdr([None, None]) == [False, False]

    def test_plain_bh_on_a_textbook_family(self) -> None:
        # m=4, alpha=0.10: bars 0.025, 0.05, 0.075, 0.10 -> the step-up keeps ranks 1..3.
        assert stats.bh_fdr([0.01, 0.04, 0.07, 0.5], 0.10, dependent=False) == [True, True, True, False]

    def test_by_is_never_looser_than_bh(self) -> None:
        rng = random.Random(7)
        for _ in range(500):
            ps: list[float | None] = [rng.random() ** 3 for _ in range(rng.randint(1, 12))]
            by, bh = stats.bh_fdr(ps), stats.bh_fdr(ps, dependent=False)
            assert all(b <= h for b, h in zip(by, bh, strict=True))

    def test_a_none_never_passes_nor_loosens_the_bar(self) -> None:
        with_none = stats.bh_fdr([0.01, None, 0.04], 0.10, dependent=False)
        without = stats.bh_fdr([0.01, 0.04], 0.10, dependent=False)
        assert with_none == [without[0], False, without[1]]

    def test_the_mask_is_in_input_order(self) -> None:
        assert stats.bh_fdr([0.9, 0.0001], 0.05) == [False, True]


def _table(a: int, b: int, c: int, d: int) -> list[tuple[str, str]]:
    """A 2x2 table as pairs: rows the first labeller (yes, no), columns the second."""
    return [("yes", "yes")] * a + [("yes", "no")] * b + [("no", "yes")] * c + [("no", "no")] * d


def test_kappa_ci_matches_statsmodels_fleiss_cohen_everitt() -> None:
    # Reference values: statsmodels.stats.inter_rater.cohens_kappa (kappa_low, kappa_upp).
    for table, low, high in (
        ((8, 2, 3, 7), 0.122357, 0.877643),
        ((11, 3, 1, 5), 0.196013, 0.934422),
        ((20, 5, 10, 15), 0.151092, 0.648908),
    ):
        got = stats.kappa_ci(_table(*table))
        assert got is not None
        assert abs(got[0] - low) < 1e-5 and abs(got[1] - high) < 1e-5


def test_kappa_ci_is_none_where_kappa_is_and_clipped_to_minus_one_one() -> None:
    assert stats.kappa_ci([("yes", "yes")]) is None
    assert stats.kappa_ci(_table(5, 0, 0, 0)) is None
    low, high = stats.kappa_ci(_table(3, 0, 0, 3)) or (0.0, 0.0)
    assert -1.0 <= low <= high <= 1.0 and high == 1.0
