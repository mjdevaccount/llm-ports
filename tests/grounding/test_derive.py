"""The derivation re-check: the closed grammar, the chain's parts held to each other, and the shown figure."""

from __future__ import annotations

import pytest

from llm_ports.grounding.derive import VERDICTS, Tolerance, evaluate, recheck

FACTS = {"reading": 590.4, "baseline": 584.2, "high": 600.0, "low": 580.0, "rate": 0.142}
GAP = "abs(reading - baseline) / reading * 100 = abs(590.4 - 584.2) / 590.4 * 100 = 1.050136"


class TestEvaluate:
    @pytest.mark.parametrize(
        ("expr", "value"),
        [
            ("1 + 2 * 3", 7.0),
            ("-4 / 2", -2.0),
            ("+3 - 1", 2.0),
            ("abs(2 - 5)", 3.0),
            ("sqrt(16)", 4.0),
            ("1 < 2 <= 2 < 3", True),
            ("3 > 2 >= 4", False),
        ],
    )
    def test_the_grammar_evaluates(self, expr: str, value: object) -> None:
        assert evaluate(expr) == value

    def test_names_resolve_from_the_mapping(self) -> None:
        assert evaluate("high - low", FACTS) == pytest.approx(20.0)

    @pytest.mark.parametrize(
        "expr",
        [
            "2 ** 10",  # power is outside the grammar
            "__import__('os')",
            "max(1, 2)",
            "abs(1, 2)",
            "abs(x=1)",
            "'a' + 'b'",
            "True + 1",
            "(1 < 2) + 1",  # a comparison inside arithmetic
            "1 == 1",
            "x.y",
            "unknown + 1",
            "1 / 0",
            "sqrt(-1)",
            "1e308 * 1e308",
            "1 +",
        ],
    )
    def test_anything_else_is_none(self, expr: str) -> None:
        assert evaluate(expr) is None


class TestRecheck:
    def test_a_correct_chain_and_figure_hold(self) -> None:
        r = recheck(GAP, "1.05%", tol=Tolerance(absolute=0.05), facts=FACTS)
        assert (r.verdict, r.value) == ("holds", pytest.approx(1.050136, abs=1e-6))

    def test_a_shifted_figure_breaks(self) -> None:
        # the same chain, the reply's figure moved: what a planted number shift must do
        assert recheck(GAP, "3.15%", tol=Tolerance(absolute=0.05), facts=FACTS).verdict == "breaks"

    def test_a_rounding_holds_with_no_tolerance(self) -> None:
        assert recheck(GAP, "1.1%").verdict == "holds"
        assert recheck(GAP, "1.2%").verdict == "breaks"

    def test_a_scaled_figure_reads_at_its_marker(self) -> None:
        op = "size = 4820000000.0 = 4820000000.0"
        assert recheck(op, "$4.82B", tol=Tolerance(relative=0.01)).verdict == "holds"
        assert recheck(op, "$9.64B", tol=Tolerance(relative=0.01)).verdict == "breaks"

    def test_a_fraction_shown_as_a_percentage_holds(self) -> None:
        assert recheck("share = 0.7486 = 0.7486", "75%").verdict == "holds"

    def test_a_percent_figure_is_judged_on_the_reading_it_plainly_means(self) -> None:
        # "0.30%" beside 0.0974 is 0.30, far outside; its fraction reading 0.003 must not be what the band sees
        op = "abs(a - b) / b * 100 = abs(750.0 - 749.27) / 749.27 * 100 = 0.0974"
        assert recheck(op, "0.30%", tol=Tolerance(absolute=0.05)).verdict == "breaks"

    def test_the_band_abstains_between_tolerance_and_its_multiple(self) -> None:
        tol = Tolerance(absolute=0.05)
        assert recheck(GAP, "1.13%", tol=tol).verdict == "abstain"  # 0.08 off: outside 0.05, inside 0.10
        assert recheck(GAP, "1.13%", tol=tol, band=1.0).verdict == "breaks"
        assert recheck(GAP, "1.20%", tol=tol).verdict == "breaks"

    def test_a_negative_figure_keeps_its_sign(self) -> None:
        op = "low - high = 580.0 - 600.0 = -20.0"
        assert recheck(op, "-20").verdict == "holds"
        assert recheck(op, "20", tol=Tolerance(absolute=0.5)).verdict == "breaks"

    def test_a_comparison_chain_holds_when_true(self) -> None:
        assert recheck("low < reading < high = 580.0 < 590.4 < 600.0 = True", facts=FACTS).verdict == "holds"
        assert recheck("high < reading = 600.0 < 590.4 = True").verdict == "breaks"  # states True, computes False
        assert recheck("high < reading = 600.0 < 590.4 = False").verdict == "breaks"  # a false comparison
        assert recheck("750.51 < 751.00").verdict == "holds"  # a bare comparison states its own result
        assert recheck("751.00 < 750.51").verdict == "breaks"

    def test_a_wrong_written_result_breaks(self) -> None:
        r = recheck("high - low = 600.0 - 580.0 = 25.0")
        assert r.verdict == "breaks" and "computes 20.0" in r.reason

    def test_the_written_result_may_round_to_its_decimals(self) -> None:
        assert recheck("rate * 100 = 0.142 * 100 = 14.2").verdict == "holds"
        assert recheck("1 / 3 = 0.333333").verdict == "holds"

    def test_a_wrong_substitution_breaks_against_the_facts(self) -> None:
        op = "high - low = 610.0 - 580.0 = 30.0"  # 610 is not the fact
        assert recheck(op).verdict == "holds"  # alone, the arithmetic is right
        assert recheck(op, facts=FACTS).verdict == "breaks"

    def test_a_name_missing_from_the_facts_is_unchecked(self) -> None:
        assert recheck("other - low = 1.0 - 580.0 = -579.0", facts=FACTS).verdict == "unchecked"

    @pytest.mark.parametrize(
        "op",
        [
            "",
            "2026-11-11T23:30Z + 13h = 2026-11-12T12:30Z",
            "|465.0 - 463.96| = 1.04",
            "a regime = pin",
            "1 + 2",  # no result stated
            "1 + 2 = 3 (see above)",
        ],
    )
    def test_what_code_cannot_read_is_unchecked(self, op: str) -> None:
        assert recheck(op).verdict == "unchecked"

    def test_a_comment_after_a_colon_is_dropped(self) -> None:
        assert recheck("290.00 = 290.00: trigger and target coincide").verdict == "holds"

    def test_a_shown_text_with_no_figure_is_unchecked(self) -> None:
        assert recheck(GAP, "about one percent").verdict == "unchecked"

    def test_a_chain_with_no_shown_figure_holds_when_it_computes(self) -> None:
        assert recheck(GAP).verdict == "holds"

    def test_every_verdict_is_named(self) -> None:
        assert set(VERDICTS) == {"holds", "breaks", "abstain", "unchecked"}


class TestTolerance:
    def test_of_reads_either_key(self) -> None:
        assert Tolerance.of({"abs": 0.05}) == Tolerance(absolute=0.05)
        assert Tolerance.of({"rel": 0.01}) == Tolerance(relative=0.01)
        assert Tolerance.of({"abs": 0}) == Tolerance(absolute=0.0)

    @pytest.mark.parametrize("raw", [None, {}, {"abs": "x"}, {"other": 1}])
    def test_of_is_none_without_a_number(self, raw: dict[str, object] | None) -> None:
        assert Tolerance.of(raw) is None

    def test_relative_scales_with_the_value(self) -> None:
        tol = Tolerance(relative=0.05)
        assert tol.allows(104.0, 100.0) and not tol.allows(106.0, 100.0)
        assert tol.allows(106.0, 100.0, widen=2.0)
