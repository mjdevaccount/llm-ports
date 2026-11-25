"""numbers: the one digit-run and spelled-number read every grader and validator shares."""

from __future__ import annotations

from typing import Any

import pytest

from llm_ports.grounding import numbers


def test_digits_drops_thousands_separators_and_keeps_decimals() -> None:
    assert numbers.digits("up 4.1B, 7,773.95 and 500") == {"4.1", "7773.95", "500"}
    assert numbers.digits("") == set() and numbers.digits(None) == set()  # type: ignore[arg-type]


def test_a_digit_run_is_compared_by_value_so_trailing_zeros_never_fail_an_honest_figure() -> None:
    assert numbers.canon("2.20") == "2.2" and numbers.canon("7.0") == "7"
    assert numbers.canon("1,234.50") == "1234.5" and numbers.canon("100") == "100"
    assert numbers.digits("peak 2.20, held 7.0") == {"2.2", "7"}
    assert numbers.uncited("peak 2.20", [("", "peak_px 2.2")]) == []


def test_spelled_catches_number_words_and_their_fragments() -> None:
    assert numbers.spelled("Nine one point seven million") == {"nine", "seven", "million"}
    assert numbers.spelled("an ighteen-year high, from ills to ills") == {"ighteen", "ills"}
    assert numbers.spelled("eighty eight") == {"eighty", "eight"}
    assert numbers.spelled("the night is light") == set()


def test_redact_fills_every_number_with_the_callers_hole() -> None:
    assert numbers.redact("SPY at 580.5, up 3%", hole="[n]") == "SPY at [n], up [n]%"


def test_loose_names_every_unlicensed_figure_in_order_with_repeats() -> None:
    allowed = numbers.digits("the level is 580.50, nine voters")
    words = "the level is 580.50\nnine voters"
    assert numbers.loose("held 580.5 then 581 and 581, nine then ten", allowed, words) == ["581", "581", "ten"]
    assert numbers.loose("an ighteen-year high", allowed, words) == ["ighteen"]
    assert numbers.loose("peak 2.20", {"2.2"}) == []  # by value, never by spelling
    assert numbers.loose("", allowed, words) == [] and numbers.loose(None, set()) == []  # type: ignore[arg-type]


def test_a_percent_sign_in_the_inputs_licenses_the_word_and_a_newline_join_never_glues_words() -> None:
    assert numbers.loose("up three percent", {"3"}, "up 3%") == ["three"]
    assert numbers.loose("up three percent", {"3"}, "up 3 points") == ["three", "percent"]
    assert numbers.loose("ten", set(), "Bit\nen route") == ["ten"]
    assert numbers.loose("ten", set(), "Biten route") == []  # glued, the join would lie


def test_uncited_is_loose_in_the_cited_spans_shape() -> None:
    spans = [("Close 580", "SPY closed at 580.5, up 3%")]
    assert numbers.uncited("at 580.5 and 580, up 3 percent", spans) == []
    assert numbers.uncited("at 581, up three percent", spans) == ["581", "three"]
    assert numbers.uncited("up 3 percent", [("", "up 3 points")]) == ["percent"]
    assert numbers.uncited("at 580", []) == ["580"]
    assert numbers.uncited("no figures here", []) == []


def test_shorten_cuts_on_a_word_and_marks_the_cut() -> None:
    head = "CHINA SAYS IT WILL HONOUR BUSAN ARRANGEMENT ON TARIFFS"
    out = numbers.shorten(head, 40)
    assert out == "CHINA SAYS IT WILL HONOUR BUSAN…" and len(out) <= 40
    assert numbers.shorten("  two   spaces\ncollapse ", 40) == "two spaces collapse"  # fits: whole, no mark
    assert numbers.shorten(None, 10) == ""  # type: ignore[arg-type]


def test_shorten_never_leaves_part_of_a_number() -> None:
    assert numbers.shorten("Fed hikes 25bp tonight", 14) == "Fed hikes…"
    assert numbers.shorten("7,773.95target", 8) == "…"  # the one long word is a number: step back whole
    assert numbers.shorten("abcdef12345", 9) == "abcdef…"
    assert numbers.shorten("abcdefghij", 6) == "abcde…"  # no number in the way: a plain hard cut
    for width in range(2, 30):
        out = numbers.shorten("SPY 580.50 to 7,773.95 then 25bp", width)
        assert numbers.loose(out, numbers.digits("SPY 580.50 to 7,773.95 then 25bp")) == [], out


def test_shorten_refuses_a_width_with_no_room_before_the_placeholder() -> None:
    with pytest.raises(ValueError, match="no room"):
        numbers.shorten("anything", 1)


class TestRerenders:
    """One input value shown again, coarser: licensed. Anything else: not. Both directions."""

    @pytest.mark.parametrize(
        ("run", "after", "value"),
        [
            ("0.59", " R", "0.5918"),  # rounded to the decimals written
            ("14.88", " points", "14.878"),
            ("60.67", "", "60.666"),
            ("0.24", "% away", "0.2409"),  # a rounded value that already was a percentage
            ("13.5", "%", "13.49"),
            ("13", " percent", "13.49"),
            ("75", "% of sessions", "0.7486"),  # a fraction shown as a percentage
            ("74.9", "pct", "0.7486"),
            ("19.8", "B in gamma", "19813893517"),  # a scale letter directly after the run
            ("19.8", " billion", "19813893517"),
            ("20", "B", "19813893517"),
            ("3", "k", "3000"),  # exact at the scale
            ("1.2", "M", "1234567"),
            ("2.5", "", "2.45"),  # half up
            ("2.4", "", "2.45"),  # half even
        ],
    )
    def test_a_coarser_rendering_of_one_value_is_licensed(self, run: str, after: str, value: str) -> None:
        assert numbers.rerenders(run, after, {value}), f"{run}{after!r} shows {value}"
        assert numbers.loose(f"it was {run}{after}", {value}, rerender=True) == []
        assert numbers.loose(f"it was {run}{after}", {value}) != [], "the belt reading stays exact"

    @pytest.mark.parametrize(
        ("run", "after", "value"),
        [
            ("15", "-minute retest", "14.878"),  # a bare integer is never a rounding
            ("48", " minutes", "48.1"),
            ("28.00", "", "28.89"),  # the decimals written are the precision claimed
            ("0.7", "", "0.5918"),
            ("21.99", "", "21.9963"),  # truncation is not rounding: 21.9963 rounds to 22.00
            ("2.6", "%", "26"),  # a slipped decimal point is not a scale
            ("36", "%", "34.3"),  # nor is arithmetic over inputs
            ("19.8", " m", "19813893517"),  # a spaced letter is a unit
            ("19.8", "M", "19813893517"),  # the wrong scale
            ("0", "%", "0.004"),  # a marked integer rounds a magnitude of 1 or more only
            ("1.2.3", "", "1.2"),  # a digit run that is not a number
        ],
    )
    def test_an_invented_or_computed_figure_is_not(self, run: str, after: str, value: str) -> None:
        assert not numbers.rerenders(run, after, {value})

    def test_loose_names_the_invented_figure_and_keeps_the_rounded_ones(self) -> None:
        allowed = numbers.digits("realized_r 0.5918, room_down 14.878, vix 28.89, net_gex -19813893517")
        text = "Realized 0.59R with 14.88 of room and -19.8B of gamma; wait for a 15-minute retest below 28.00."
        assert numbers.loose(text, allowed, rerender=True) == ["15", "28"]

    def test_the_scale_word_is_licensed_only_when_the_rendering_needed_it(self) -> None:
        assert numbers.loose("about 19.8 billion", {"19813893517"}, rerender=True) == []
        assert numbers.loose("about 14.88 million", {"14.878"}, rerender=True) == ["million"], "an invented scale"
        assert numbers.loose("19.8 billion, then a billion more", {"19813893517"}, rerender=True) == ["billion"]

    @pytest.mark.parametrize(
        ("text", "inputs", "unlicensed"),
        [
            # timeframe and unit shorthand is not a scale: only a capital letter, or `k`, is
            ("wait for a 15m retest", "volume 15234567", ["15"]),
            ("on the 5m chart", "volume 5234567", ["5"]),
            ("a 5mm gap", "volume 5234567", ["5"]),
            ("down 2b", "net 2000000000", ["2"]),
            ("volume 15.2M, 3k lots, 2.1bn notional", "15234567 3000 2100000000", []),
            # times 100 shows a fraction; a whole input moved is an invented figure
            ("a 100% chance", "step 1. read the tape", ["100"]),
            ("up 200%", "2 DTE", ["200"]),
            ("up 200.0%", "2 DTE", ["200"]),
            ("75% of sessions, 150% of the range", "p_up 0.7486 ratio 1.5", []),
            # a newline ends the figure: the marker is on the next line, so it marks nothing
            ("was 75" + chr(10) + "percent of", "p_up 0.7486", ["75", "percent"]),
            ("19.8" + chr(10) + "billion", "net 19813893517", ["19.8", "billion"]),
            ("was 75 percent of", "p_up 0.7486", []),
        ],
    )
    def test_a_marker_is_read_narrowly(self, text: str, inputs: str, unlicensed: list[str]) -> None:
        assert numbers.loose(text, numbers.digits(inputs), inputs, rerender=True) == unlicensed

    @pytest.mark.parametrize(
        ("text", "inputs", "unlicensed"),
        [
            # the run equals another input (a date's 20, a dte of 75) and re-renders the one meant
            ("net gex is -20 billion", "date 2026-11-20 net_gex -19813893517", []),
            ("about 75 percent", "dte 75 p_up 0.7486", []),
            ("about 75 pct", "dte 75 p_up 0.7486", []),
            # and with no input for the scale to re-render, the scale word is still invented
            ("about 20 billion", "date 2026-11-20", ["billion"]),
            ("about 20 thousand", "date 2026-11-20 lots 21000", ["thousand"]),
        ],
    )
    def test_an_equal_run_still_has_its_marker_licensed_by_the_value_it_renders(
        self, text: str, inputs: str, unlicensed: list[str]
    ) -> None:
        assert numbers.loose(text, numbers.digits(inputs), inputs, rerender=True) == unlicensed

    def test_a_long_prompt_and_a_long_reply_are_checked_in_bounded_time(self) -> None:
        """Each figure is one bisect over the sorted inputs, not a pass over all of them. Read as a
        count of the values looked at, so the bound does not depend on the machine."""
        inputs = {f"{i}.{i % 97:02d}" for i in range(5000)}
        values = numbers._values(inputs)
        seen: list[int] = []
        real = numbers._between

        def counted(vals: Any, lo: Any, hi: Any) -> Any:
            got = real(vals, lo, hi)
            seen.append(len(got))
            return got

        text = " ".join(f"{i}.{(i * 7) % 100:02d}%" for i in range(0, 4000, 10))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(numbers, "_between", counted)
            hits = numbers.loose(text, inputs, rerender=True)
        assert len(values) == 5000 and len(seen) <= 1200, (
            "at most three bisects a figure: as a percentage, moved, itself"
        )
        assert max(seen) <= 2 and sum(seen) < 800, f"{sum(seen)} values looked at for 400 figures over 5000 inputs"
        assert len(hits) > 300, "and the invented ones are still named"

    def test_inputs_clustered_inside_one_step_cost_one_look_a_figure(self) -> None:
        """5000 inputs in [0, 1) all sit within half a step of a small whole number. A bare integer
        is never a rounding, so it looks at none; a marked one stops at the first value that rounds."""
        inputs = {f"0.{i:04d}" for i in range(5000)}
        looked: list[int] = []
        real = numbers._rounds_to

        def counted(value: Any, shown: Any, step: Any) -> bool:
            looked.append(1)
            return real(value, shown, step)

        text = " ".join(["0 1 2 3"] * 100 + ["1%", "0.5", "2 percent"] * 100)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(numbers, "_rounds_to", counted)
            numbers.loose(text, inputs, rerender=True)
        assert len(looked) <= 2 * 300, f"{len(looked)} values rounded for 700 figures over 5000 clustered inputs"

    @pytest.mark.parametrize(
        ("text", "inputs", "unlicensed"),
        [
            # an equal run licenses the run, never a magnitude: the letter needs an input at that scale
            ("gamma is about 20B", "date 2026-11-20 spot 545.2", ["20B"]),
            ("gamma is about 20bn", "date 2026-11-20 spot 545.2", ["20bn"]),
            ("volume near 20K", "date 2026-11-20 spot 545.2", ["20K"]),
            ("about $2B of notional", "2 DTE", ["2B"]),
            ("a 0.6B gap", "ratio 0.5918", ["0.6B"]),  # a rounding is no licence for a scale either
            # ... and the word does too, even when another figure in the inputs carries it
            ("about 20 billion", "date 2026-11-20 cap 3.2 billion", ["billion"]),
            # the input shows this value at this scale, as a letter or as a word: licensed both ways
            ("net gex -1.2B", "net gex -1.2B", []),
            ("net gex of -1.2 billion", "net gex -1.2B", []),
            ("gex 1.2B", "gex 1.2 billion", []),
            ("gex near 0.59B", "gex 0.5918B", []),
            ("cap of 3.2 billion", "date 2026-11-20 cap 3.2 billion", []),
            # a percentage the value already was keeps its word, with decimals or without
            ("IV 18.4 percent", "iv 18.37", []),
            ("IV 18 percent", "iv 18", []),
        ],
    )
    def test_a_scale_after_a_licensed_run_needs_an_input_at_that_scale(
        self, text: str, inputs: str, unlicensed: list[str]
    ) -> None:
        assert numbers.loose(text, numbers.digits(inputs), inputs, rerender=True) == unlicensed

    def test_a_belt_over_printed_values_reads_no_scale(self) -> None:
        """`rerender` off is the belt: values as code printed them, and a letter after one is a unit."""
        assert numbers.loose("about 20B", {"20"}, "") == []

    def test_an_unreadable_allowed_run_is_skipped_never_raised(self) -> None:
        assert numbers.rerenders("0.59", "", {"1.2.3", "0.5918"}) is True
        assert numbers.rerenders("0.59", "", {"1.2.3"}) is False
        assert numbers.rerenders("1.5", "", {"9" * 40}) is False  # more digits than the context holds
