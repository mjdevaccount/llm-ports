"""numbers: the one digit-run and spelled-number read every grader and validator shares."""

from __future__ import annotations

from llm_ports import numbers


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


def test_uncited_is_the_one_number_law() -> None:
    spans = [("Close 580", "SPY closed at 580.5, up 3%")]
    assert numbers.uncited("at 580.5 and 580, up 3 percent", spans) == []
    assert numbers.uncited("at 581, up three percent", spans) == ["581", "three"]
    assert numbers.uncited("up 3 percent", [("", "up 3 points")]) == ["percent"]
    assert numbers.uncited("at 580", []) == ["580"]
    assert numbers.uncited("no figures here", []) == []
