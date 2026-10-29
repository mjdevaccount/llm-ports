"""grounding: one Problem, one retry text, one belt -- on a synthetic sheet (no consumer data)."""

from __future__ import annotations

from llm_ports.grounding import RETRY_PREFIX, Problem, allowed_numbers, belt, describe, loose, retry_message


def test_retry_text_names_path_reason_and_detail_only_when_there_is_one() -> None:
    probs = [Problem("claims.0", "number_not_in_span", "740"), Problem("headline", "trails off")]
    assert describe(probs) == "claims.0: number_not_in_span (740); headline: trails off"
    assert retry_message(probs) == RETRY_PREFIX + describe(probs)
    assert describe([]) == "" and retry_message([]) == RETRY_PREFIX


def test_a_problem_is_a_frozen_value() -> None:
    assert Problem("a", "b") == Problem("a", "b", "") and hash(Problem("a", "b")) == hash(Problem("a", "b"))


def test_allowed_numbers_drops_thousands_separators() -> None:
    assert allowed_numbers(["close 7,773.95 on 500 shares", "4.1B"]) == {"7773.95", "500", "4.1"}
    assert allowed_numbers([]) == set()


def test_the_belt_counts_unrendered_digits_and_spelled_numbers() -> None:
    allowed = allowed_numbers(["the level is 580.50", "nine voters"])
    words = "the level is 580.50\nnine voters"
    assert belt("held 580.50", allowed, words) == 0
    assert belt("held 581", allowed, words) == 1
    assert belt("nine voted, ten abstained", allowed, words) == 1     # "ten" no input said
    assert belt("an ighteen-year high", allowed, words) == 1          # a number word's tail is a number
    assert belt("", allowed, words) == 0 and belt(None, allowed) == 0  # type: ignore[arg-type]


def test_a_newline_join_cannot_glue_a_spelled_word_from_two_inputs() -> None:
    assert belt("ten", set(), "Bit\nen route") == 1
    assert belt("ten", set(), "Biten route") == 0                     # glued, the join would lie


def test_the_belt_is_loose_counted_so_a_verifier_can_name_what_it_refused() -> None:
    allowed = allowed_numbers(["Into the close, shock's elevated at 58; vol jump 1.8x; sizing 0.56x"])
    tl = "Into the close, shock's elevated at 58; vol jump 1.8x; sizing 0.56x"
    assert loose("Shock's up to 58, vol popping 1.8x, sizing cut to 0.56x.", allowed, tl) == []
    assert loose("Shock's ripping to 62 into the close.", allowed, tl) == ["62"]
    assert loose("Shock's at fifty-eight.", allowed, tl) == ["fifty", "eight"]   # spelled is a number too
    assert loose("3 percent", {"3"}, "up 3%") == [] and belt("3 percent", {"3"}, "up 3 points") == 1
    md = "rally [example.com](https://example.com/2026/10/07/x) to 741"
    assert loose(md, {"740"}, strip_links=True) == ["741"] and belt(md, {"740"}) == 4


def test_strip_links_drops_link_targets_only() -> None:
    md = "rally [example.com](https://example.com/2026/10/07/x) to 740"
    assert belt(md, {"740"}, strip_links=True) == 0
    assert belt(md, {"740"}) == 3                                     # 2026, 10, 07 counted unstripped
    assert belt("[740 rally](https://x.test/1)", set(), strip_links=True) == 0   # the label goes with it
