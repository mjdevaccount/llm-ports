"""slots: the model cites a value by name, code prints it, and the belt is 0 by construction."""

from __future__ import annotations

import re

import pytest

from llm_ports import slots

VALUES = {"entry_px": "2.20", "exit_px": "2.35", "hold_min": "14", "side": "call"}


def test_fill_prints_each_value_and_leaves_an_unknown_slot_as_written() -> None:
    assert slots.fill("in at {entry_px}, out at {exit_px}", VALUES) == "in at 2.20, out at 2.35"
    assert slots.fill("held {hold_min} min {nope}", VALUES) == "held 14 min {nope}"
    assert slots.fill("", VALUES) == "" and slots.fill(None, VALUES) == ""  # type: ignore[arg-type]


def test_a_filled_text_is_clean_on_the_belt_and_an_invented_figure_is_not() -> None:
    assert slots.belt(slots.fill("in at {entry_px}, held {hold_min} min", VALUES), VALUES) == 0
    assert slots.belt("in at 2.20, out near 2.40", VALUES) == 1        # 2.40 no value renders
    assert slots.belt("in at 2.2", VALUES) == 0                          # the same value as 2.20
    assert slots.belt("held twenty minutes", VALUES) == 1                # spelled is a number too


def test_loose_names_the_figure_so_a_retry_can_say_which() -> None:
    assert slots.loose("in at {entry_px}, held twenty minutes near 2.40", VALUES) == ["2.4", "twenty"]
    assert slots.loose("in at {entry_px}, out at {exit_px}", VALUES) == []
    assert slots.loose("fourteen minutes", {**VALUES, "note": "fourteen minutes in"}) == []   # any value licenses it


def test_problems_name_an_unknown_slot_a_stray_brace_and_a_value_cited_twice() -> None:
    assert slots.problems("in at {entry_px}", VALUES) == []
    assert slots.problems("in at {strike}", VALUES) == ["unknown slot {strike}"]
    assert slots.problems("in at {entry_px", VALUES) == ["a stray brace; write a slot as {id} or not at all"]
    assert slots.problems("{entry_px} then {entry_px}", VALUES) == [
        "cites {entry_px} twice; cite each value once, next to its name"]
    assert slots.problems("{entry_px} then {entry_px}", VALUES, once=False) == []


@pytest.mark.parametrize("ids", [(), ("entry_px",), ("entry_px", "exit_px"), ("a", "b", "c")])
def test_a_slot_pattern_admits_its_slots_and_never_a_digit(ids: tuple[str, ...]) -> None:
    pat = re.compile(slots.slot_pattern(ids, 80))
    text = "the trade went " + " and ".join("{" + i + "}" for i in ids)
    assert pat.fullmatch(text)
    assert not pat.fullmatch(text + " at 2.20")
    assert not pat.fullmatch("the trade went {other}")


def test_one_or_two_slots_are_each_cited_at_most_once_by_the_grammar() -> None:
    assert not re.fullmatch(slots.slot_pattern(["a"], 80), "{a} then {a}")
    assert not re.fullmatch(slots.slot_pattern(["a", "b"], 80), "{a} {b} {a}")


def test_a_digit_in_a_slot_id_is_refused_because_the_grammar_could_not_write_it() -> None:
    with pytest.raises(ValueError, match="slot ids"):
        slots.slot_pattern(["r2"], 80)


def test_prose_pattern_carries_its_cap_and_admits_no_digit() -> None:
    assert re.fullmatch(slots.prose_pattern(5), "abcde")
    assert not re.fullmatch(slots.prose_pattern(5), "abcdef")
    assert not re.fullmatch(slots.PROSE_PATTERN, "up 3")


def test_to_prose_removes_every_digit_run_whole_and_every_non_prose_character() -> None:
    # the shapes that fed the 2026-10-30 loops: an entry note, a JSON key, a `k=v`, a percent
    assert slots.to_prose("p_reverse=0.81 (trusted, at level +0.13%)") == "p reverse (trusted, at level )"
    assert slots.to_prose("exit 14:00 at 3.1% on 5,000 lots") == "exit at on lots"   # no half a number
    assert slots.to_prose('{"stop_level": 277.89, "action": "hold"}') == "stop level : , action : hold"
    assert slots.to_prose("twenty lots near the wall") == "twenty lots near the wall"   # spelled: the belt's
    assert slots.to_prose("") == "" and slots.to_prose(None) == ""  # type: ignore[arg-type]
    assert slots.to_prose("  two\tlines\nhere ") == "two lines here"


def test_to_prose_admits_every_prose_character_and_its_output_always_matches_the_grammar() -> None:
    import string
    admitted = "".join(c for c in string.printable if re.fullmatch(slots.PROSE_CHARS, c))
    assert "a" in admitted and "'" in admitted and "-" in admitted and " " in admitted
    assert slots.to_prose(admitted) == admitted.strip()          # nothing admitted is touched
    refused = [c for c in string.printable if not re.fullmatch(slots.PROSE_CHARS, c)]
    assert "0" in refused and "_" in refused and "=" in refused and "{" in refused
    for c in refused:
        assert slots.to_prose(f"a{c}b") in ("a b", "ab"), c       # a space, or gone with its digits
    for s in ("a1b", "x{y}", "k=v", "a_b", "50%", "#tag", "say \"hi\"", "14:00 ET"):
        assert re.fullmatch(slots.PROSE_PATTERN, slots.to_prose(s)), s
