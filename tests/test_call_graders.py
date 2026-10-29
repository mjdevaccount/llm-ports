"""call_graders: each grader passes and fails on a concrete value, a crashed or unknown grader is a
FAILED score (never an exception), and the board shape names an outage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from llm_ports import call_graders as cg
from tests.conftest import Role

GRADERS = ("schema_complete", "numbers_from_inputs", "echo")
ROLE = Role("test.graders", tier="box", max_tokens=5, graders=GRADERS)
PROMPT = "spot 766.2 at the put_wall 765.0; the market rallied hard into the close on strong breadth today."
COPIED = "The market rallied hard into the close on strong breadth today."


class _Note(BaseModel):
    summary: str
    keys: list[str]


class TestLeaves:
    def test_dict_pydantic_and_list_flatten_to_paths(self) -> None:
        assert list(cg.leaves({"a": 1, "b": {"c": "x"}})) == [("a", 1), ("b.c", "x")]
        assert list(cg.leaves(_Note(summary="s", keys=["k1", "k2"]))) == [("summary", "s"), ("keys[0]", "k1"),
                                                                           ("keys[1]", "k2")]
        assert list(cg.leaves(["p", "q"])) == [("[0]", "p"), ("[1]", "q")]

    def test_an_empty_list_is_itself_a_leaf(self) -> None:
        assert list(cg.leaves({"keys": []})) == [("keys", [])]


class TestEachGrader:
    def test_schema_complete(self) -> None:
        ok = cg.schema_complete(PROMPT, {"a": "x", "b": ["y"]}, ROLE)
        assert (ok.passed, ok.value, ok.reason) == (True, 1.0, "")
        bad = cg.schema_complete(PROMPT, {"a": "x", "b": None, "c": "  "}, ROLE)
        assert not bad.passed and bad.value == 0.3333 and bad.reason == "empty: ['b', 'c']"
        assert cg.schema_complete(PROMPT, {"a": "x", "cited": []}, ROLE).passed, "an empty list is an answer"
        none = cg.schema_complete(PROMPT, {}, ROLE)
        assert not none.passed and none.value == 0.0

    def test_numbers_from_inputs(self) -> None:
        ok = cg.numbers_from_inputs(PROMPT, {"why": "spot sat at 766.2 above 765.0"}, ROLE)
        assert ok.passed and ok.value == 0.0
        bad = cg.numbers_from_inputs(PROMPT, {"why": "spot sat at 999 after a 42 point drop"}, ROLE)
        assert not bad.passed and bad.value == 2.0 and "999" in bad.reason

    def test_numbers_from_inputs_reads_a_spelled_number_as_the_writers_belts_do(self) -> None:
        # A digit-free grammar made the model SPELL it ("nine one point seven million"): no input
        # rendered that figure, and the per-call grader must agree with slots.belt about it.
        spelled = cg.numbers_from_inputs(PROMPT, {"why": "spot sat seven points above the wall"}, ROLE)
        assert not spelled.passed and spelled.value == 1.0 and "seven" in spelled.reason
        # The prompt's own words license a word; a '%' in the prompt licenses 'percent'.
        ok = cg.numbers_from_inputs("nine voters, up 3% (three points)", {"why": "nine voted for three percent"}, ROLE)
        assert ok.passed, ok.reason
        assert cg.numbers_from_inputs(PROMPT, {"why": "peak 766.20"}, ROLE).passed, "766.20 is 766.2 by value"

    def test_echo(self) -> None:
        own = cg.echo(PROMPT, {"a": "Breadth was the tell and the dealer book was short gamma all session."}, ROLE)
        assert own.passed and own.value == 0.0
        bad = cg.echo(PROMPT, {"a": COPIED, "b": COPIED}, ROLE)
        assert not bad.passed and bad.value == 1.0 and "2/2 sentences copied" in bad.reason
        assert cg.echo(PROMPT, {"a": "short"}, ROLE).passed, "a label is not a sentence"


class TestNeverRaises:
    def test_an_unknown_name_is_a_failed_score(self) -> None:
        s = cg.score_one("ghost", PROMPT, {"a": "x"}, ROLE)
        assert (s.name, s.value, s.passed) == ("ghost", cg.CRASHED, False) and "no grader named" in s.reason

    def test_a_raising_grader_is_a_failed_score(self) -> None:
        def boom(_p: str, _v: Any, _r: Any) -> cg.CallScore:
            raise ZeroDivisionError("bad math")

        scores = cg.score_all(ROLE, PROMPT, {"a": "x"}, {**cg.GRADERS, "schema_complete": boom})
        assert [s.name for s in scores] == list(GRADERS)
        crashed = scores[0]
        assert crashed.value == cg.CRASHED and not crashed.passed and "ZeroDivisionError: bad math" in crashed.reason
        assert all(s.passed for s in scores[1:]), "the crash is one grader's, not the call's"


class TestBoardShape:
    def test_a_row_without_a_reply_fails_every_grader_naming_the_outage(self) -> None:
        board = cg.board_graders(ROLE)
        assert set(board) == set(GRADERS)
        for fn in board.values():
            passed, reason = fn({"value": None, "reason": "box refused", "outcome": "unavailable"})
            assert not passed and reason == "no reply: box refused"
        assert board["schema_complete"]({"prompt": PROMPT, "value": {"a": "x"}}) == (True, "")

    def test_the_built_in_names_are_this_modules_functions(self) -> None:
        assert set(GRADERS) == set(cg.GRADERS)
