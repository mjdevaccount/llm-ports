"""call_graders: each grader passes and fails on a concrete value, a crashed or unknown grader is a
FAILED score (never an exception), and the board shape names an outage."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from llm_ports.call import graders as cg
from tests.conftest import Role

GRADERS = ("schema_complete", "numbers_from_inputs", "echo")
ROLE = Role("test.graders", tier="local", max_tokens=5, graders=GRADERS)
PROMPT = "spot 766.2 at the put_wall 765.0; the market rallied hard into the close on strong breadth today."
COPIED = "The market rallied hard into the close on strong breadth today."


class _Note(BaseModel):
    summary: str
    keys: list[str]


class _Level(BaseModel):
    """A deep_extract.mentions row: the prompt orders null unless the advisor STATED the figure."""

    ticker: str
    entry_low: float | None = Field(default=None)
    note: str | None = None


class _Call(BaseModel):
    thesis: str
    stop: Optional[float] = None  # noqa: UP045 -- the old spelling must read as nullable too
    tickers: list[_Level]
    hedge: _Level | None = None


class TestLeaves:
    def test_dict_pydantic_and_list_flatten_to_paths(self) -> None:
        assert list(cg.leaves({"a": 1, "b": {"c": "x"}})) == [("a", 1), ("b.c", "x")]
        assert list(cg.leaves(_Note(summary="s", keys=["k1", "k2"]))) == [
            ("summary", "s"),
            ("keys[0]", "k1"),
            ("keys[1]", "k2"),
        ]
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

    def test_schema_complete_reads_a_null_on_a_nullable_field_as_an_answer(self) -> None:
        # The alert this fixes: deep_extract.mentions / post.axes declared `entry_low: float | None`
        # and ORDERED null when the advisor never said the number -- an honest reply failed by law.
        ok = cg.schema_complete(PROMPT, _Level(ticker="SPY"), ROLE)
        assert (ok.passed, ok.value, ok.reason) == (True, 1.0, ""), ok.reason
        nested = cg.schema_complete(
            PROMPT, _Call(thesis="t", tickers=[_Level(ticker="SPY", entry_low=1.0), _Level(ticker="QQQ")]), ROLE
        )
        assert nested.passed and nested.value == 1.0, nested.reason
        assert cg.nullable_paths(_Call) == {
            "stop",
            "tickers[].entry_low",
            "tickers[].note",
            "hedge",
            "hedge.entry_low",
            "hedge.note",
        }

    def test_schema_complete_still_fails_a_null_on_a_required_field(self) -> None:
        # The annotation is the law: a REQUIRED field left None (model_construct skips validation,
        # as a grammar-parsed reply with a missing key would) is still unanswered.
        bad = cg.schema_complete(PROMPT, _Level.model_construct(ticker=None, entry_low=None), ROLE)
        assert not bad.passed and bad.reason == "empty: ['ticker']"
        row = _Call.model_construct(thesis="t", stop=None, tickers=[_Level.model_construct(ticker=None)], hedge=None)
        deep = cg.schema_complete(PROMPT, row, ROLE)
        assert not deep.passed and deep.reason == "empty: ['tickers[0].ticker']", deep.reason
        # A dict carries no annotations, so a None in one is graded exactly as before.
        plain = cg.schema_complete(PROMPT, {"ticker": "SPY", "entry_low": None}, ROLE)
        assert not plain.passed and plain.value == 0.5 and plain.reason == "empty: ['entry_low']"

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
        assert set(GRADERS) | {"cites_url", "no_refusal"} == set(cg.GRADERS)
        assert all(cg.GRADERS[n] is getattr(cg, n) for n in cg.GRADERS)


class TestWebGraders:
    """The research-tier checks: a web reply cites its pages, no reply refuses on stock phrases."""

    def test_cites_url_counts_distinct_urls_in_any_string_field(self) -> None:
        v = {"summary": "Up 2% (https://a.example/x).", "sources": ["https://a.example/x", "http://b.example/y."]}
        s = cg.cites_url("", v, ROLE)
        assert (s.value, s.passed, s.reason) == (2.0, True, "")

    def test_cites_url_fails_a_reply_with_no_url(self) -> None:
        """MUST FIRE: a reply with a bare domain and no scheme cites nothing checkable."""
        s = cg.cites_url("", {"summary": "per reuters.com the print was hot"}, ROLE)
        assert (s.value, s.passed, s.reason) == (0.0, False, "the reply cites no URL")

    def test_no_refusal_fires_on_stock_refusals(self) -> None:
        for text in (
            "I don't have access to real-time data.",
            "I cannot browse the web, sorry.",
            "As of my last training update the price was 100.",
            "Please consult a financial advisor.",
        ):
            s = cg.no_refusal("", {"summary": text}, ROLE)
            assert not s.passed and s.value >= 1 and s.reason.startswith("refusal:"), text

    def test_no_refusal_passes_an_honest_gap(self) -> None:
        """MUST NOT FIRE: saying the sources are silent is grounded, not a refusal."""
        s = cg.no_refusal("", {"summary": "The sources do not cover the guidance cut; access fees rose 3%."}, ROLE)
        assert (s.value, s.passed) == (0.0, True)


class TestGradersForInputs:
    """Moved from trade-data's platform/call_graders (v0.9.0): the text-only exemption."""

    STRUCTURED = ("schema_complete", "numbers_from_inputs", "echo")

    def test_the_text_only_set_is_the_number_law(self) -> None:
        assert {"numbers_from_inputs"} == cg.TEXT_ONLY_GRADERS

    def test_an_image_or_a_web_tier_drops_it_and_keeps_order(self) -> None:
        """MUST FIRE: a filter that ignored `has_web` (or `has_images`) fails one of these."""
        assert cg.graders_for_inputs(self.STRUCTURED, has_images=True) == ("schema_complete", "echo")
        assert cg.graders_for_inputs(self.STRUCTURED, has_images=False, has_web=True) == ("schema_complete", "echo")
        assert cg.graders_for_inputs(("echo", "numbers_from_inputs"), has_images=True, has_web=True) == ("echo",)

    def test_a_text_call_keeps_every_grader(self) -> None:
        assert cg.graders_for_inputs(self.STRUCTURED, has_images=False) == self.STRUCTURED
        assert cg.graders_for_inputs((), has_images=True) == ()

    def test_the_web_only_set_is_the_citation_check(self) -> None:
        assert {"cites_url"} == cg.WEB_ONLY_GRADERS

    def test_a_call_without_web_drops_cites_url_and_a_web_call_keeps_it(self) -> None:
        """MUST FIRE both ways: a text call never fails for citing no pages; a web call is held to it."""
        names = ("schema_complete", "cites_url", "no_refusal", "numbers_from_inputs")
        assert cg.graders_for_inputs(names, has_images=False) == (
            "schema_complete",
            "no_refusal",
            "numbers_from_inputs",
        )
        assert cg.graders_for_inputs(names, has_images=True) == ("schema_complete", "no_refusal")
        assert cg.graders_for_inputs(names, has_images=False, has_web=True) == (
            "schema_complete",
            "cites_url",
            "no_refusal",
        )
