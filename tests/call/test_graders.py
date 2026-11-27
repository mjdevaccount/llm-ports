"""call_graders: each grader passes and fails on a concrete value, a crashed or unknown grader is a
FAILED score (never an exception), and the board shape names an outage."""

from __future__ import annotations

from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from llm_ports.call import graders as cg
from tests.conftest import Role

GRADERS = ("schema_complete", "numbers_from_inputs", "echo")
ROLE = Role("test.graders", tier="local", max_tokens=5, graders=GRADERS)
PROMPT = "level 766.2 at the spillway 765.0; the river rose hard into the evening on heavy rain today."
COPIED = "The river rose hard into the evening on heavy rain today."


class _Note(BaseModel):
    summary: str
    keys: list[str]


class _Line(BaseModel):
    """An extracted row: the prompt orders null unless the source STATED the figure."""

    sku: str
    unit_low: float | None = Field(default=None)
    note: str | None = None


class _Order(BaseModel):
    memo: str
    cap: Optional[float] = None  # noqa: UP045 -- the old spelling must read as nullable too
    lines: list[_Line]
    backup: _Line | None = None


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
        # The alert this fixes: two extraction roles declared `unit_low: float | None`
        # and ORDERED null when the source never said the number -- an honest reply failed by law.
        ok = cg.schema_complete(PROMPT, _Line(sku="A-1"), ROLE)
        assert (ok.passed, ok.value, ok.reason) == (True, 1.0, ""), ok.reason
        nested = cg.schema_complete(
            PROMPT, _Order(memo="t", lines=[_Line(sku="A-1", unit_low=1.0), _Line(sku="B-2")]), ROLE
        )
        assert nested.passed and nested.value == 1.0, nested.reason
        assert cg.nullable_paths(_Order) == {
            "cap",
            "lines[].unit_low",
            "lines[].note",
            "backup",
            "backup.unit_low",
            "backup.note",
        }

    def test_schema_complete_still_fails_a_null_on_a_required_field(self) -> None:
        # The annotation is the law: a REQUIRED field left None (model_construct skips validation,
        # as a grammar-parsed reply with a missing key would) is still unanswered.
        bad = cg.schema_complete(PROMPT, _Line.model_construct(sku=None, unit_low=None), ROLE)
        assert not bad.passed and bad.reason == "empty: ['sku']"
        row = _Order.model_construct(memo="t", cap=None, lines=[_Line.model_construct(sku=None)], backup=None)
        deep = cg.schema_complete(PROMPT, row, ROLE)
        assert not deep.passed and deep.reason == "empty: ['lines[0].sku']", deep.reason
        # A dict carries no annotations, so a None in one is graded exactly as before.
        plain = cg.schema_complete(PROMPT, {"sku": "A-1", "unit_low": None}, ROLE)
        assert not plain.passed and plain.value == 0.5 and plain.reason == "empty: ['unit_low']"

    def test_schema_complete_reads_a_dict_against_a_declared_model(self) -> None:
        # A strict grammar replies a DICT; the declared model supplies the answer paths it cannot carry.
        reply = {
            "memo": "t",
            "cap": None,
            "lines": [{"sku": "A-1", "unit_low": None, "note": None}],
            "backup": None,
        }
        assert not cg.schema_complete(PROMPT, reply, ROLE).passed, "undeclared: today's strict read"
        ok = cg.schema_complete(PROMPT, reply, ROLE, model=_Order)
        assert (ok.passed, ok.value, ok.reason) == (True, 1.0, ""), ok.reason

    def test_a_declared_model_still_fails_a_null_on_a_required_field_of_a_dict(self) -> None:
        reply = {"memo": "", "cap": None, "lines": [{"sku": None, "unit_low": None, "note": None}]}
        bad = cg.schema_complete(PROMPT, reply, ROLE, model=_Order)
        assert not bad.passed and bad.reason == "empty: ['memo', 'lines[0].sku']", bad.reason
        assert bad.value == 0.6

    def test_named_paths_exempt_only_themselves(self) -> None:
        # A field typed nullable whose prompt never asks for null: the caller names the subset.
        reply = {"memo": "t", "cap": None, "lines": [{"sku": "A-1", "unit_low": None, "note": None}]}
        some = cg.schema_complete(PROMPT, reply, ROLE, nullable={"lines[].unit_low", "lines[].note"})
        assert not some.passed and some.reason == "empty: ['cap']", some.reason
        assert cg.schema_complete(PROMPT, reply, ROLE, nullable={"cap", "lines[].unit_low", "lines[].note"}).passed
        both = cg.schema_complete(PROMPT, {"x": None, "cap": None}, ROLE, model=_Order, nullable={"x"})
        assert both.passed, "model and named paths join"

    def test_a_declaration_replaces_a_pydantic_values_own_annotations(self) -> None:
        # Declared = the caller's whole answer set, on any value: an empty declaration is strict.
        value = _Line(sku="A-1")
        strict = cg.schema_complete(PROMPT, value, ROLE, nullable=())
        assert not strict.passed and strict.reason == "empty: ['unit_low', 'note']", strict.reason

    def test_a_bound_declaration_is_a_table_entry(self) -> None:
        from functools import partial

        table = {**cg.GRADERS, "schema_complete": partial(cg.schema_complete, model=_Order)}
        s = cg.score_one("schema_complete", PROMPT, {"memo": "t", "cap": None, "lines": []}, ROLE, table)
        assert s.passed, s.reason
        assert cg.GRADERS["schema_complete"] is cg.schema_complete, "the built-in table is untouched"

    def test_numbers_from_inputs(self) -> None:
        ok = cg.numbers_from_inputs(PROMPT, {"why": "level sat at 766.2 above 765.0"}, ROLE)
        assert ok.passed and ok.value == 0.0
        bad = cg.numbers_from_inputs(PROMPT, {"why": "level sat at 999 after a 42 point drop"}, ROLE)
        assert not bad.passed and bad.value == 2.0 and "999" in bad.reason

    def test_numbers_from_inputs_licenses_one_input_value_shown_coarser_and_nothing_computed(self) -> None:
        prompt = "ratio 0.5918 | headroom 14.878 | net_flow -19813893517 | win_rate 0.7486 | gauge 28.89"
        rounded = {"why": "Measured 0.59R with 14.88 of room, -19.8B of flow and a 75% win rate."}
        assert cg.numbers_from_inputs(prompt, rounded, ROLE).passed
        invented = cg.numbers_from_inputs(prompt, {"why": "Wait for a 15-minute retest below 28.00."}, ROLE)
        assert not invented.passed and invented.value == 2.0 and "['15', '28']" in invented.reason

    @pytest.mark.parametrize(
        ("prompt", "reply", "named"),
        [
            # A level moved to a round figure is not a rounding of the input: 28.89 is 28.89 at two decimals.
            ("gauge 28.89 | level 766.2", "Hold off until the gauge reads below 28.00.", "28"),
            # A dropped digit is not a percent of the input: -26 is -26%, and -0.026 would be -2.6%.
            ("pct_from_52w_high -26 | level 766.2", "It sits -2.6% below its 52-week high.", "2.6"),
            # A bare whole number is never a rounding, and no input holds a 15.
            ("ratio 0.5918 | minutes_from_anchor -48.1", "Wait for a 15-minute recheck.", "15"),
        ],
    )
    def test_numbers_from_inputs_must_fail_fixtures(self, prompt: str, reply: str, named: str) -> None:
        # The three sentences the licence must never pass, shaped on live replies (2026-11-25).
        bad = cg.numbers_from_inputs(prompt, {"why": reply}, ROLE)
        assert not bad.passed and bad.value == 1.0, bad.reason
        assert f"'{named}'" in bad.reason, bad.reason

    def test_numbers_from_inputs_licenses_a_marked_whole_number_and_never_a_bare_one(self) -> None:
        # Rounding ONE input to fewer digits includes rounding it to none, but only where a percent
        # sign or a scale says which value is meant. Bare, "716" for 715.87 is an invented level.
        prompt = "net_flow -19813893517 | win_rate 0.7486 | chg_pct 13.49 | floor_mark 715.87"
        marked = {"why": "Net flow is -20B, the win rate 75% and the move 13 percent."}
        assert cg.numbers_from_inputs(prompt, marked, ROLE).passed
        bare = cg.numbers_from_inputs(prompt, {"why": "The floor sits at 716 and the move was 13 wide."}, ROLE)
        assert not bare.passed and bare.value == 2.0 and "['13', '716']" in bare.reason, bare.reason

    def test_numbers_from_inputs_reads_a_spelled_number_as_the_writers_belts_do(self) -> None:
        # A digit-free grammar made the model SPELL it ("nine one point seven million"): no input
        # rendered that figure, and the per-call grader must agree with slots.belt about it.
        spelled = cg.numbers_from_inputs(PROMPT, {"why": "level sat seven points above the spillway"}, ROLE)
        assert not spelled.passed and spelled.value == 1.0 and "seven" in spelled.reason
        # The prompt's own words license a word; a '%' in the prompt licenses 'percent'.
        ok = cg.numbers_from_inputs("nine voters, up 3% (three points)", {"why": "nine voted for three percent"}, ROLE)
        assert ok.passed, ok.reason
        assert cg.numbers_from_inputs(PROMPT, {"why": "peak 766.20"}, ROLE).passed, "766.20 is 766.2 by value"

    def test_echo(self) -> None:
        own = cg.echo(PROMPT, {"a": "Rain was the tell and the upstream gauge read high all evening."}, ROLE)
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

    def test_the_table_is_read_only_and_extends_by_copying(self) -> None:
        """Packs-and-evals acceptance 10: assigning into GRADERS raises; a copy extends it."""

        def mine(_p: str, _v: Any, _r: Any) -> cg.CallScore:
            return cg.CallScore("mine", 1.0, True, "")

        with pytest.raises(TypeError):
            cg.GRADERS["mine"] = mine  # type: ignore[index]  # the assignment under test is refused
        extended = {**cg.GRADERS, "mine": mine}
        assert extended["mine"] is mine and "mine" not in cg.GRADERS and set(extended) - set(cg.GRADERS) == {"mine"}
        assert cg.score_one("mine", "", None, ROLE, extended).passed


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

    def test_no_refusal_fires_on_the_punts_the_old_phrasebook_missed(self) -> None:
        """MUST FIRE: each was a measured punt that the narrower phrasebook let through."""
        for text in (
            "I don't have enough information to answer that.",
            "I do not have current figures for this.",
            "As of my last update, I cannot say.",
            "I was unable to find the release.",
            "I am unable to determine the cause.",
            "I cannot determine which version shipped.",
            "I am not able to provide a figure.",
            "The search did not return anything useful.",
            "No relevant results were found.",
        ):
            s = cg.no_refusal("", {"summary": text}, ROLE)
            assert not s.passed and s.value >= 1, text

    def test_no_refusal_passes_honest_hedges(self) -> None:
        """MUST NOT FIRE: naming a gap or an unconfirmed figure is grounded prose, not a refusal."""
        for text in (
            "I cannot confirm the second figure; the two sources disagree.",
            "No data was available for the third quarter, so it is left out.",
            "The page does not state the date.",
        ):
            s = cg.no_refusal("", {"summary": text}, ROLE)
            assert (s.value, s.passed) == (0.0, True), text

    def test_a_grounded_reply_is_not_scored_by_its_hedges(self) -> None:
        """A URL plus three distinct figures is an answer; its "please check" is a hedge."""
        text = "Revenue was 12.4, margin 31% and units 870 (https://example.com/q3). Please check the source."
        assert cg.ungrounded_refusals(text) == []
        s = cg.no_refusal("", {"summary": text}, ROLE)
        assert (s.value, s.passed) == (0.0, True)

    def test_grounding_is_judged_over_the_whole_reply(self) -> None:
        """The URL in one field and the figures in another still ground the reply."""
        value = {"source": "https://example.com/r", "summary": "Rose 4%, to 210, from 198. Please check."}
        assert cg.no_refusal("", value, ROLE).passed

    def test_figures_inside_the_url_do_not_ground_it(self) -> None:
        """MUST FIRE: the digits of a URL are not figures the reply states."""
        text = "Please check https://example.com/2026/11/27/item-4410 for details."
        assert cg.ungrounded_refusals(text) == ["Please check"]

    def test_is_grounded_needs_a_url_and_three_figures_outside_it(self) -> None:
        assert cg.is_grounded("Up 4% to 210 from 198, per https://example.com/r")
        assert not cg.is_grounded("Up 4% to 210 from 198.")  # no URL
        assert not cg.is_grounded("See https://example.com/2026/11/27/4410 for 2 figures.")
        assert not cg.is_grounded("")

    def test_a_url_with_too_few_figures_is_still_a_refusal(self) -> None:
        text = "I am unable to provide the total; see https://example.com/r for 2 figures."
        assert cg.ungrounded_refusals(text) == ["unable to provide"]


class TestGradersForInputs:
    """Moved from the first consumer's call graders (v0.9.0): the text-only exemption."""

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


class TestNullish:
    @pytest.mark.parametrize("v", ["N/A", " none ", "TBD", "-", "", "Not Stated", "unknown", None])
    def test_an_abstention_spelling_is_nullish(self, v: Any) -> None:
        assert cg.is_nullish(v) and cg.nullish_to_none(v) is None

    @pytest.mark.parametrize("v", ["nonetheless", "n/a/b", "12", 0, 0.0, False, [], "unknown origin"])
    def test_a_value_passes_through_unchanged(self, v: Any) -> None:
        assert not cg.is_nullish(v) and cg.nullish_to_none(v) is v

    def test_a_pydantic_before_validator_turns_the_abstention_into_none(self) -> None:
        from pydantic import field_validator

        class Out(BaseModel):
            price: float | None

            @field_validator("price", mode="before")
            @classmethod
            def _abstain(cls, v: Any) -> Any:
                return cg.nullish_to_none(v)

        assert Out.model_validate({"price": "N/A"}).price is None
        assert Out.model_validate({"price": "4.5"}).price == 4.5
        with pytest.raises(ValueError):
            Out.model_validate({"price": "four"})  # malformed is still seen, never nulled

    def test_schema_complete_scores_nullish_as_unanswered_when_asked(self) -> None:
        reply = {"a": "x", "b": "N/A", "c": "none"}
        assert cg.schema_complete(PROMPT, reply, ROLE).passed, "the default scores exactly as before"
        s = cg.schema_complete(PROMPT, reply, ROLE, nullish=True)
        assert not s.passed and s.value == 0.3333 and s.reason == "empty: ['b', 'c']"

    def test_a_declared_path_exempts_a_nullish_choice(self) -> None:
        s = cg.schema_complete(PROMPT, {"a": "x", "category": "none"}, ROLE, nullish=True, nullable={"category"})
        assert s.passed
