"""judge_sampling: the pure half of a sampled LLM judge -- pinned with no model, database or clock.

The draw is deterministic on the reply's stamp and ~1-in-N over many keys; the rubric composes the
four laws from the role, its grounding mode, the call's web tier and the injected calendar; the
judge's Input names everything the judged call saw.
"""

from __future__ import annotations

import datetime as dt

from llm_ports.evals import judge as js
from llm_ports.ports import JudgeVerdict, json_output
from tests.conftest import Role

ROLE = Role("test.judged", tier="box", graders=("schema_complete", "numbers_from_inputs"))
PROSE_ROLE = Role("test.prose", tier="box")
DAY = dt.date(2026, 10, 31)


def _rubric(role: Role = ROLE, seen: js.Seen | None = None, *, mode: str = "graded") -> str:
    return js.rubric(role, seen, platform="ACME support platform", mode=mode, today=DAY, tz_label="UTC")


class TestDraw:
    """MUST-FIRE: the 1-in-N selection is deterministic on (role, prompt, reply) and ~1/mod."""

    def test_the_key_is_deterministic_on_role_prompt_and_reply(self) -> None:
        a = js.sample_key(ROLE, "p", "r")
        assert a == js.sample_key(ROLE, "p", "r") and len(a) == 16
        assert a != js.sample_key(ROLE, "p", "r2")
        assert a != js.sample_key(ROLE, "p2", "r")
        assert a != js.sample_key(PROSE_ROLE, "p", "r")

    def test_the_key_is_frozen_on_fixed_inputs(self) -> None:
        """A recorded key must stay answerable: same algorithm forever (packs.sha over name/prompt/reply)."""
        assert js.sample_key(ROLE, "p", "r") == "a046cc4f510f8bac"  # sha256("test.judged\np\nr")[:16]

    def test_selected_is_about_one_in_mod_and_stable(self) -> None:
        keys = [js.sample_key(ROLE, f"prompt {i}", f"reply {i}") for i in range(4000)]
        hits = sum(js.selected(k, 20) for k in keys)
        assert 120 <= hits <= 280, f"{hits}/4000 selected at mod 20; expected ~200"
        assert [js.selected(k, 20) for k in keys[:50]] == [js.selected(k, 20) for k in keys[:50]]

    def test_selection_reads_the_keys_first_32_bits(self) -> None:
        assert js.selected("00000014" + "f" * 8, 20) is True  # 0x14 = 20
        assert js.selected("00000015" + "0" * 8, 20) is False

    def test_mod_zero_selects_nothing_and_mod_one_everything(self) -> None:
        k = js.sample_key(ROLE, "p", "r")
        assert js.selected(k, 0) is False and js.selected(k, -3) is False
        assert js.selected(k, 1) is True


class TestRubric:
    """MUST-FIRE: the four laws compose from role, mode, web and calendar -- each switch both ways."""

    def test_the_rubric_carries_the_four_laws_the_role_and_the_platform(self) -> None:
        r = _rubric()
        assert r.startswith("You are grading one reply from the ACME support platform's `test.judged` role, ")
        assert "a structured object under a strict grammar" in r
        for law in ("SCHEMA ANSWERED HONESTLY", "NUMBERS FROM THE INPUT ONLY", "NO ECHO", "GROUNDED"):
            assert law in r
        assert "spelled out" in r and "is the model's ANSWER" in r

    def test_the_shape_reads_graders_pack_or_grammar(self) -> None:
        assert "prose or a structured object" in _rubric(PROSE_ROLE)
        assert "under a strict grammar" in _rubric(PROSE_ROLE, js.Seen(schema={"type": "object"}))

    def test_the_calendar_is_the_injected_one(self) -> None:
        r = _rubric()
        assert "Today is 2026-10-31 (UTC)." in r and "looks like the future to you" in r

    def test_law_two_reads_the_grounding_mode(self) -> None:
        """A `slots` role cites `{id}`: the law is cite-never-write, not numbers-from-the-input."""
        r = _rubric(mode="slots")
        assert "FIGURES ARE CITED, NEVER WRITTEN" in r and "{id}" in r
        assert "NUMBERS FROM THE INPUT ONLY" not in r
        assert "is the model's ANSWER" in r  # every mode: a grammar value is an answer
        assert "'Spot', 'put wall', 'stand down'" in r and "cited or not" in r
        assert "NUMBERS FROM THE INPUT ONLY" in _rubric(mode="none")

    def test_a_slots_citation_is_not_a_placeholder_but_a_written_number_still_fails(self) -> None:
        slots = _rubric(mode="slots")
        assert "A `{id}` citation is this role's mandated way to state a value, never a placeholder" in slots
        assert "A number written out in the Output, as digits or as number words, is invented and fails" in slots
        assert "never a placeholder" not in _rubric()

    def test_a_web_call_is_not_held_to_numbers_from_the_input(self) -> None:
        r = _rubric(seen=js.Seen(web=True))
        assert "WEB FIGURES ARE NOT JUDGED ABSENT" in r and "CONTRADICTS the Input" in r
        assert "NUMBERS FROM THE INPUT ONLY" not in r and "or from the web the role read" in r
        assert "NUMBERS FROM THE INPUT ONLY" in _rubric(seen=js.Seen())
        assert "WEB FIGURES" in _rubric(seen=js.Seen(web=True), mode="slots"), "web wins over slots"

    def test_the_rubric_names_every_part_of_the_input(self) -> None:
        r = _rubric(seen=js.Seen(system="sys", schema={"type": "object"}))
        assert "SYSTEM INSTRUCTIONS" in r and "OUTPUT GRAMMAR" in r and "attached image" in r
        assert "never fail it under this law" in r and "a vocabulary" in r
        assert "source quote" in r, "a field the Input asks to hold a quote is not echo"


class TestJudgeInput:
    def test_a_bare_call_is_judged_on_its_bare_prompt(self) -> None:
        assert js.judge_input("the prompt", js.Seen()) == "the prompt"

    def test_system_grammar_and_images_precede_the_prompt(self) -> None:
        seen = js.Seen(
            system="EVENT TYPES: geo_deescalation, geo_threat",
            schema={"properties": {"event_type": {"enum": ["geo_deescalation"]}}},
            images=((b"png", "image/png"),),
        )
        got = js.judge_input("Iran reopens talks", seen)
        assert got.index("SYSTEM INSTRUCTIONS") < got.index("OUTPUT GRAMMAR") < got.index("IMAGES: 1 image")
        assert got.index("IMAGES: 1 image") < got.index("## PROMPT")
        assert "geo_deescalation, geo_threat" in got and '"enum": ["geo_deescalation"]' in got
        assert got.endswith("Iran reopens talks")

    def test_a_long_system_prompt_is_capped(self) -> None:
        got = js.judge_input("p", js.Seen(system="x" * (js.SYSTEM_CAP + 500)))
        assert got.count("x") == js.SYSTEM_CAP

    def test_schema_of_each_output_shape(self) -> None:
        from pydantic import BaseModel

        class M(BaseModel):
            conviction: int

        assert js.schema_of(json_output({"type": "object", "title": "t"}, "t")) == {"type": "object", "title": "t"}
        got = js.schema_of(M)
        assert got is not None and "conviction" in got["properties"]
        assert js.schema_of(str) is None and js.schema_of(None) is None

    def test_seen_for_carries_everything(self) -> None:
        s = js.seen_for(system="s", output=str, images=[(b"i", "image/png")], web=True)
        assert s == js.Seen(system="s", schema=None, images=((b"i", "image/png"),), web=True)

    def test_reply_text_of_model_mapping_and_prose(self) -> None:
        from pydantic import BaseModel

        class M(BaseModel):
            a: int

        assert js.reply_text(M(a=1)) == '{"a":1}'
        assert js.reply_text({"a": 1}) == '{"a": 1}'
        assert js.reply_text("prose") == "prose" and js.reply_text(None) == ""


def test_verdict_labels() -> None:
    assert js.verdict_label(None) == "could_not_run"
    assert js.verdict_label(JudgeVerdict(passed=True, score=1.0)) == "pass"
    assert js.verdict_label(JudgeVerdict(passed=False, score=0.0)) == "fail"
