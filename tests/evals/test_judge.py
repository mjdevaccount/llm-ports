"""judge_sampling: the pure half of a sampled LLM judge -- pinned with no model, database or clock.

The draw is deterministic on the reply's stamp and ~1-in-N over many keys; the rubric composes the
four laws from the role, its grounding mode, the call's web tier and the injected calendar; the
judge's Input names everything the judged call saw.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import logging
from collections.abc import Mapping
from typing import Any

import pytest

from llm_ports.evals import judge as js
from llm_ports.grounding import numbers
from llm_ports.ports import Generated, JudgeVerdict, json_output
from tests.conftest import Role

ROLE = Role("test.judged", tier="local", graders=("schema_complete", "numbers_from_inputs"))
PROSE_ROLE = Role("test.prose", tier="local")
DAY = dt.date(2026, 10, 31)
_UNIT: contextvars.ContextVar[str | None] = contextvars.ContextVar("_UNIT", default=None)


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
        assert "a word naming a fact is never a reason to fail" in r and "cited or not" in r
        named = js.rubric(
            ROLE,
            platform="ACME support platform",
            mode="slots",
            today=DAY,
            tz_label="UTC",
            examples=numbers.LawExamples(paraphrase=("'pending'",), fact_names=("'Order total'", "'Ship date'")),
        )
        assert "paraphrase ('pending') are not figures" in named
        assert "a word naming a fact ('Order total', 'Ship date') is never" in named
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


class TestJudgeLane:
    """The lane schedules a drawn reply off the await path, holds the task, and never raises."""

    @staticmethod
    def _ok(value: object = "a reply") -> Generated[object]:
        return Generated(value=value, outcome="ok", model="m")

    @staticmethod
    def _lane() -> tuple[js.JudgeLane, list[tuple[int, str, Mapping[str, Any]]]]:
        events: list[tuple[int, str, Mapping[str, Any]]] = []
        return js.JudgeLane(on_event=lambda lv, ev, f: events.append((lv, ev, dict(f)))), events

    async def test_a_drawn_ok_reply_schedules_one_task_with_its_sample(self) -> None:
        lane, events = self._lane()
        got: list[js.Sample] = []

        async def run(s: js.Sample) -> None:
            got.append(s)

        seen = js.Seen(system="rules")
        res = self._ok({"a": 1})
        key = lane.start(ROLE, "p", res, mod=1, run=run, seen=seen)
        assert key == js.sample_key(ROLE, "p", '{"a": 1}') and lane.pending() == 1
        await lane.drain()
        assert lane.pending() == 0
        assert got == [js.Sample(ROLE, "p", '{"a": 1}', key, seen, res)]
        assert events == [(logging.INFO, "llm_judge_sampled", {"role": ROLE.name, "sample_key": key, "mod": 1})]

    async def test_nothing_is_scheduled_off_ok_off_switch_ineligible_or_undrawn(self) -> None:
        lane, _ = self._lane()
        calls: list[tuple[str, int]] = []

        async def run(_s: js.Sample) -> None:
            raise AssertionError("never scheduled")

        def eligible(role: Role, mod: int) -> bool:
            calls.append((role.name, mod))
            return False

        failed: Generated[object] = Generated(value=None, outcome="truncated", model="m")
        assert lane.start(ROLE, "p", failed, mod=1, run=run) is None
        assert lane.start(ROLE, "p", self._ok(), mod=0, run=run) is None
        assert lane.start(ROLE, "p", self._ok(), mod=lambda: 7, run=run, eligible=eligible) is None
        assert calls == [(ROLE.name, 7)], "a callable mod is resolved before the gate reads it"
        key = js.sample_key(ROLE, "p", "a reply")
        undrawn = int(key[:8], 16) + 1  # the remainder is the key itself, never 0
        assert lane.start(ROLE, "p", self._ok(), mod=undrawn, run=run) is None
        assert lane.pending() == 0

    def test_no_running_loop_is_a_quiet_no(self) -> None:
        lane, events = self._lane()

        async def run(_s: js.Sample) -> None: ...

        assert lane.start(ROLE, "p", self._ok(), mod=1, run=run) is None and events == []

    async def test_a_seen_thunk_runs_only_for_a_draw_and_a_raising_one_is_judged_blind_and_said(self) -> None:
        lane, events = self._lane()
        got: list[js.Seen] = []

        async def run(s: js.Sample) -> None:
            got.append(s.seen)

        def boom() -> js.Seen:
            raise RuntimeError("schema render broke")

        assert lane.start(ROLE, "p", self._ok(), mod=0, run=run, seen=boom) is None, "undrawn: never called"
        assert lane.start(ROLE, "p", self._ok(), mod=1, run=run, seen=boom) is not None
        await lane.drain()
        assert got == [js.Seen()]
        failed = [f for lv, ev, f in events if ev == "llm_judge_seen_failed" and lv == logging.WARNING]
        assert len(failed) == 1 and "schema render broke" in failed[0]["error"]

    async def test_a_sampler_failure_never_reaches_the_caller(self) -> None:
        lane, events = self._lane()

        async def run(_s: js.Sample) -> None: ...

        def broken(_r: Role, _m: int) -> bool:
            raise RuntimeError("gate bug")

        assert lane.start(ROLE, "p", self._ok(), mod=1, run=run, eligible=broken) is None
        assert [(lv, ev) for lv, ev, _ in events] == [(logging.WARNING, "llm_judge_sampler_failed")]
        assert "gate bug" in events[0][2]["error"]

    async def test_the_task_starts_in_a_fresh_context_not_the_callers(self) -> None:
        lane, _ = self._lane()
        seen: list[str | None] = []

        async def run(_s: js.Sample) -> None:
            seen.append(_UNIT.get())

        token = _UNIT.set("the caller's unit")
        try:
            assert lane.start(ROLE, "p", self._ok(), mod=1, run=run) is not None
        finally:
            _UNIT.reset(token)
        await lane.drain()
        assert seen == [None]

    async def test_drain_survives_a_task_that_raised(self) -> None:
        lane, _ = self._lane()

        async def run(_s: js.Sample) -> None:
            raise RuntimeError("the store refused")

        assert lane.start(ROLE, "p", self._ok(), mod=1, run=run) is not None
        await lane.drain()
        assert lane.pending() == 0

    async def test_without_a_hook_events_go_to_the_stdlib_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        lane = js.JudgeLane(name="lane")

        async def run(_s: js.Sample) -> None: ...

        with caplog.at_level(logging.INFO, logger="llm_ports.evals.judge"):
            assert lane.start(ROLE, "p", self._ok(), mod=1, run=run) is not None
        await lane.drain()
        assert any(r.getMessage().startswith("lane_sampled") for r in caplog.records)
