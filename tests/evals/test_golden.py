"""golden: `fire` renders each item's slots through the candidate source and the call path; `grade`
puts the role's call graders and any extra row graders on one board; a row the model never answered
is named, not graded."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from llm_ports.evals import GauntletEvaluator, golden
from llm_ports.ports import Generated, GenRequest
from llm_ports.prompts import load
from llm_ports.testing import StubGenerator
from tests.conftest import PACKS, Role

ROLE = Role("test.golden", tier="local", pack="demo.one", graders=("schema_complete", "echo"))
PACK = load("demo.one", PACKS)
ITEMS: list[dict[str, Any]] = [
    {"id": "g-1", "stratum": "win", "slots": {"who": "a clerk", "what": "hello"}, "known": ["a"]},
    {"id": "g-2", "stratum": "loss", "slots": {"who": "a judge", "what": ""}, "known": ["b"]},
]
REPLY = {"summary": "The clerk greets the room.", "cited": ["a"], "note": None}


def _output(_item: Mapping[str, Any]) -> type[dict[str, Any]]:
    return dict


async def _fire(gen: StubGenerator, attempts: list[Any] | None = None, **kw: Any) -> dict[str, dict[str, Any]]:
    return await golden.fire(
        ITEMS,
        role=ROLE,
        pack=PACK,
        source=PACK.source,
        output=_output,
        generator_for=lambda: (gen, attempts if attempts is not None else []),
        **kw,
    )


class TestFire:
    async def test_each_items_slots_render_through_the_source_and_the_call_path(self) -> None:
        gen = StubGenerator(REPLY)
        outs = await _fire(gen, ["attempt-0"])
        assert set(outs) == {"g-1", "g-2"}
        for it in ITEMS:
            o = outs[it["id"]]
            assert o["prompt"] == PACK.render(**it["slots"]) and o["value"] == REPLY
            assert o["verdict"] == "ok" and o["outcome"] == "ok" and o["attempts"] == ["attempt-0"]
            assert [s["name"] for s in o["scores"]] == ["schema_complete", "echo"]
        assert sorted(r.prompt for r in gen.requests) == sorted(o["prompt"] for o in outs.values())

    async def test_a_candidate_source_is_what_renders(self) -> None:
        outs = await golden.fire(
            ITEMS[:1],
            role=ROLE,
            pack=PACK,
            source="Candidate for {{ who }}.",
            output=_output,
            generator_for=lambda: (StubGenerator(REPLY), []),
        )
        assert outs["g-1"]["prompt"] == "Candidate for a clerk."

    async def test_a_failed_call_is_an_error_verdict_never_a_raise(self) -> None:
        def down(req: GenRequest, _out: type[Any]) -> Any:
            return Generated(value=None, outcome="unavailable", model=req.model, reason="wire down")

        events: list[tuple[int, str, Mapping[str, Any]]] = []
        outs = await _fire(StubGenerator(down), on_event=lambda lv, ev, f: events.append((lv, ev, f)))
        assert {o["verdict"] for o in outs.values()} == {"error"}
        assert outs["g-1"]["reason"] == "wire down" and outs["g-1"]["scores"] == []
        assert [(lv, ev, f["outcome"]) for lv, ev, f in events] == [(logging.INFO, "golden_item", "unavailable")] * 2

    async def test_a_consumer_facade_replaces_the_call_path(self) -> None:
        seen: list[str] = []

        async def facade(role: Role, *, prompt: str, output: Any, generator: Any) -> Generated[Any]:
            seen.append(prompt)
            return Generated(value=REPLY, outcome="gated", model="m", reason="switched off")

        outs = await _fire(StubGenerator(REPLY), generate=facade)
        assert len(seen) == 2 and {o["verdict"] for o in outs.values()} == {"gated"}


class TestGrade:
    async def test_call_graders_and_extra_row_graders_share_one_board(self) -> None:
        bad = {**REPLY, "cited": ["ghost"]}
        outs = await _fire(StubGenerator(bad))
        extra = {"citations_resolve": golden.citations_resolve("cited", "known")}
        board, rows = golden.grade(ITEMS, outs, ROLE, GauntletEvaluator(), extra=extra)
        assert board["surface"] == "demo.one" and board["n"] == 2
        assert set(board["by_grader"]) == {"schema_complete", "echo", "citations_resolve"}
        cite = board["by_grader"]["citations_resolve"]
        assert cite["fail"] == 2 and "ghost" in cite["examples"][0]["reason"]
        assert board["by_grader"]["schema_complete"]["fail"] == 2, "`note: None` on a dict is unanswered"
        assert board["monitors"] == {"answered": 2, "mean_retries": 0.0}
        assert all(r["known"] and r["stratum"] and "slots" not in r for r in rows), "the item's keys ride along"

    def test_a_row_the_model_never_answered_is_named_not_graded(self) -> None:
        outs = {"g-1": {"value": None, "outcome": "unavailable", "reason": "box down", "prompt": "p"}}
        board, rows = golden.grade(ITEMS, outs, ROLE, GauntletEvaluator())
        assert board["rows_failed"] == 2
        assert board["by_grader"]["echo"]["examples"][0] == {"id": "g-1", "reason": "no reply: box down"}
        assert rows[1]["outcome"] == "not_run", "an item with no result is not_run"

    def test_the_surface_falls_back_to_the_role_name_without_a_pack(self) -> None:
        board, _ = golden.grade([], {}, Role("test.packless", tier="local"), GauntletEvaluator())
        assert board["surface"] == "test.packless"

    def test_monitors_average_the_answered_rows_retries(self) -> None:
        rows = [{"value": 1, "retries": 1}, {"value": 2, "retries": 2}, {"value": None, "retries": 5}]
        assert golden.monitors(rows) == {"answered": 2, "mean_retries": 1.5}


class TestCitationsResolve:
    def test_every_shape_a_reply_can_take(self) -> None:
        g = golden.citations_resolve("cited", "known")
        assert g({"value": {"cited": ["a"]}, "known": ["a", "b"]}) == (True, "")
        assert g({"value": {"cited": []}, "known": []}) == (True, "")
        assert g({"value": None})[0] is False
        assert g({"value": {"cited": "a"}, "known": ["a"]}) == (False, "cited is str, not a list")
        ok, why = g({"value": {"cited": ["a", "z"]}, "known": ["a"]})
        assert not ok and "z" in why
