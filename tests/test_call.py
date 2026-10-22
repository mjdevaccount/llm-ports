"""call.generate: role -> feature -> admit -> one observation -> the Generator -> graders -> hooks.

A `StubGenerator` stands in for the wire, a `RecordingTracer` (through `tracer_trace`) for the
trace backend. What is pinned: the role resolves to the request, the category is the feature
during the call, a refused role opens no observation, every non-ok outcome ends the observation in
error while the call still returns, and the graders/scores/ok hooks fire exactly when they should.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from llm_ports import call, call_graders
from llm_ports.context import current_feature
from llm_ports.failure import GenerationFailed
from llm_ports.ports import Generated, GenRequest
from llm_ports.testing import RecordingTracer, StubGenerator
from tests.conftest import Role

GRADERS = ("schema_complete", "numbers_from_inputs", "echo")
ROLE = Role("test.facade", tier="box", max_tokens=300, temperature=0.2, settings={"seed": 7}, pack="test.pack")
WEB = Role("test.facade_web", tier="box-web")
GRADED = Role("test.graded", tier="box", max_tokens=300, graders=GRADERS)


def _rig(result: Any) -> tuple[StubGenerator, RecordingTracer, call.Hooks, list[tuple[int, str, dict[str, Any]]]]:
    seen: list[str | None] = []

    def reply(_req: GenRequest, _out: type[Any]) -> Any:
        seen.append(current_feature())
        return result

    gen = StubGenerator(reply)
    gen.features = seen  # type: ignore[attr-defined]
    tracer = RecordingTracer()
    events: list[tuple[int, str, dict[str, Any]]] = []
    hooks = call.Hooks(trace=call.tracer_trace(tracer), on_event=lambda lvl, ev, f: events.append((lvl, ev, dict(f))))
    return gen, tracer, hooks, events


def test_the_role_resolves_to_one_request() -> None:
    req = call.request_for(ROLE, prompt="p", system="s", settings={"seed": 9})
    assert (req.model, req.timeout_s, req.box, req.label) == ("small-chat", 120.0, True, "test.facade")
    assert req.settings["max_tokens"] == 300 and req.settings["temperature"] == 0.2
    assert req.settings["seed"] == 9, "a per-call override beats the role's default"
    assert req.settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_a_web_role_gets_the_web_rung_of_the_ladder() -> None:
    req = call.request_for(WEB, prompt="p")
    assert req.model == "small-web" and req.timeout_s == 260.0
    assert req.images == (), "no images = an empty tuple, never None"


async def test_image_bytes_reach_the_generator_untouched() -> None:
    gen, _t, hooks, _e = _rig("a chart")
    await call.generate(ROLE, prompt="read it", generator=gen, hooks=hooks,
                        images=[(b"\x89PNG", "image/png"), (b"\xff\xd8", "image/jpeg")])
    assert gen.requests[0].images == ((b"\x89PNG", "image/png"), (b"\xff\xd8", "image/jpeg"))


async def test_ok_traces_one_generation_under_the_category() -> None:
    gen, tracer, hooks, events = _rig(Generated(value="fine", outcome="ok", model="small-chat",
                                                usage={"input": 3, "output": 4}))
    res = await call.generate(ROLE, prompt="say it", generator=gen, hooks=hooks, metadata={"symbol": "SPY"})
    assert res.outcome == "ok" and res.value == "fine"
    assert gen.features == ["test.facade"], "the category is the feature for the whole call"  # type: ignore[attr-defined]
    assert current_feature() is None, "and is reset after it"
    (g,) = tracer.generations
    assert g.opened["name"] == "test.facade" and g.opened["model"] == "small-chat"
    assert g.opened["prompt_name"] == "test.pack" and g.opened["metadata"] == {"role": "test.facade", "symbol": "SPY"}
    assert g.ended is not None and g.ended["level"] == "DEFAULT"
    assert g.ended["usage"] == {"input": 3, "output": 4} and g.ended["output"] == "fine"
    assert res.stamp["generation"] is not None and events == []


async def test_unmeasured_usage_is_no_usage_never_zero() -> None:
    gen, tracer, hooks, _e = _rig(Generated(value="fine", outcome="ok", usage=None))
    await call.generate(ROLE, prompt="x", generator=gen, hooks=hooks)
    assert tracer.generations[0].ended["usage"] is None  # type: ignore[index]


@pytest.mark.parametrize("outcome", ["unavailable", "request_fault", "truncated", "invalid"])
async def test_a_failed_outcome_is_an_error_observation_and_still_returns(outcome: str) -> None:
    gen, tracer, hooks, events = _rig(Generated(value=None, outcome=outcome, reason="why", retries=None))  # type: ignore[arg-type]
    res = await call.generate(ROLE, prompt="x", generator=gen, hooks=hooks)
    assert res.outcome == outcome and res.value is None
    ended = tracer.generations[0].ended
    assert ended is not None and ended["level"] == "ERROR" and outcome in ended["status_message"]
    assert events == [(logging.WARNING, "llm_call_failed", {"role": "test.facade", "outcome": outcome, "reason": "why"})]


async def test_an_admit_refusal_opens_no_observation() -> None:
    gen, tracer, _h, _e = _rig("never")
    refused: list[str] = []

    def admit(role: Role) -> str | None:
        refused.append(role.category)
        return f"category_disabled: {role.category} is switched off"

    res = await call.generate(ROLE, prompt="x", generator=gen,
                              hooks=call.Hooks(admit=admit, trace=call.tracer_trace(tracer)))
    assert res.outcome == "gated" and "switched off" in (res.reason or "") and res.model == "small-chat"
    assert gen.requests == [] and tracer.generations == [] and refused == ["test.facade"]


async def test_a_gate_refusal_on_the_wire_is_not_an_error() -> None:
    gen, tracer, hooks, events = _rig(Generated(value=None, outcome="gated", reason="category_capped"))
    res = await call.generate(ROLE, prompt="x", generator=gen, hooks=hooks)
    assert res.outcome == "gated" and tracer.generations[0].ended["level"] == "DEFAULT"  # type: ignore[index]
    assert events == []


async def test_no_hooks_is_role_generator_graders_and_logs_to_stdlib(caplog: pytest.LogCaptureFixture) -> None:
    gen = StubGenerator(Generated(value=None, outcome="unavailable", reason="down"))
    with caplog.at_level(logging.WARNING, logger="llm_ports.call"):
        res = await call.generate(ROLE, prompt="x", generator=gen)
    assert res.outcome == "unavailable" and "llm_call_failed" in caplog.text


async def test_a_raising_generator_inside_the_trace_ends_it_in_error_and_propagates() -> None:
    class Boom:
        async def generate(self, *_a: Any, **_k: Any) -> Generated[Any]:
            raise RuntimeError("adapter bug")

    tracer = RecordingTracer()
    with pytest.raises(RuntimeError, match="adapter bug"):
        await call.generate(ROLE, prompt="x", generator=Boom(), hooks=call.Hooks(trace=call.tracer_trace(tracer)))
    assert tracer.generations[0].ended["level"] == "ERROR"  # type: ignore[index]


def test_require_ok_raises_the_named_failure() -> None:
    assert call.require_ok(Generated(value=1, outcome="ok")) == 1
    with pytest.raises(GenerationFailed) as exc:
        call.require_ok(Generated(value=None, outcome="truncated", reason="cut"))
    assert (exc.value.outcome, exc.value.reason, str(exc.value)) == ("truncated", "cut", "truncated: cut")
    with pytest.raises(GenerationFailed, match="gated"):
        call.require_ok(Generated(value=None, outcome="gated"))


def test_trace_output_is_json_for_models_and_mappings_and_capped() -> None:
    from pydantic import BaseModel

    class M(BaseModel):
        a: int

    assert call.trace_output(None) == "" and call.trace_output(M(a=1)) == '{"a":1}'
    assert call.trace_output({"k": 1}) == '{"k": 1}'
    assert len(call.trace_output("x" * 9000)) == call.OUTPUT_TRACE_MAX


def test_json_output_is_a_port_type_carrying_the_schema() -> None:
    t = call.json_output({"type": "object"}, "verdicts")
    assert t.json_schema == {"type": "object"} and t.schema_name == "verdicts"  # type: ignore[attr-defined]


class TestGraders:
    async def test_an_ok_reply_is_scored_per_grader_onto_the_observation(self) -> None:
        gen, tracer, _h, _e = _rig({"summary": "fine"})
        noted: list[tuple[str, tuple[Any, ...]]] = []

        async def on_scores(role: Role, scores: tuple[Any, ...]) -> None:
            noted.append((role.name, scores))

        res = await call.generate(GRADED, prompt="say it", generator=gen,
                                  hooks=call.Hooks(trace=call.tracer_trace(tracer), on_scores=on_scores))
        names = [f"test.graded.{g}" for g in GRADERS]
        assert [s[0] for s in tracer.generations[0].scores] == names
        assert [s.name for s in res.stamp["scores"]] == list(GRADERS) and all(s.passed for s in res.stamp["scores"])
        assert noted and noted[0][0] == "test.graded" and noted[0][1] == res.stamp["scores"]

    async def test_a_non_ok_outcome_scores_nothing_and_offers_nothing(self) -> None:
        gen, tracer, hooks, _e = _rig(Generated(value=None, outcome="truncated", reason="cut"))
        offered: list[Any] = []
        hooks = call.Hooks(trace=hooks.trace, on_event=hooks.on_event, on_ok=lambda *a: offered.append(a))
        res = await call.generate(GRADED, prompt="x", generator=gen, hooks=hooks)
        assert tracer.generations[0].scores == [] and "scores" not in res.stamp and offered == []

    async def test_a_failed_grader_is_an_event_and_never_changes_the_outcome(self) -> None:
        gen, _t, hooks, events = _rig({"summary": "", "n": "the number 4242 came from nowhere"})
        res = await call.generate(GRADED, prompt="say it", generator=gen, hooks=hooks)
        assert res.outcome == "ok"
        assert [(e[1], e[2]["failed"]) for e in events] == [("llm_call_graded", ["schema_complete", "numbers_from_inputs"])]

    async def test_a_consumer_grader_table_is_used_and_a_raise_is_a_crashed_score(self) -> None:
        def boom(*_a: Any) -> Any:
            raise RuntimeError("grader bug")

        def in_vocab(_p: str, value: Any, _r: Any) -> call_graders.CallScore:
            ok = value.get("category") in {"groceries"}
            return call_graders.CallScore("in_vocabulary", 1.0 if ok else 0.0, ok, "" if ok else "oov")

        role = Role("test.vocab", tier="cloud", graders=("in_vocabulary", "echo"))
        gen, tracer, hooks, _e = _rig({"category": "rocket fuel"})
        table = {**call_graders.GRADERS, "in_vocabulary": in_vocab, "echo": boom}
        res = await call.generate(role, prompt="x", generator=gen,
                                  hooks=call.Hooks(trace=hooks.trace, on_event=hooks.on_event, graders=table))
        scores = {s.name: s for s in res.stamp["scores"]}
        assert scores["in_vocabulary"].passed is False and scores["in_vocabulary"].reason == "oov"
        assert scores["echo"].value == call_graders.CRASHED and "RuntimeError: grader bug" in scores["echo"].reason
        assert res.outcome == "ok" and len(tracer.generations[0].scores) == 2
        assert call_graders.GRADERS["echo"] is call_graders.echo, "the package table is never mutated"


async def test_an_ok_reply_is_offered_to_on_ok_with_its_observation() -> None:
    gen, _t, hooks, _e = _rig("fine")
    offered: list[tuple[Any, str, Any]] = []
    hooks = call.Hooks(trace=hooks.trace, on_ok=lambda r, p, res: offered.append((r, p, res)))
    res = await call.generate(ROLE, prompt="say it", generator=gen, hooks=hooks)
    assert offered and offered[0][0] is ROLE and offered[0][1] == "say it"
    assert offered[0][2].stamp["generation"] is res.stamp["generation"]
