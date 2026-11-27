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

from llm_ports import call
from llm_ports.call import _gate, _generate
from llm_ports.call import graders as call_graders
from llm_ports.observability import current_feature
from llm_ports.ports import Generated, GenRequest
from llm_ports.ports.outcomes import GenerationFailed
from llm_ports.testing import RecordingTracer, Reply, StubGenerator
from tests.conftest import Role

GRADERS = ("schema_complete", "numbers_from_inputs", "echo")
ROLE = Role("test.facade", tier="local", max_tokens=300, temperature=0.2, settings={"seed": 7}, pack="test.pack")
WEB = Role("test.facade_web", tier="local-web")
GRADED = Role("test.graded", tier="local", max_tokens=300, graders=GRADERS)


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
    assert (req.model, req.timeout_s, req.self_hosted, req.label) == ("small-chat", 120.0, True, "test.facade")
    assert req.settings["max_tokens"] == 300 and req.settings["temperature"] == 0.2
    assert req.settings["seed"] == 9, "a per-call override beats the role's default"
    assert req.settings["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_a_web_role_gets_the_web_rung_of_the_ladder() -> None:
    req = call.request_for(WEB, prompt="p")
    assert req.model == "small-web" and req.timeout_s == 260.0
    assert req.images == (), "no images = an empty tuple, never None"


async def test_image_bytes_reach_the_generator_untouched() -> None:
    gen, _t, hooks, _e = _rig("a chart")
    await call.generate(
        ROLE,
        prompt="read it",
        generator=gen,
        hooks=hooks,
        images=[(b"\x89PNG", "image/png"), (b"\xff\xd8", "image/jpeg")],
    )
    assert gen.requests[0].images == ((b"\x89PNG", "image/png"), (b"\xff\xd8", "image/jpeg"))


async def test_ok_traces_one_generation_under_the_category() -> None:
    gen, tracer, hooks, events = _rig(
        Generated(value="fine", outcome="ok", model="small-chat", usage={"input": 3, "output": 4})
    )
    res = await call.generate(ROLE, prompt="say it", generator=gen, hooks=hooks, metadata={"region": "NYC"})
    assert res.outcome == "ok" and res.value == "fine"
    assert gen.features == ["test.facade"], "the category is the feature for the whole call"  # type: ignore[attr-defined]
    assert current_feature() is None, "and is reset after it"
    (g,) = tracer.generations
    assert g.opened["name"] == "test.facade" and g.opened["model"] == "small-chat"
    assert g.opened["prompt_name"] == "test.pack" and g.opened["metadata"] == {"role": "test.facade", "region": "NYC"}
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
    assert events == [
        (logging.WARNING, "llm_call_failed", {"role": "test.facade", "outcome": outcome, "kind": None, "reason": "why"})
    ]


async def test_an_admit_refusal_opens_no_observation() -> None:
    gen, tracer, _h, _e = _rig("never")
    refused: list[str] = []

    def admit(role: Role) -> str | None:
        refused.append(role.category)
        return f"category_disabled: {role.category} is switched off"

    res = await call.generate(
        ROLE, prompt="x", generator=gen, hooks=call.Hooks(admit=admit, trace=call.tracer_trace(tracer))
    )
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

    assert _generate._trace_output(None) == "" and _generate._trace_output(M(a=1)) == '{"a":1}'
    assert _generate._trace_output({"k": 1}) == '{"k": 1}'
    assert len(_generate._trace_output("x" * 9000)) == call.OUTPUT_TRACE_MAX


def test_json_output_is_a_port_type_carrying_the_schema() -> None:
    t = call.json_output({"type": "object"}, "verdicts")
    assert t.json_schema == {"type": "object"} and t.schema_name == "verdicts"  # type: ignore[attr-defined]


class TestGraders:
    async def test_an_ok_reply_is_scored_per_grader_onto_the_observation(self) -> None:
        gen, tracer, _h, _e = _rig({"summary": "fine"})
        noted: list[tuple[str, tuple[Any, ...]]] = []

        async def on_scores(role: Role, scores: tuple[Any, ...]) -> None:
            noted.append((role.name, scores))

        res = await call.generate(
            GRADED,
            prompt="say it",
            generator=gen,
            hooks=call.Hooks(trace=call.tracer_trace(tracer), on_scores=on_scores),
        )
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
        assert [(e[1], e[2]["failed"]) for e in events] == [
            ("llm_call_graded", ["schema_complete", "numbers_from_inputs"])
        ]

    async def test_a_consumer_grader_table_is_used_and_a_raise_is_a_crashed_score(self) -> None:
        def boom(*_a: Any) -> Any:
            raise RuntimeError("grader bug")

        def in_vocab(_p: str, value: Any, _r: Any) -> call_graders.CallScore:
            ok = value.get("category") in {"groceries"}
            return call_graders.CallScore("in_vocabulary", 1.0 if ok else 0.0, ok, "" if ok else "oov")

        role = Role("test.vocab", tier="cloud", graders=("in_vocabulary", "echo"))
        gen, tracer, hooks, _e = _rig({"category": "rocket fuel"})
        table = {**call_graders.GRADERS, "in_vocabulary": in_vocab, "echo": boom}
        res = await call.generate(
            role, prompt="x", generator=gen, hooks=call.Hooks(trace=hooks.trace, on_event=hooks.on_event, graders=table)
        )
        scores = {s.name: s for s in res.stamp["scores"]}
        assert scores["in_vocabulary"].passed is False and scores["in_vocabulary"].reason == "oov"
        assert scores["echo"].value == call_graders.CRASHED and "RuntimeError: grader bug" in scores["echo"].reason
        assert res.outcome == "ok" and len(tracer.generations[0].scores) == 2
        assert call_graders.GRADERS["echo"] is call_graders.echo, "the package table is never mutated"


GATED = Role("test.gated", tier="local", graders=GRADERS, gates=("numbers_from_inputs",))
INVENTED = {"summary": "wait for a 15-minute retest"}
GROUNDED = {"summary": "held 14.88 of room"}
PROMPT = "room_down 14.878"


class TestGates:
    """A grader the role names in `gates` decides whether the reply is returned. Both directions:
    the same reply is `ok` on the ungated role and refused on the gated one, and a reply the gate
    passes is returned exactly as an ungated role's."""

    @staticmethod
    def _hooks(tracer: RecordingTracer, **more: Any) -> call.Hooks:
        return call.Hooks(trace=call.tracer_trace(tracer), **more)

    async def test_a_failed_gate_is_re_asked_and_the_fixed_reply_is_ok(self) -> None:
        gen, tracer = StubGenerator([Reply.ok(INVENTED), Reply.ok(GROUNDED)]), RecordingTracer()
        offered: list[Any] = []
        res = await call.generate(
            GATED, prompt=PROMPT, generator=gen, hooks=self._hooks(tracer, on_ok=lambda *a: offered.append(a))
        )
        gen.assert_spent()
        assert (res.outcome, res.value, res.retries, res.problems) == ("ok", GROUNDED, 1, ())
        assert all(s.passed for s in res.stamp["scores"]) and call.GATES_FAILED not in res.stamp
        assert len(offered) == 1 and tracer.generations[0].ended is not None
        assert tracer.generations[0].ended["level"] == "DEFAULT"
        assert "numbers_from_inputs: numbers no input rendered: ['15']" in gen.reasks[0], "the model is told why"

    async def test_a_re_ask_that_fixed_the_reply_still_leaves_a_witness(self) -> None:
        """Every score passes on the kept reply, so the stamp, the trace and an event say one was refused."""
        gen, tracer = StubGenerator([Reply.ok(INVENTED), Reply.ok(GROUNDED)]), RecordingTracer()
        events: list[tuple[int, str, dict[str, Any]]] = []
        hooks = self._hooks(tracer, on_event=lambda lvl, ev, f: events.append((lvl, ev, dict(f))))
        res = await call.generate(GATED, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "ok" and res.stamp[call.GATES_REASKED] == ("numbers_from_inputs",)
        assert events == [
            (
                logging.INFO,
                "llm_gate_reasked",
                {
                    "role": "test.gated",
                    "gates": ["numbers_from_inputs"],
                    "reasons": {"numbers_from_inputs": "numbers no input rendered: ['15']"},
                },
            )
        ]
        ended = tracer.generations[0].ended
        assert ended is not None and ended["metadata"][call.GATES_REASKED] == ["numbers_from_inputs"]

    async def test_a_reply_that_passed_first_time_carries_no_re_ask_witness(self) -> None:
        gen, tracer, hooks, events = _rig(GROUNDED)
        res = await call.generate(GATED, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "ok" and call.GATES_REASKED not in res.stamp and events == []
        ended = tracer.generations[0].ended
        assert ended is not None and call.GATES_REASKED not in ended["metadata"]

    async def test_a_gate_is_matched_by_its_table_key_not_by_the_score_name(self) -> None:
        """A built-in bound under another key still scores as `schema_complete`. Both directions:
        the gate named by that key refuses a blank reply, and returns an answered one."""
        from functools import partial

        table = {**call_graders.GRADERS, "complete_nullable": partial(call_graders.schema_complete, nullable={"b"})}
        role = Role("test.keyed", tier="local", graders=("echo", "complete_nullable"), gates=("complete_nullable",))
        blank, answered = {"a": "", "b": None}, {"a": "fine", "b": None}
        gen, tracer = StubGenerator([Reply.ok(blank), Reply.ok(blank)]), RecordingTracer()
        res = await call.generate(role, prompt=PROMPT, generator=gen, hooks=self._hooks(tracer, graders=table))
        gen.assert_spent()
        assert res.outcome == "invalid" and res.stamp[call.GATES_FAILED] == ("complete_nullable",)
        assert res.reason == "gate: complete_nullable: empty: ['a']"
        assert [s.name for s in res.stamp["scores"]] == ["echo", "schema_complete"], "the score keeps its own name"
        assert "complete_nullable: empty: ['a']" in gen.reasks[0], "and the re-ask names the gate by its key"
        gen, tracer = StubGenerator([Reply.ok(answered)]), RecordingTracer()
        res = await call.generate(role, prompt=PROMPT, generator=gen, hooks=self._hooks(tracer, graders=table))
        assert res.outcome == "ok" and res.value == answered and res.retries == 0

    def test_a_gate_with_no_score_refuses(self) -> None:
        """`refused` reads the scores by position; a gate past the end of them was not run."""
        ok = Generated(value="x", outcome="ok", stamp={"scores": ()})
        res = _gate.refused(GATED, ok)
        assert res.outcome == "invalid" and res.stamp[call.GATES_FAILED] == ("numbers_from_inputs",)
        assert res.reason == "gate: numbers_from_inputs: the gate was not scored"

    async def test_a_gate_the_table_does_not_hold_is_a_request_fault_before_the_wire(self) -> None:
        """It could only crash and refuse every reply after its re-ask. Both directions: the same
        role is served once the table holds the grader."""
        role = Role("test.typo", tier="local", graders=("in_vocab",), gates=("in_vocab",))
        gen, tracer, hooks, events = _rig({"summary": "fine"})
        res = await call.generate(role, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "request_fault" and gen.requests == [] and tracer.generations == []
        assert res.reason == "gate: role 'test.typo' gates on ['in_vocab'], which the grader table does not hold"
        assert [(e[1], e[2]["outcome"]) for e in events] == [("llm_call_failed", "request_fault")]
        table = {"in_vocab": lambda _p, _v, _r: call_graders.CallScore("in_vocab", 1.0, True)}
        held = call.Hooks(trace=hooks.trace, graders=table)
        res = await call.generate(role, prompt=PROMPT, generator=gen, hooks=held)
        assert res.outcome == "ok" and len(gen.requests) == 1

    async def test_a_reply_still_failing_after_the_re_ask_is_refused_and_still_counted(self) -> None:
        gen, tracer = StubGenerator([Reply.ok(INVENTED), Reply.ok(INVENTED)]), RecordingTracer()
        noted: list[tuple[str, tuple[Any, ...]]] = []
        offered: list[Any] = []
        events: list[tuple[str, dict[str, Any]]] = []

        async def on_scores(role: Role, scores: tuple[Any, ...]) -> None:
            noted.append((role.name, scores))

        hooks = self._hooks(
            tracer,
            on_scores=on_scores,
            on_ok=lambda *a: offered.append(a),
            on_event=lambda _lvl, ev, f: events.append((ev, dict(f))),
        )
        res = await call.generate(GATED, prompt=PROMPT, generator=gen, hooks=hooks)
        gen.assert_spent()
        assert (res.outcome, res.value) == ("invalid", None), "a refused reply carries no value to store"
        assert res.failure is not None and res.failure.kind == "invalid_output"
        assert res.reason == "gate: numbers_from_inputs: numbers no input rendered: ['15']"
        assert res.stamp[call.GATES_FAILED] == ("numbers_from_inputs",)
        failed = [s.name for s in res.stamp["scores"] if not s.passed]
        assert failed == ["numbers_from_inputs"], "the post-hoc score is still stamped"
        assert noted == [("test.gated", res.stamp["scores"])], (
            "and still handed to on_scores: the counter keeps counting"
        )
        assert offered == [], "on_ok never sees a refused reply"
        assert [e[0] for e in events] == ["llm_gate_reasked", "llm_call_graded", "llm_call_failed"]
        assert events[2][1]["outcome"] == "invalid" and events[2][1]["kind"] == "invalid_output"
        assert res.stamp[call.GATES_REASKED] == ("numbers_from_inputs",)
        ended = tracer.generations[0].ended
        assert ended is not None and ended["level"] == "ERROR"
        assert ended["metadata"]["outcome"] == "invalid" and ended["metadata"][call.GATES_FAILED] == [
            "numbers_from_inputs"
        ]
        assert ("test.gated.numbers_from_inputs", 1.0) == tracer.generations[0].scores[1][:2]

    async def test_the_same_reply_on_the_ungated_role_is_ok(self) -> None:
        gen, _t, hooks, _e = _rig(INVENTED)
        res = await call.generate(GRADED, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "ok" and res.value == INVENTED and res.retries == 0

    async def test_a_grader_that_is_not_a_gate_never_refuses(self) -> None:
        """`schema_complete` fails here and is only a grader on this role: measured, not refused."""
        gen, _t, hooks, events = _rig({"summary": ""})
        res = await call.generate(GATED, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "ok" and res.retries == 0
        assert [e[2]["failed"] for e in events] == [["schema_complete"]]

    async def test_no_re_asks_left_refuses_on_the_first_reply(self) -> None:
        role = Role("test.gated_once", tier="local", graders=GRADERS, gates=("numbers_from_inputs",), output_retries=0)
        gen, _t, hooks, _e = _rig(INVENTED)
        res = await call.generate(role, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "invalid" and res.retries == 0 and len(gen.requests) == 1

    async def test_the_callers_validator_runs_first_and_its_problems_win(self) -> None:
        seen: list[Any] = []

        def validate(value: dict[str, str]) -> list[str]:
            seen.append(value)
            return ["too short"] if len(value["summary"]) < 12 else []

        gen, _t, hooks, _e = _rig({"summary": "15"})
        res = await call.generate(GATED, prompt=PROMPT, output=dict, generator=gen, hooks=hooks, validate=validate)
        assert len(seen) == 2 and res.outcome == "invalid", "kept with the caller's problems, then refused by the gate"
        assert res.problems == ("too short",), "the gate was not asked about a reply the caller refused"

    async def test_a_gate_blind_to_the_calls_inputs_is_dropped_with_its_grader_and_says_so(self) -> None:
        """An image or a web tier's pages are inputs no text grader sees: no grader, so no gate. The
        reply goes back unchecked, so the result names the gate it skipped and the log says it once."""
        warned: list[dict[str, Any]] = []
        for extra in ({"images": [(b"png", "image/png")]}, {"web": True}):
            gen, tracer, hooks, events = _rig(INVENTED)
            res = await call.generate(GATED, prompt=PROMPT, generator=gen, hooks=hooks, **extra)
            assert res.outcome == "ok" and [s.name for s in res.stamp["scores"]] == ["schema_complete", "echo"]
            assert res.stamp[call.GATES_DROPPED] == ("numbers_from_inputs",), "an invented figure went back ok"
            ended = tracer.generations[0].ended
            assert ended is not None and ended["metadata"][call.GATES_DROPPED] == ["numbers_from_inputs"]
            warned += [f for lvl, ev, f in events if ev == "llm_call_warning" and lvl == logging.WARNING]
        assert [w["setting"] for w in warned] == ["gates"], "once a process for the role, not once a call"
        assert "test.gated" in warned[0]["detail"] and "numbers_from_inputs" in warned[0]["detail"]
        assert GATED.gates == ("numbers_from_inputs",), "the role itself is untouched"

    async def test_a_call_that_applies_every_gate_carries_no_dropped_stamp(self) -> None:
        gen, _t, hooks, events = _rig(GROUNDED)
        res = await call.generate(GATED, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "ok" and call.GATES_DROPPED not in res.stamp
        assert not [ev for _lvl, ev, _f in events if ev == "llm_call_warning"]

    async def test_a_re_ask_for_a_gate_and_another_problem_together_still_records_the_gate(self) -> None:
        """A reply may carry a problem of its own AND fail a gate. The caller's message builder words
        that re-ask, and the gate's part of it is still an event and a stamp."""
        gen = StubGenerator([Reply.problems(INVENTED, ["too short"]), Reply.ok(GROUNDED)])
        events: list[tuple[int, str, dict[str, Any]]] = []
        hooks = self._hooks(RecordingTracer(), on_event=lambda lvl, ev, f: events.append((lvl, ev, dict(f))))
        res = await call.generate(GATED, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "ok" and res.stamp[call.GATES_REASKED] == ("numbers_from_inputs",)
        assert [f["gates"] for _lvl, ev, f in events if ev == "llm_gate_reasked"] == [["numbers_from_inputs"]]
        assert "too short" in gen.reasks[0] and "numbers_from_inputs" in gen.reasks[0]

    async def test_a_crashed_gate_refuses_never_passes(self) -> None:
        def boom(*_a: Any) -> Any:
            raise RuntimeError("grader bug")

        role = Role("test.gated_boom", tier="local", graders=("echo",), gates=("echo",))
        gen, tracer, _h, _e = _rig({"summary": "fine"})
        hooks = self._hooks(tracer, graders={**call_graders.GRADERS, "echo": boom})
        res = await call.generate(role, prompt=PROMPT, generator=gen, hooks=hooks)
        assert res.outcome == "invalid" and "grader raised RuntimeError" in (res.reason or "")

    def test_the_re_ask_names_the_gate_and_a_caller_problem_goes_to_the_callers_message(self) -> None:
        told: list[Any] = []
        arm = _gate.armed(GATED, PROMPT, None, None, lambda problems: f"caller: {problems}", told.append)
        check, message = arm.validate, arm.retry_message
        assert check is not None and message is not None
        problems = check(INVENTED)
        assert [str(p) for p in problems] == ["numbers_from_inputs: numbers no input rendered: ['15']"]
        assert arm.reasked == [] and told == [], "a failed check is not yet a re-ask"
        assert "numbers_from_inputs: numbers no input rendered: ['15']" in message(problems)
        assert message(["too short"]) == "caller: ['too short']"
        assert check(GROUNDED) == []
        assert arm.gates_reasked == ("numbers_from_inputs",) and told == [problems], "only the gate re-ask counts"

    def test_an_ungated_role_keeps_the_callers_validator_untouched(self) -> None:
        def validate(_v: Any) -> list[str]:
            return []

        arm = _gate.armed(GRADED, PROMPT, None, validate, None)
        assert (arm.validate, arm.retry_message, arm.reasked) == (validate, None, [])


async def test_an_ok_reply_is_offered_to_on_ok_with_its_observation() -> None:
    gen, _t, hooks, _e = _rig("fine")
    offered: list[tuple[Any, str, Any]] = []
    hooks = call.Hooks(trace=hooks.trace, on_ok=lambda r, p, res: offered.append((r, p, res)))
    res = await call.generate(ROLE, prompt="say it", generator=gen, hooks=hooks)
    assert offered and offered[0][0] is ROLE and offered[0][1] == "say it"
    assert offered[0][2].stamp["generation"] is res.stamp["generation"]


class TestBlindInputs:
    """An image beside the prompt, or a web tier's fetched pages, is an input no text grader sees:
    `numbers_from_inputs` would fail every honest reply (a chart's '2027', a page's '1.47'). The
    call drops `TEXT_ONLY_GRADERS` for THAT call; the role itself is untouched."""

    async def test_an_image_call_drops_the_text_only_graders(self) -> None:
        gen, tracer, hooks, events = _rig({"summary": "resistance near 385 into 2027"})
        res = await call.generate(
            GRADED, prompt="read the chart", generator=gen, hooks=hooks, images=[(b"png", "image/png")]
        )
        assert [s.name for s in res.stamp["scores"]] == ["schema_complete", "echo"]
        assert "test.graded.numbers_from_inputs" not in {s[0] for s in tracer.generations[0].scores}
        assert events == [] and GRADED.graders == GRADERS

    async def test_a_web_call_drops_them_too(self) -> None:
        gen, _t, hooks, _e = _rig({"summary": "the 2y moved 1.47 to 1.52 on the print"})
        res = await call.generate(GRADED, prompt="what moved rates today", generator=gen, hooks=hooks, web=True)
        assert [s.name for s in res.stamp["scores"]] == ["schema_complete", "echo"]

    async def test_a_text_call_keeps_every_grader_and_still_fails_the_number_law(self) -> None:
        """BOTH DIRECTIONS: the same reply, text-only, is still caught."""
        gen, _t, hooks, events = _rig({"summary": "resistance near 385 into 2027"})
        res = await call.generate(GRADED, prompt="read the chart", generator=gen, hooks=hooks)
        assert [s.name for s in res.stamp["scores"]] == list(GRADERS)
        assert [e[2]["failed"] for e in events] == [["numbers_from_inputs"]]

    async def test_a_role_with_only_blind_graders_scores_nothing_on_an_image_call(self) -> None:
        role = Role("test.numbers_only", tier="local", graders=("numbers_from_inputs",))
        gen, tracer, hooks, _e = _rig({"summary": "385"})
        res = await call.generate(role, prompt="x", generator=gen, hooks=hooks, images=[(b"png", "image/png")])
        assert "scores" not in res.stamp and tracer.generations[0].scores == []


async def test_no_trace_hook_observes_into_the_one_null_sink() -> None:
    """The null trace is `tracer_trace(NoopTracer())`: a graded call with no hooks still scores and
    returns, and the package has ONE null sink (`reference.NoopGeneration`)."""
    from llm_ports.observability import NoopGeneration

    gen = StubGenerator({"summary": "fine"})
    res = await call.generate(GRADED, prompt="say it", generator=gen)
    assert res.outcome == "ok" and [s.name for s in res.stamp["scores"]] == list(GRADERS)
    assert isinstance(res.stamp["generation"]._gen, NoopGeneration)
