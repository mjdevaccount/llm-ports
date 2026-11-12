"""1.24 observability: one end on every exit (cancel included), one payload for every path, the tee,
`record`, the pinned semconv names, and no `gen_ai.system` anywhere."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from llm_ports import call, ports
from llm_ports.observability import SINK_FAILED, NoopTracer, TeeTracer, observe, record, semconv
from llm_ports.observability._context import _feature_var
from llm_ports.observability._payload import deliver, end_payload, legacy_usage
from llm_ports.testing import FakeClock, RecordingTracer, Reply, StubGenerator
from tests.conftest import Role

ROLE = Role("test.trace", tier="local")
SRC = Path(__file__).resolve().parents[2] / "src"


# -- 1: ended exactly once on every exit -----------------------------------------------------------


async def test_a_cancelled_block_ends_once_as_cancelled_with_latency_and_the_cancel_propagates() -> None:
    t = RecordingTracer()
    now = iter([1.0, 1.5])
    with pytest.raises(asyncio.CancelledError):
        async with observe(t, name="c", model="m", provider="p", input="", clock=lambda: next(now)) as obs:
            obs.update(usage_details={"input": 4, "output": None})
            raise asyncio.CancelledError
    (g,) = t.generations
    assert g.ends == 1
    assert g.ended is not None and g.ended["level"] == "CANCELLED"
    assert g.ended["status_message"] == "CancelledError"
    assert g.ended["metadata"]["latency_ms"] == 500 and g.ended["usage"] == {"input": 4, "output": None}


async def test_a_cancelled_generate_ends_its_generation_once_as_cancelled() -> None:
    started = asyncio.Event()

    class _Hangs:
        async def generate(self, req: Any, output: Any, **kw: Any) -> Any:
            started.set()
            await asyncio.sleep(3600)

    t = RecordingTracer()
    task = asyncio.create_task(
        call.generate(ROLE, prompt="p", output=str, generator=_Hangs(), hooks=call.Hooks(trace=call.tracer_trace(t)))
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (g,) = t.generations
    assert g.ends == 1 and g.ended is not None and g.ended["level"] == "CANCELLED"


async def test_both_ways_a_clean_exit_ends_once_at_the_default_level() -> None:
    t = RecordingTracer()
    async with observe(t, name="c", model="m", provider="p", input=""):
        pass
    (g,) = t.generations
    assert g.ends == 1 and g.ended is not None and g.ended["level"] == "DEFAULT"


async def test_a_keyboard_interrupt_also_ends_once_as_cancelled() -> None:
    t = RecordingTracer()
    with pytest.raises(KeyboardInterrupt):
        async with observe(t, name="c", model="m", provider="p", input=""):
            raise KeyboardInterrupt
    assert t.generations[0].ends == 1 and t.generations[0].ended["level"] == "CANCELLED"  # type: ignore[index]


# -- 2: generate and record send the same payload --------------------------------------------------


async def test_generate_and_record_send_one_golden_payload() -> None:
    usage = ports.Usage(input=5, output=2, requests=1)
    gen = StubGenerator([Reply.ok("done", usage=usage, answered_model="served")])
    t = RecordingTracer()
    res = await call.generate(ROLE, prompt="p", output=str, generator=gen, hooks=call.Hooks(trace=call.tracer_trace(t)))
    assert res.outcome == "ok"
    from_generate = t.generations[0].ended
    assert from_generate is not None
    r = RecordingTracer()
    record(
        r,
        name="demo.cat",
        model=res.model,
        provider="p",
        input="p",
        output=from_generate["output"],
        started_at=0.0,
        ended_at=-1.0,  # before the start: latency omitted
        usage=res.usage_v2,
        cost=res.cost,
        finish=res.finish,
        answered=res.answered,
        failure=res.failure,
        metadata=from_generate["metadata"],
    )
    assert r.generations[0].ended == from_generate


def test_record_with_no_usage_is_unmeasured_and_never_zero() -> None:
    t = RecordingTracer()
    record(t, name="n", model="m", provider="p", input="i", output="o", started_at=1.0, ended_at=1.25)
    end = t.generations[0].ended
    assert end is not None and end["usage"] is None
    assert end["metadata"] == {"latency_ms": 250, "tokens_unmeasured": True}
    assert end["usage_v2"] == ports.Usage(requests=1, unmeasured=1)


def test_record_never_raises_over_a_tracer_that_does() -> None:
    class _Boom:
        def start_generation(self, **kw: Any) -> Any:
            raise RuntimeError("down")

    record(_Boom(), name="n", model="m", provider="p", input="i", output="o", started_at=0, ended_at=1)  # type: ignore[arg-type]


def test_a_failure_derives_the_error_level_and_its_status() -> None:
    p = end_payload(failure=ports.failure("rate_limited", reason="429 from upstream"))
    assert p["level"] == "ERROR" and p["status_message"] == "rate_limited: 429 from upstream"
    assert end_payload()["metadata"] is None and "level" not in end_payload()
    assert legacy_usage(ports.Usage(requests=1)) is None


def test_a_1x_generation_gets_the_five_keywords_only() -> None:
    got: list[dict[str, Any]] = []

    class _Old:
        def end(self, *, output=None, usage=None, level="DEFAULT", status_message=None, metadata=None) -> None:
            got.append({"output": output, "usage": usage, "level": level})

        def score(self, *a: Any) -> None: ...

    deliver(_Old(), end_payload(output="o", cost=ports.Cost(Decimal(1), "provider")))  # type: ignore[arg-type]
    assert got == [{"output": "o", "usage": None, "level": "DEFAULT"}]


def test_deliver_never_raises_over_a_broken_end() -> None:
    class _Broken:
        def end(self, **kw: Any) -> None:
            raise RuntimeError("sink down")

    deliver(_Broken(), end_payload(output="o"))  # type: ignore[arg-type]


# -- 6: the tee isolates sinks ---------------------------------------------------------------------


class _Raises:
    def __init__(self, where: str) -> None:
        self.where = where

    def start_generation(self, **kw: Any) -> Any:
        if self.where == "start":
            raise RuntimeError("cannot open")
        return self

    def end(self, **kw: Any) -> None:
        raise RuntimeError("cannot end")

    def score(self, *a: Any) -> None:
        raise RuntimeError("cannot score")


def _open(t: TeeTracer) -> Any:
    return t.start_generation(
        name="n", model="m", provider="p", input="i", metadata={}, session_id=None, prompt_name=None
    )


def test_a_sink_raising_on_end_never_stops_the_other_and_is_counted_with_one_event() -> None:
    events: list[tuple[int, str, dict[str, Any]]] = []
    good = RecordingTracer()
    clock = FakeClock()
    tee = TeeTracer(_Raises("end"), good, on_event=lambda lvl, ev, f: events.append((lvl, ev, dict(f))), clock=clock)  # type: ignore[arg-type]
    for _ in range(2):
        g = _open(tee)
        g.end(output="o", usage=None, metadata=None)
        g.score("s", 1.0)
    assert [e.ended for e in good.generations] == [
        {"output": "o", "usage": None, "level": "DEFAULT", "status_message": None, "metadata": None}
    ] * 2
    assert tee.stats() == {
        "0:_Raises": {"started": 2, "ended": 0, "failed": 4},
        "1:RecordingTracer": {"started": 2, "ended": 2, "failed": 0},
    }
    assert [(lvl, ev, f["sink"], f["action"]) for lvl, ev, f in events] == [
        (logging.WARNING, SINK_FAILED, "0:_Raises", "end")
    ], "throttled: one event per sink per hold"
    clock.advance(301)
    _open(tee).end(output="o")
    assert len(events) == 2


def test_a_sink_failing_to_open_gets_a_noop_and_the_others_open() -> None:
    good = RecordingTracer()
    tee = TeeTracer(_Raises("start"), good, on_event=lambda *_: None)  # type: ignore[arg-type]
    _open(tee).end(output="o")
    assert tee.stats()["0:_Raises"] == {"started": 0, "ended": 0, "failed": 1}
    assert good.generations[0].ends == 1


def test_a_tee_of_noop_tracers_is_a_tracer() -> None:
    assert isinstance(TeeTracer(NoopTracer(), NoopTracer()), ports.Tracer)


# -- 4, 8, 10: names are pinned, provider-neutral, and `gen_ai.system` is gone ---------------------


def test_the_pinned_names_are_one_set_and_disjoint_from_the_bodies() -> None:
    assert semconv.SEMCONV_SOURCE.startswith("open-telemetry/semantic-conventions-genai@")
    assert not semconv.BODY_ATTRIBUTES & (semconv.CONVENTION_ATTRIBUTES | semconv.LIBRARY_ATTRIBUTES)
    assert all(k.startswith("llm_ports.") for k in semconv.LIBRARY_ATTRIBUTES)
    assert all(m.startswith("gen_ai.client.") for m in semconv.METRICS)


def test_gen_ai_system_is_emitted_nowhere_in_the_source() -> None:
    # MUST FIRE: the conventions removed it at v1.37.0; a sink that still writes it splits every query.
    hits = [p for p in SRC.rglob("*.py") if '"gen_ai.system"' in p.read_text(encoding="utf-8")]
    assert hits == []


def test_the_feature_context_variable_is_provider_neutral() -> None:
    assert _feature_var.name == "llm_feature"


async def test_the_trace_provider_comes_from_the_tier_not_a_vendor_default() -> None:
    t = RecordingTracer()
    await call.generate(
        ROLE, prompt="p", output=str, generator=StubGenerator("x"), hooks=call.Hooks(trace=call.tracer_trace(t))
    )
    # A schema 1 table names no provider: "unknown", never a vendor guessed for it (1.x said "openai").
    assert ROLE.resolved_tier.info is None and t.generations[0].opened["provider"] == "unknown"
    t2 = RecordingTracer()
    await call.generate(
        ROLE,
        prompt="p",
        output=str,
        generator=StubGenerator("x"),
        hooks=call.Hooks(trace=call.tracer_trace(t2, provider="local")),
    )
    assert t2.generations[0].opened["provider"] == "local"
