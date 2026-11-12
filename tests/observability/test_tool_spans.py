"""Tool executions as child spans (observability test 7): OTel opens `execute_tool {name}` under the
call's span, a tee fans out, a sink without the seam is skipped, and a broken one never breaks the call."""

from __future__ import annotations

from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from llm_ports import call
from llm_ports.adapters.otel import OtelTracer
from llm_ports.observability import NoopTracer, TeeTracer, observe, record_tool, semconv
from llm_ports.testing import RecordingTracer, StubChat, ToolCall, Turn
from tests.call.test_chat import TOOLS, R, _user


def _otel() -> tuple[OtelTracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OtelTracer(provider), exporter


async def test_each_tool_is_a_child_span_of_the_call() -> None:
    tracer, exporter = _otel()
    hooks = call.Hooks(trace=call.tracer_trace(tracer))
    script = [Turn.tools(ToolCall("echo", {"x": "1"}), ToolCall("boom", {"x": "2"})), Turn.reply("done")]
    res = await call.chat(
        R("c.otel", tier="chat"), messages=[_user("q")], chat=StubChat(script), tools=TOOLS, hooks=hooks
    )
    assert res.outcome == "ok"
    spans = {s.name: s for s in exporter.get_finished_spans()}
    parent = next(s for s in spans.values() if not s.name.startswith("execute_tool"))
    echo, boom = spans["execute_tool echo"], spans["execute_tool boom"]
    for s in (echo, boom):
        assert s.kind == SpanKind.INTERNAL and s.parent is not None and s.parent.span_id == parent.context.span_id
        assert s.attributes is not None and s.attributes[semconv.OPERATION_NAME] == "execute_tool"
    assert boom.status.status_code == StatusCode.ERROR and boom.attributes[semconv.ERROR_TYPE] == "tool_error"  # type: ignore[index]
    assert echo.attributes[semconv.TOOL_CALL_ID] and semconv.ERROR_TYPE not in echo.attributes  # type: ignore[index,operator]


async def test_a_tee_fans_tools_out_and_skips_a_sink_without_the_seam() -> None:
    rec = RecordingTracer()
    tee = TeeTracer(rec, NoopTracer())
    async with observe(tee, name="n", model="m", provider="p", input="i", metadata={}) as obs:
        obs.tool(name="t", call_id="c", duration_s=0.5, is_error=False)  # type: ignore[attr-defined]
    assert rec.generations[0].tools == [{"name": "t", "call_id": "c", "duration_s": 0.5, "is_error": False}]


def test_record_tool_ignores_a_missing_seam_and_swallows_a_broken_one() -> None:
    class Bare:
        pass

    class Broken:
        def tool(self, **_kw: Any) -> None:
            raise RuntimeError("sink down")

    record_tool(Bare(), name="t", call_id="c", duration_s=0.0, is_error=False)  # type: ignore[arg-type]
    record_tool(Broken(), name="t", call_id="c", duration_s=0.0, is_error=True)  # type: ignore[arg-type]
