"""otel_gen_ai -- GenAI spans: convention names, no 0 ms after-the-fact calls, none is not zero,
tracing off is a no-op. A recording fake stands in for the SDK; one test reads a real SDK span."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from llm_ports.adapters import otel as ga


class _Span:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}
        self.status: Any = None
        self.ended_at: int | None = None

    def set_attribute(self, k: str, v: Any) -> None:
        self.attrs[k] = v

    def set_status(self, status: Any) -> None:
        self.status = status

    def end(self, end_time: int | None = None) -> None:
        self.ended_at = end_time

    def __enter__(self) -> _Span:
        return self

    def __exit__(self, *_a: Any) -> bool:
        return False


class _Tracer:
    def __init__(self) -> None:
        self.spans: list[_Span] = []
        self.names: list[str] = []
        self.starts: list[int | None] = []

    def start_as_current_span(self, name: str, attributes: dict[str, Any] | None = None) -> _Span:
        return self._make(name, attributes, None)

    def start_span(self, name: str, start_time: int | None = None, attributes: dict[str, Any] | None = None) -> _Span:
        return self._make(name, attributes, start_time)

    def _make(self, name: str, attributes: dict[str, Any] | None, start: int | None) -> _Span:
        sp = _Span()
        sp.attrs.update(attributes or {})
        self.spans.append(sp)
        self.names.append(name)
        self.starts.append(start)
        return sp


@pytest.fixture
def tracer() -> _Tracer:
    return _Tracer()


def test_the_attribute_names_are_the_conventions() -> None:
    assert (ga.SYSTEM, ga.PROVIDER_NAME, ga.OPERATION, ga.REQUEST_MODEL, ga.INPUT_TOKENS, ga.OUTPUT_TOKENS) == (
        "gen_ai.system",
        "gen_ai.provider.name",
        "gen_ai.operation.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    )


def test_a_live_span_is_named_operation_model_and_carries_the_consumer_dimensions(tracer) -> None:
    with ga.gen_ai_span(
        tracer, operation="chat", model="m", provider="p", attributes={"app.feature": "f", "app.none": None}
    ) as rec:
        rec.set_usage(11, 7)
    (sp,) = tracer.spans
    assert tracer.names == ["chat m"]
    assert sp.attrs == {
        ga.SYSTEM: "p",
        ga.PROVIDER_NAME: "p",
        ga.OPERATION: "chat",
        ga.REQUEST_MODEL: "m",
        "app.feature": "f",
        ga.INPUT_TOKENS: 11,
        ga.OUTPUT_TOKENS: 7,
    }


def test_a_caller_exception_propagates_out_of_the_span(tracer) -> None:
    # MUST FIRE: a try/except around the `yield` swallows this behind "generator didn't stop after throw()".
    with (
        pytest.raises(RuntimeError, match="upstream 500"),
        ga.gen_ai_span(tracer, operation="chat", model="m", provider="p"),
    ):
        raise RuntimeError("upstream 500")


def test_the_after_the_fact_span_is_back_dated_by_the_measured_latency(tracer) -> None:
    # MUST FIRE: drop `start_time` and the span says 0 ms for a call that took 1500.
    ga.record_gen_ai(tracer, operation="embeddings", model="e5", provider="box", latency_ms=1500)
    sp = tracer.spans[0]
    assert sp.ended_at is not None and tracer.starts[0] is not None
    assert 1499 <= (sp.ended_at - tracer.starts[0]) / 1_000_000 <= 1501


def test_a_reported_error_is_recorded_as_one(tracer) -> None:
    ga.record_gen_ai(tracer, operation="chat", model="m", provider="p", latency_ms=10, error="ConnectionError: refused")
    assert tracer.spans[0].status is not None and tracer.spans[0].status.status_code == StatusCode.ERROR


def test_unreported_tokens_are_marked_unmeasured_not_zero(tracer) -> None:
    ga.record_gen_ai(tracer, operation="chat", model="m", provider="p", latency_ms=5, unmeasured_attr="app.unmeasured")
    attrs = tracer.spans[0].attrs
    assert attrs.get("app.unmeasured") is True and ga.INPUT_TOKENS not in attrs


def test_a_real_zero_is_kept_as_a_number(tracer) -> None:
    ga.record_gen_ai(tracer, operation="chat", model="m", provider="p", latency_ms=5, input_tokens=12, output_tokens=0)
    attrs = tracer.spans[0].attrs
    assert attrs[ga.OUTPUT_TOKENS] == 0 and ga.UNMEASURED not in attrs


def test_no_tracer_means_no_span_and_no_crash() -> None:
    with ga.gen_ai_span(None, operation="chat", model="m", provider="p") as rec:
        rec.set_usage(1, 2)
    ga.record_gen_ai(None, operation="chat", model="m", provider="p", latency_ms=1)


def test_a_tracer_that_cannot_open_a_span_yields_a_no_op() -> None:
    class Broken:
        def start_as_current_span(self, *a, **k):
            raise RuntimeError("no provider")

        def start_span(self, *a, **k):
            raise RuntimeError("no provider")

    with ga.gen_ai_span(Broken(), operation="chat", model="m", provider="p") as rec:
        rec.set_usage(1, 2)
    ga.record_gen_ai(Broken(), operation="chat", model="m", provider="p", latency_ms=1)


def test_a_real_sdk_span_is_a_child_of_the_ambient_span() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t = provider.get_tracer("t")
    with t.start_as_current_span("job") as job, ga.gen_ai_span(t, operation="chat", model="m", provider="p"):
        pass
    chat = next(s for s in exporter.get_finished_spans() if s.name == "chat m")
    assert chat.parent is not None and chat.parent.span_id == job.get_span_context().span_id
