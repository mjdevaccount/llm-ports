"""OtelTracer over the real SDK with in-memory exporters: span name and kind, the exact attribute key
sets (ok, failed, cache hit, unmeasured), no bodies whatever the environment says, the metrics, and
never raising."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from llm_ports import ports
from llm_ports.adapters.otel import OtelTracer
from llm_ports.observability import TeeTracer, observe, semconv
from llm_ports.observability._payload import end_payload
from llm_ports.testing import FakeClock, RecordingTracer

OPEN = {semconv.OPERATION_NAME, semconv.PROVIDER_NAME, semconv.REQUEST_MODEL}


class _Rig:
    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.reader = InMemoryMetricReader()
        self.meters = MeterProvider(metric_readers=[self.reader])
        self.clock = FakeClock(10.0)
        self.tracer = OtelTracer(self.provider, self.meters, clock=self.clock)

    def open(self, **meta: Any) -> Any:
        return self.tracer.start_generation(
            name="demo.cat", model="m-req", provider="local", input="SECRET PROMPT", metadata=meta,
            session_id=None, prompt_name=None,
        )  # fmt: skip

    def span(self) -> Any:
        (s,) = self.exporter.get_finished_spans()
        return s

    def metrics(self) -> dict[str, list[Any]]:
        data = self.reader.get_metrics_data()
        out: dict[str, list[Any]] = {}
        for rm in data.resource_metrics if data else ():
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    out[m.name] = list(m.data.data_points)
        return out


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


def _ok(**over: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "output": "SECRET OUTPUT",
        "usage": ports.Usage(input=12, output=5, requests=1),
        "cost": ports.Cost(Decimal("0.002"), "price_table"),
        "finish": ports.Finish("stop", "stop"),
        "answered": ports.Answered(requested="m-req", model="m-resp", tier="local", response_id="r-1"),
        "metadata": {"outcome": "ok", "attempts": 0},
    }
    kw.update(over)
    return end_payload(**kw)


def test_an_ok_call_is_one_client_span_with_the_exact_attribute_keys(rig: _Rig) -> None:
    rig.open(role="test.role").end(**_ok())
    s = rig.span()
    assert s.name == "chat m-req" and s.kind is SpanKind.CLIENT
    assert set(s.attributes) == OPEN | {
        semconv.ROLE,
        semconv.USAGE_INPUT,
        semconv.USAGE_OUTPUT,
        semconv.RESPONSE_MODEL,
        semconv.RESPONSE_ID,
        semconv.TIER,
        semconv.FINISH_REASONS,
        semconv.COST_SOURCE,
        semconv.COST_USD,
        semconv.OUTCOME,
        semconv.ATTEMPTS,
    }
    assert s.attributes[semconv.USAGE_INPUT] == 12 and s.attributes[semconv.COST_USD] == pytest.approx(0.002)
    assert s.status.status_code is StatusCode.UNSET


def test_a_failed_call_sets_error_type_to_the_kind_and_an_error_status(rig: _Rig) -> None:
    rig.open().end(**end_payload(failure=ports.failure("rate_limited", reason="429")))
    s = rig.span()
    assert s.attributes[semconv.ERROR_TYPE] == "rate_limited" and s.status.status_code is StatusCode.ERROR
    assert set(s.attributes) == OPEN | {semconv.ERROR_TYPE, semconv.USAGE_UNMEASURED}


def test_a_cache_hit_carries_the_cache_read_count(rig: _Rig) -> None:
    rig.open().end(**_ok(usage=ports.Usage(input=100, output=3, cache_read=80, requests=1)))
    assert rig.span().attributes[semconv.USAGE_CACHE_READ] == 80


def test_an_unmeasured_call_has_no_usage_keys_and_the_flag(rig: _Rig) -> None:
    # MUST FIRE: write None as 0 and a dashboard sum understates silently.
    rig.open().end(**_ok(usage=ports.Usage(requests=1, unmeasured=1)))
    a = rig.span().attributes
    assert not [k for k in a if k.startswith("gen_ai.usage.")] and a[semconv.USAGE_UNMEASURED] is True


def test_no_body_is_ever_carried_whatever_the_environment_says(rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", "true")
    rig.open().end(**_ok())
    a = rig.span().attributes
    assert not set(a) & semconv.BODY_ATTRIBUTES
    assert "SECRET" not in repr(dict(a))


def test_every_emitted_key_is_a_pinned_constant(rig: _Rig) -> None:
    rig.open(role="r", pack="p").end(**_ok(usage=ports.Usage(1, 1, 1, 1, 1, requests=1)))
    rig.open().end(**end_payload(level="CANCELLED", status_message="CancelledError"))
    keys = {k for s in rig.exporter.get_finished_spans() for k in s.attributes}
    assert keys <= semconv.CONVENTION_ATTRIBUTES | semconv.LIBRARY_ATTRIBUTES


def test_the_meter_records_the_duration_and_one_counter_per_measured_kind(rig: _Rig) -> None:
    g = rig.open()
    rig.clock.advance(1.5)
    g.end(**_ok())
    m = rig.metrics()
    (d,) = m[semconv.METRIC_INFERENCE_DURATION]
    assert d.sum == pytest.approx(1.5) and d.attributes[semconv.RESPONSE_MODEL] == "m-resp"
    assert m[semconv.METRIC_USAGE_INPUT][0].value == 12 and m[semconv.METRIC_USAGE_OUTPUT][0].value == 5
    assert m[semconv.METRIC_USAGE_INPUT][0].attributes[semconv.TOKEN_MODALITY] == "unknown"
    assert semconv.METRIC_USAGE_CACHE_READ not in m, "an unmeasured kind is never counted"


def test_errors_and_cancels_carry_error_type_on_the_duration(rig: _Rig) -> None:
    rig.open().end(**end_payload(level="CANCELLED"))
    rig.open().end(**end_payload(level="ERROR", status_message="boom"))
    spans = rig.exporter.get_finished_spans()
    assert [s.attributes[semconv.ERROR_TYPE] for s in spans] == ["cancelled", "_OTHER"]
    durations = rig.metrics()[semconv.METRIC_INFERENCE_DURATION]
    assert sorted(p.attributes[semconv.ERROR_TYPE] for p in durations) == ["_OTHER", "cancelled"]


def test_end_is_once_and_score_is_a_no_op(rig: _Rig) -> None:
    g = rig.open()
    g.end(**_ok())
    g.end(**_ok())
    g.score("s", 1.0)
    assert len(rig.exporter.get_finished_spans()) == 1


def test_a_tracer_that_cannot_open_a_span_gives_a_noop_and_never_raises() -> None:
    class _Broken:
        def get_tracer(self, *_a: Any) -> Any:
            return self

        def start_span(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("sdk down")

    from llm_ports.observability import NoopGeneration

    g = OtelTracer(_Broken()).start_generation(
        name="n", model="m", provider="p", input="", metadata={}, session_id="s", prompt_name=None
    )
    assert isinstance(g, NoopGeneration)


def test_only_the_pinned_semconv_exists() -> None:
    with pytest.raises(ValueError, match="pinned"):
        OtelTracer(semconv="head")  # type: ignore[arg-type]


async def test_a_tee_of_otel_and_a_recorder_both_see_one_cancelled_call(rig: _Rig) -> None:
    rec = RecordingTracer()
    with pytest.raises(asyncio.CancelledError):
        async with observe(TeeTracer(rig.tracer, rec), name="c", model="m", provider="p", input="", session_id="s"):
            raise asyncio.CancelledError
    assert rig.span().attributes[semconv.ERROR_TYPE] == "cancelled"
    assert rig.span().attributes[semconv.CONVERSATION_ID] == "s"
    assert rec.generations[0].ends == 1
