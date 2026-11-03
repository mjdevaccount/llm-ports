"""OpenTelemetry GenAI spans around a model call, exported to memory.

Needs the `otel` extra and the OpenTelemetry SDK (`pip install 'llm-ports[otel]' opentelemetry-sdk`).
The model call lands as a child span of whatever request or job span is already open, named and
attributed by the GenAI semantic conventions. A provider that reported no token counts is
flagged unmeasured rather than recorded as zero, and no prompt or completion text is ever kept.
"""

import asyncio

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_ports import Role, generate, load_tiers
from llm_ports.adapters.otel import gen_ai_span, record_gen_ai
from llm_ports.testing import StubGenerator

TABLE = load_tiers(
    {"models": {"m": "gpt-5-nano"}, "ladder": {"fast": 30.0}, "tiers": {"default": {"model": "m", "timeout": "fast"}}}
)


class AppRole(Role):
    TIERS = TABLE.tiers


TAG = AppRole("tickets.tag", tier="default")

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
tracer = provider.get_tracer("example")


async def main() -> None:
    with tracer.start_as_current_span("handle ticket"):
        with gen_ai_span(tracer, operation="chat", model=TAG.resolved_model, provider="openai") as rec:
            res = await generate(TAG, prompt="I was charged twice", generator=StubGenerator("billing"))
            rec.set_usage(res.usage.get("input") if res.usage else None, None)
        # A call that learns its latency only when it is over is recorded after the fact.
        record_gen_ai(
            tracer, operation="embeddings", model="embed-small", provider="openai", latency_ms=84, input_tokens=12
        )

    for span in exporter.get_finished_spans():
        parent = span.parent.span_id if span.parent else None
        print(span.name, "| child" if parent else "| root", dict(span.attributes or {}))


asyncio.run(main())
