"""NoopTracer is a Tracer whose generation swallows everything."""

from __future__ import annotations

from llm_ports import ports
from llm_ports.observability import NoopGeneration, NoopTracer


def test_the_noop_tracer_is_a_tracer_whose_generation_swallows_everything() -> None:
    t = NoopTracer()
    assert isinstance(t, ports.Tracer)
    g = t.start_generation(name="n", model="m", provider="p", input="i", metadata={}, session_id=None, prompt_name=None)
    assert isinstance(g, NoopGeneration) and isinstance(g, ports.Generation)
    assert g.end(output="o", usage=None, level="ERROR", status_message="x") is None
    assert g.score("s", 1.0, "c") is None
