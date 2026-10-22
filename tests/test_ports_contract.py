"""The ports are the seam: every adapter and double this package ships satisfies its Protocol, and
the value types keep their laws."""

from __future__ import annotations

from pydantic_ai.models.function import FunctionModel

from llm_ports import ports
from llm_ports.pydantic_ai_generator import PydanticAIGenerator
from llm_ports.testing import RecordedGeneration, RecordingTracer, StubGenerator


def test_the_shipped_generators_satisfy_the_port() -> None:
    model = FunctionModel(lambda _m, _i: None)  # type: ignore[arg-type,return-value]
    assert isinstance(PydanticAIGenerator(model), ports.Generator)
    assert isinstance(StubGenerator("x"), ports.Generator)


def test_the_recording_tracer_satisfies_the_port() -> None:
    assert isinstance(RecordingTracer(), ports.Tracer)
    assert isinstance(RecordedGeneration(opened={}), ports.Generation)


def test_cited_answer_names_ids_it_was_never_given() -> None:
    a = ports.CitedAnswer(question="q", claims=(ports.Claim("x", ("c1", "zz")),),
                          chunks={"c1": ports.Chunk(id="c1", text="t")})
    assert a.unknown_ids() == ("zz",)


async def test_a_stub_turns_a_raise_into_an_outcome_never_an_exception() -> None:
    def boom(_req, _out):
        raise ConnectionError("down")

    res = await StubGenerator(boom).generate(ports.GenRequest(model="m", prompt="p"), str)
    assert res.outcome == "unavailable" and res.value is None and res.retries is None
    assert res.reason == "ConnectionError: down"


async def test_a_stub_keeps_validator_problems_with_the_value() -> None:
    res = await StubGenerator("v").generate(ports.GenRequest(model="m", prompt="p"), str,
                                            validate=lambda v: ["bad"] if v == "v" else [])
    assert res.outcome == "ok" and res.value == "v" and res.problems == ("bad",)
