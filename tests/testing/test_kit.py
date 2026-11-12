"""The testing kit: scripted `StubGenerator`, `FakeClock`, and the port conformance suites -- each
proven to pass for the reference implementations and the stubs, and to FIRE on one mutant per
property (a suite that never fails proves nothing)."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import replace
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from llm_ports import ports
from llm_ports.adapters import litellm_proxy
from llm_ports.adapters.genai_prices import GenaiPricesPricer
from llm_ports.adapters.http import HttpProbe
from llm_ports.adapters.otel import OtelTracer
from llm_ports.adapters.pydantic_ai import PydanticAIGenerator
from llm_ports.call.resilience import LocalLimiter
from llm_ports.evals import GauntletEvaluator
from llm_ports.grounding import NumberLawGrounder
from llm_ports.observability import NoopTracer, TeeTracer
from llm_ports.ports import Failure, Generated, GenRequest
from llm_ports.testing import (
    ContractViolation,
    FakeClock,
    RecordedGeneration,
    RecordingTracer,
    Reply,
    ScriptedReply,
    StubCiter,
    StubEmbedder,
    StubGenerator,
    StubPromptSource,
    StubReranker,
)
from llm_ports.testing import conformance as cf

REQ = GenRequest(model="m", prompt="p", output_retries=1)


# -- the scripted stub -----------------------------------------------------------------------------


async def test_problems_then_ok_is_one_re_ask_with_summed_usage() -> None:
    gen = StubGenerator(
        [Reply.problems("draft", ["too short"], usage=ports.Usage(2, 1, requests=1)), Reply.ok("final")]
    )
    res = await gen.generate(REQ, str)
    assert res.outcome == "ok" and res.value == "final" and res.retries == 1 and res.problems == ()
    assert res.usage_v2 == ports.Usage(2, 1, requests=1) and gen.requests == [REQ]
    assert res.finish == ports.Finish("stop", "stop") and res.answered == ports.Answered("m", "stub-generator")


async def test_problems_left_after_the_last_re_ask_keep_the_value() -> None:
    res = await StubGenerator([Reply.problems("a", ["x"]), Reply.problems("b", ["y"])]).generate(REQ, str)
    assert res.outcome == "ok" and res.value == "b" and res.problems == ("y",) and res.retries == 1


async def test_an_exhausted_script_is_a_contract_violation_not_a_default_reply() -> None:
    gen = StubGenerator([Reply.ok("one")])
    await gen.generate(REQ, str)
    with pytest.raises(ContractViolation) as e:
        await gen.generate(REQ, str)
    assert e.value.invariant == "script_exhausted"


def test_unused_replies_fail_assert_spent_and_both_ways_a_spent_script_passes() -> None:
    gen = StubGenerator([Reply.ok("one"), Reply.failed("server")])
    assert len(gen.unused) == 2
    with pytest.raises(ContractViolation, match="script_unused"):
        gen.assert_spent()
    StubGenerator("constant").assert_spent()


def test_a_plain_list_value_is_a_constant_not_a_script() -> None:
    assert StubGenerator(["a", "b"]).unused == () and StubGenerator([]).unused == ()


async def test_a_failure_in_a_script_is_that_kind_and_a_raising_validator_is_internal() -> None:
    res = await StubGenerator([Reply.failed("rate_limited", retry_after_s=2.0)]).generate(REQ, str)
    assert res.outcome == "unavailable" and res.failure is not None and res.failure.retry_after_s == 2.0

    def bug(_v: str) -> list[str]:
        raise ValueError("bug")

    res = await StubGenerator("x").generate(REQ, str, validate=bug)
    assert res.failure is not None and res.failure.kind == "internal"


async def test_a_raising_reply_function_is_a_server_failure() -> None:
    def boom(_r: Any, _o: Any) -> Any:
        raise RuntimeError("down")

    res = await StubGenerator(boom).generate(REQ, str)
    assert res.failure is not None and res.failure.kind == "server"


async def test_the_fixture_fails_teardown_on_unused_replies(stub_generator: Callable[..., StubGenerator]) -> None:
    gen = stub_generator([Reply.ok("x")])
    assert (await gen.generate(REQ, str)).value == "x"


def test_fake_clock_moves_only_when_told() -> None:
    c = FakeClock(5.0)
    assert c() == 5.0
    c.advance(1.5)
    assert c() == 6.5
    with pytest.raises(ValueError, match="never goes back"):
        c.advance(-1)


async def test_fake_clock_sleep_records_and_advances_without_waiting(fake_clock: FakeClock) -> None:
    await fake_clock.sleep(2.0)
    await fake_clock.sleep(-1.0)
    assert fake_clock() == 2.0 and fake_clock.sleeps == [2.0, -1.0]


# -- check_generator: the stub and the reference adapter both pass ---------------------------------


def _pydantic_ai_from(script: Sequence[Any]) -> PydanticAIGenerator:
    """The reference adapter over a FunctionModel answering each request from the next reply."""
    replies = list(script)

    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        r = replies.pop(0)
        if isinstance(r, Failure):
            raise ModelHTTPError(status_code=400, model_name="scripted", body={"error": r.reason})
        u = r.usage
        usage = (
            RequestUsage(input_tokens=u.input or 0, output_tokens=u.output or 0) if u is not None else RequestUsage()
        )
        return ModelResponse(parts=[TextPart(str(r.value))], usage=usage)

    return PydanticAIGenerator(FunctionModel(fn), policy=ports.WirePolicy.none())


async def test_check_generator_passes_for_the_stub() -> None:
    cases = await cf.check_generator(StubGenerator)
    assert "reask_counts_retries" in cases and "callback_internal" in cases


async def test_check_generator_passes_for_the_pydantic_ai_adapter() -> None:
    assert await cf.check_generator(_pydantic_ai_from) == await cf.check_generator(StubGenerator)


def _if(cond: Callable[[Generated[Any]], bool], change: Callable[[Generated[Any]], Generated[Any]]) -> Any:
    return lambda r: change(r) if cond(r) else r


MUTANTS: dict[str, Callable[[Generated[Any]], Generated[Any]]] = {
    "ok_value": _if(lambda r: r.value == "alpha", lambda r: replace(r, value="beta")),
    "answered_requested": _if(
        lambda r: r.answered is not None, lambda r: replace(r, answered=ports.Answered("other", "x"))
    ),
    "failure_never_raises": _if(
        lambda r: r.failure is not None and r.failure.kind == "bad_request",
        lambda r: replace(r, failure=ports.failure("server")),
    ),
    "reask_counts_retries": lambda r: replace(r, retries=0),
    "usage_sums_reasks": _if(
        lambda r: r.retries == 1 and r.usage_v2 is not None,
        lambda r: replace(r, usage_v2=ports.Usage(4, 1, requests=1)),
    ),
    "last_problems_kept": _if(lambda r: r.value == "draft", lambda r: replace(r, problems=())),
    "callback_internal": _if(
        lambda r: r.failure is not None and r.failure.kind == "internal",
        lambda r: replace(r, failure=ports.failure("server")),
    ),
}


class _Mutant(StubGenerator):
    def __init__(self, script: Any, mutate: Callable[[Generated[Any]], Generated[Any]]) -> None:
        super().__init__(script)
        self._mutate = mutate

    async def generate(self, req: Any, output: Any, **kw: Any) -> Any:
        return self._mutate(await super().generate(req, output, **kw))


@pytest.mark.parametrize("case", sorted(MUTANTS))
async def test_check_generator_fires_on_one_mutant_per_property(case: str) -> None:
    with pytest.raises(ContractViolation) as e:
        await cf.check_generator(lambda s: _Mutant(s, MUTANTS[case]))
    assert e.value.invariant == case


async def test_check_generator_fires_on_a_raise_and_on_a_non_generator() -> None:
    class _Raises(StubGenerator):
        async def generate(self, req: Any, output: Any, **kw: Any) -> Any:
            raise RuntimeError("model down")

    with pytest.raises(ContractViolation, match="ok_value: generate raised RuntimeError"):
        await cf.check_generator(_Raises)
    with pytest.raises(ContractViolation) as e:
        await cf.check_generator(lambda _s: object())  # type: ignore[arg-type,return-value]
    assert e.value.invariant == "is_generator"


# -- every other port: the reference implementations pass ------------------------------------------


class _Retriever:
    async def search(self, q: str, *, k: int = 8, date_from: Any = None, date_to: Any = None) -> list[ports.Chunk]:
        return [] if not q.strip() else [ports.Chunk(id=str(i), text=q) for i in range(k)]


class _Judge:
    async def judge(self, *, prompt: str, output: Any, rubric: str) -> ports.JudgeVerdict:
        return ports.JudgeVerdict(passed=True, score=1.0)


def _models(_r: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": "m"}]})


def _otel() -> OtelTracer:
    from opentelemetry.sdk.trace import TracerProvider

    return OtelTracer(TracerProvider())


def _langfuse() -> Any:
    from llm_ports.adapters.langfuse.tracer import LangfuseTracer
    from llm_ports.testing.langfuse import LangfuseRig

    return LangfuseTracer(LangfuseRig().client)


PASSING: list[tuple[str, Callable[[], Any], tuple[Any, ...]]] = [
    ("Embedder", StubEmbedder, ()),
    ("Reranker", StubReranker, ()),
    ("Tracer", RecordingTracer, ()),
    ("Tracer", NoopTracer, ()),
    ("Tracer", lambda: TeeTracer(RecordingTracer(), NoopTracer()), ()),
    ("Tracer", _otel, ()),
    ("Tracer", _langfuse, ()),
    ("Generation", lambda: RecordedGeneration(opened={}), ()),
    ("Pricer", GenaiPricesPricer, ()),
    ("Probe", lambda: HttpProbe("http://gateway.test/v1/models", transport=httpx.MockTransport(_models)), ()),
    ("Retriever", _Retriever, ()),
    ("Citer", StubCiter, ()),
    ("Grounder", NumberLawGrounder, ()),
    ("Evaluator", GauntletEvaluator, ()),
    ("PromptSource", lambda: StubPromptSource({"greet": "hello {name}"}), ("greet",)),
    ("Judge", _Judge, ()),
    ("GatewayReader", lambda: litellm_proxy, ()),
    ("Limiter", lambda: LocalLimiter({}, default=1), ()),
]


@pytest.mark.parametrize(("port", "make", "args"), PASSING, ids=[f"{p}-{i}" for i, (p, _, _) in enumerate(PASSING)])
async def test_the_reference_implementations_pass_their_port_check(port: str, make: Any, args: tuple[Any, ...]) -> None:
    assert await cf.CHECKS[port](make, *args)


def test_checks_cover_every_runtime_checkable_port() -> None:
    from llm_ports import learn

    ports_ = {
        n
        for mod in (ports, learn)
        for n in getattr(mod, "__all__", ())
        if inspect.isclass(getattr(mod, n)) and getattr(getattr(mod, n), "_is_runtime_protocol", False)
    }
    assert ports_ - set(cf.CHECKS) == set(), "a new port has no conformance check"
    assert len(ports_) >= 15


class _Bad:
    """Every port's method, each broken the way its check is meant to catch."""

    stamp = ""

    async def embed(self, texts: Any, *, input_type: Any = "document") -> ports.Embedded:
        return ports.Embedded(vectors=((1.0,),), outcome="ok", stamp="")

    async def rerank(self, query: str, documents: Any) -> ports.Reranked:
        return ports.Reranked(scores=(1.0,), outcome="ok", model="")

    def price(self, usage: Any, answered: Any) -> ports.Cost:
        return ports.Cost(Decimal("0.01"), "price_table")

    async def probe(self) -> Any:
        return ports.Health(state="maybe")  # type: ignore[arg-type]

    async def search(self, q: str, *, k: int = 8, date_from: Any = None, date_to: Any = None) -> list[Any]:
        return [ports.Chunk(id="x", text="x")] * (k + 1)

    async def answer(self, q: str, chunks: Any) -> ports.CitedAnswer:
        return ports.CitedAnswer(question=q, claims=(ports.Claim("c", ("nope",)),), chunks={})

    def verdict(self, claim: Any, spans: Any) -> ports.GroundingVerdict:
        return ports.GroundingVerdict(True)

    def run(self, surface: str, rows: Any, graders: Any, **kw: Any) -> dict[str, Any]:
        return {"surface": surface}

    def get(self, name: str, *, label: str = "production") -> ports.Prompt:
        return ports.Prompt(name, "")

    async def judge(self, **kw: Any) -> Any:
        return "yes"

    def cost_from_headers(self, headers: Any) -> Any:
        return Decimal(0)

    def answered_from_headers(self, headers: Any) -> Any:
        return (None, None)

    def slot(self, group: str) -> Any:
        raise RuntimeError("no slots")

    def start_generation(self, **kw: Any) -> Any:
        raise RuntimeError("cannot open")

    def end(self, **kw: Any) -> None:
        raise RuntimeError("cannot end")

    def score(self, *a: Any) -> None: ...


@pytest.mark.parametrize(
    ("port", "case"),
    [
        ("Embedder", "stamp_constant"),
        ("Reranker", "one_score_each"),
        ("Pricer", "unknown_is_none"),
        ("Probe", "known_state"),
        ("Retriever", "blank_is_empty"),
        ("Citer", "no_chunks_no_claims"),
        ("Grounder", "no_spans_unsupported"),
        ("Evaluator", "board_shape"),
        ("PromptSource", "text_not_empty"),
        ("Judge", "verdict_shape"),
        ("GatewayReader", "absent_is_none"),
        ("Limiter", "never_raises"),
        ("Tracer", "never_raises"),
        ("Generation", "never_raises"),
    ],
)
async def test_each_port_check_fires_on_a_broken_implementation(port: str, case: str) -> None:
    args = ("greet",) if port == "PromptSource" else ()
    with pytest.raises((ContractViolation, RuntimeError)) as e:
        await cf.CHECKS[port](_Bad, *args)
    if isinstance(e.value, ContractViolation):
        assert e.value.invariant == case
    else:  # a limiter that cannot hand out a slot raises its own error: the check does not hide it
        assert port == "Limiter"


async def test_the_embedder_and_reranker_checks_fire_on_a_wrong_count() -> None:
    class _Short(StubEmbedder):
        async def embed(self, texts: Any, *, input_type: Any = "document") -> ports.Embedded:
            return ports.Embedded(vectors=(), outcome="ok", stamp=self.stamp)

    with pytest.raises(ContractViolation, match="one_vector_each"):
        await cf.check_embedder(_Short)

    class _Empty(StubReranker):
        async def rerank(self, query: str, documents: Any) -> ports.Reranked:
            return ports.Reranked(scores=(1.0,) * max(len(documents), 1), outcome="ok", model="")

    with pytest.raises(ContractViolation, match="empty_is_ok"):
        await cf.check_reranker(_Empty)


async def test_a_failed_embed_or_rerank_must_name_its_failure() -> None:
    assert await cf.check_embedder(lambda: StubEmbedder(fail="unavailable"))
    assert await cf.check_reranker(lambda: StubReranker(fail="unavailable"))


def test_scripted_reply_is_frozen() -> None:
    r = Reply.ok("v")
    assert isinstance(r, ScriptedReply)
    with pytest.raises(AttributeError):
        r.value = "w"  # type: ignore[misc]
