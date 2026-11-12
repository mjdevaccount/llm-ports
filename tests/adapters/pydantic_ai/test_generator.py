"""PydanticAIGenerator: the Generator port on pydantic-ai.

Two halves. FunctionModel drives the agent loop (re-ask, kept problems, parse failure) with no
wire. `openai_compatible` over an httpx MockTransport drives a real OpenAI wire -- JSON in, the
capture of finish_reason/usage/sources, the status mapping -- with no network.
"""

from __future__ import annotations

import json
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import TextPart
from pydantic_ai.models import Model

from llm_ports import ports
from llm_ports.adapters import pydantic_ai as pag
from llm_ports.adapters.pydantic_ai import PydanticAIGenerator, _generator, classify
from llm_ports.ports import GenRequest, WirePolicy
from tests.adapters.pydantic_ai._scripted import scripted as _replies


class _Note(BaseModel):
    headline: str
    confidence: float


class _Refused(Exception):
    """A consumer's switchboard refusal, raised from its own transport."""


REQ = GenRequest(model="small-chat", prompt="narrate", output_retries=1)


def _j(headline: str, c: float = 0.5) -> str:
    return json.dumps({"headline": headline, "confidence": c})


def test_exactly_one_model_source_is_required() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PydanticAIGenerator()
    model, _ = _replies("x")
    with pytest.raises(ValueError, match="exactly one"):
        PydanticAIGenerator(model, model_for=pag.openai_compatible("http://x/v1"))


# ── the agent loop (no wire) ─────────────────────────────────────────────────────────────────────


async def test_a_valid_reply_is_ok_with_zero_retries() -> None:
    model, calls = _replies(_j("SPY pinned"))
    res = await PydanticAIGenerator(model).generate(REQ, _Note)
    assert res.outcome == "ok" and res.value == _Note(headline="SPY pinned", confidence=0.5)
    assert res.retries == 0 and res.problems == () and len(calls) == 1


async def test_a_validator_problem_re_asks_with_the_retry_message() -> None:
    model, calls = _replies(_j(""), _j("QQQ leads"))
    seen: list[str] = []
    res = await PydanticAIGenerator(model).generate(
        REQ,
        _Note,
        validate=lambda n: ["empty headline"] if not n.headline else [],
        retry_message=lambda p: seen.append(p[0]) or "fix: " + p[0],
    )
    assert res.outcome == "ok" and res.value is not None and res.value.headline == "QQQ leads"
    assert res.retries == 1 and len(calls) == 2 and seen == ["empty headline"]


async def test_problems_left_after_the_re_asks_are_kept_with_the_value() -> None:
    model, calls = _replies(_j(""))
    res = await PydanticAIGenerator(model).generate(REQ, _Note, validate=lambda _n: ["still bad"])
    assert res.outcome == "ok" and res.value is not None, "the caller prunes; the port never drops"
    assert res.problems == ("still bad",) and len(calls) == 2


async def test_a_reply_that_never_parses_is_invalid_and_retries_unknown() -> None:
    model, _ = _replies("not json")
    res = await PydanticAIGenerator(model).generate(REQ, _Note)
    assert res.outcome == "invalid" and res.value is None
    assert res.retries is None, "the run died mid-flight: how many it took is unknown, never 0"


async def test_prose_output_is_a_str() -> None:
    model, _ = _replies("the tape is quiet")
    res = await PydanticAIGenerator(model).generate(REQ, str)
    assert res.outcome == "ok" and res.value == "the tape is quiet"


async def test_a_port_json_output_comes_back_as_a_plain_dict() -> None:
    model, _ = _replies('{"category": "groceries"}')
    out = ports.json_output(
        {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}, "cat"
    )
    res = await PydanticAIGenerator(model).generate(REQ, out)
    assert res.outcome == "ok" and res.value == {"category": "groceries"}


# ── classify: the outcome taxonomy, both directions ─────────────────────────────────────────────


def _chain(top: BaseException, cause: BaseException) -> BaseException:
    top.__cause__ = cause
    return top


@pytest.mark.parametrize(
    ("exc", "finish", "want"),
    [
        (_chain(RuntimeError("APIConnectionError"), _Refused("category_disabled: x")), None, "gated"),
        (UnexpectedModelBehavior("Exceeded maximum output retries"), "length", "truncated"),
        (UnexpectedModelBehavior("Exceeded maximum output retries"), "stop", "invalid"),
        (ModelHTTPError(400, "m", body={"error": "bad"}), None, "request_fault"),
        (ModelHTTPError(413, "m"), None, "request_fault"),
        (ModelHTTPError(500, "m", body="li..."), None, "unavailable"),
        (ModelHTTPError(429, "m"), None, "unavailable"),
        (TimeoutError(), None, "unavailable"),
        (_chain(RuntimeError("wrapped"), _Refused("capped")), "length", "gated"),
    ],
    ids=["gated", "truncated", "invalid", "400", "413", "500", "429", "timeout", "gated-beats-length"],
)
def test_classify(exc: BaseException, finish: str | None, want: str) -> None:
    assert classify(exc, finish, gated=(_Refused,))[0] == want


def test_with_no_gated_types_a_refusal_is_a_bug_not_an_outage() -> None:
    # 2.0 behaviour fix: an unrecognised exception is `internal` (request_fault), never `unavailable`.
    assert classify(_chain(RuntimeError("x"), _Refused("no")))[0] == "request_fault"


def test_a_reason_never_carries_the_error_body() -> None:
    _, reason = classify(ModelHTTPError(500, "m", body={"message": "li" * 400}))
    assert reason == "ModelHTTPError: status 500"


# ── a real OpenAI wire through openai_compatible, no network ────────────────────────────────────


def _completion(content: str, *, finish: str = "stop", **extra: Any) -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "small-chat",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        **extra,
    }


@pytest.fixture(name="wire")
def _wire() -> dict[str, Any]:
    """A MockTransport recording what reached the wire; `wire["transport"]` goes to openai_compatible."""
    state: dict[str, Any] = {"replies": [], "sent": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["sent"].append(json.loads(request.content))
        code, body = state["replies"][min(len(state["sent"]) - 1, len(state["replies"]) - 1)]
        return httpx.Response(code, json=body)

    state["transport"] = httpx.MockTransport(handler)
    return state


def _gen(wire: dict[str, Any], policy: WirePolicy | None = None) -> PydanticAIGenerator:
    return PydanticAIGenerator(
        model_for=pag.openai_compatible("http://gateway.test/v1", "k", transport=wire["transport"]),
        gated=(_Refused,),
        policy=policy,
    )


async def test_the_wire_returns_value_usage_and_the_sources_pydantic_ai_drops(wire) -> None:
    wire["replies"] = [(200, _completion(_j("cited"), sources=["https://a.example/1", 3]))]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "ok" and res.value is not None and res.value.headline == "cited"
    assert res.sources == ("https://a.example/1",)
    assert res.usage == {"input": 11, "output": 7} and res.finish_reason == "stop"
    assert wire["sent"][0]["response_format"]["type"] == "json_schema", "NativeOutput sends the grammar"
    assert wire["sent"][0]["model"] == "small-chat"


async def test_a_length_finish_is_truncated_not_invalid(wire) -> None:
    wire["replies"] = [(200, _completion('{"headline": "cut', finish="length"))]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "truncated" and res.finish_reason == "length"


@pytest.mark.parametrize(("code", "want"), [(400, "request_fault"), (500, "unavailable")])
async def test_status_codes_map_to_outcomes(wire, code: int, want: str) -> None:
    wire["replies"] = [(code, {"error": {"message": "nope"}})]
    res = await _gen(wire, WirePolicy.none()).generate(REQ, _Note)
    assert res.outcome == want
    assert len(wire["sent"]) == 1, "the SDK's own retries are off by default"


async def test_the_default_wire_policy_retries_a_server_error_once_and_never_a_bad_request(wire) -> None:
    wire["replies"] = [(500, {"error": {"message": "boom"}})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "unavailable" and len(wire["sent"]) == 2 and len(res.attempts) == 1
    wire["sent"].clear()
    wire["replies"] = [(400, {"error": {"message": "bad"}})]
    assert (await _gen(wire).generate(REQ, _Note)).outcome == "request_fault" and len(wire["sent"]) == 1


async def test_a_refusing_transport_is_gated_and_never_raises() -> None:
    def refusing(_req: GenRequest) -> AbstractAsyncContextManager[Model]:
        raise _Refused("category_disabled: x is switched off")

    res = await PydanticAIGenerator(model_for=refusing, gated=(_Refused,)).generate(REQ, _Note)
    assert res.outcome == "gated" and "switched off" in (res.reason or "") and res.retries is None


async def test_wrap_decorates_the_wire_model_and_sees_every_attempt(wire) -> None:
    from pydantic_ai.models.wrapper import WrapperModel

    seen: list[str] = []

    class _Spy(WrapperModel):
        async def request(self, messages, model_settings, model_request_parameters):
            resp = await super().request(messages, model_settings, model_request_parameters)
            seen.append("".join(p.content for p in resp.parts if isinstance(p, TextPart)))
            return resp

    wire["replies"] = [(200, _completion(_j(""))), (200, _completion(_j("second try")))]
    gen = PydanticAIGenerator(
        model_for=pag.openai_compatible("http://gateway.test/v1", transport=wire["transport"]), wrap=_Spy
    )
    res = await gen.generate(
        REQ, _Note, validate=lambda n: [] if n.headline else ["headline: empty"], retry_message=lambda p: "; ".join(p)
    )
    assert res.outcome == "ok" and res.value is not None and res.value.headline == "second try" and res.retries == 1
    assert seen == [_j(""), _j("second try")] and len(wire["sent"]) == 2


async def test_by_default_the_built_model_carries_no_profile_or_settings_of_its_own(wire) -> None:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    async with pag.openai_compatible("http://gateway.test/v1", transport=wire["transport"])(REQ) as model:
        assert isinstance(model, OpenAIChatModel)
        assert model.settings is None
        bare = OpenAIChatModel(REQ.model, provider=OpenAIProvider(openai_client=model.client))
        assert model.profile == bare.profile, "no profile= resolves exactly as the provider does"


async def test_a_profile_passed_in_reaches_the_model_and_moves_a_plain_agent_onto_the_grammar(wire) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.profiles import ModelProfile

    wire["replies"] = [(200, _completion(_j("native")))]
    model_for = pag.openai_compatible(
        "http://gateway.test/v1",
        transport=wire["transport"],
        profile=ModelProfile(default_structured_output_mode="native"),
    )
    async with model_for(REQ) as model:
        assert model.profile["default_structured_output_mode"] == "native"
        res = await Agent(model, output_type=_Note).run("narrate")  # no NativeOutput: the profile decides
    assert res.output.headline == "native"
    sent = wire["sent"][0]
    assert sent["response_format"]["type"] == "json_schema" and "tools" not in sent


async def test_a_profile_function_receives_the_resolved_profile(wire) -> None:
    from pydantic_ai.profiles import ModelProfile

    seen: list[ModelProfile] = []

    def spec(resolved: ModelProfile) -> ModelProfile:
        seen.append(resolved)
        return ModelProfile(**{**resolved, "default_structured_output_mode": "prompted"})

    async with pag.openai_compatible("http://gateway.test/v1", transport=wire["transport"], profile=spec)(REQ) as m:
        assert m.profile["default_structured_output_mode"] == "prompted"
    assert len(seen) == 1 and "supports_json_schema_output" in seen[0]


async def test_model_settings_reach_the_wire_and_a_request_setting_overrides_them(wire) -> None:
    wire["replies"] = [(200, _completion(_j("a")))]
    gen = PydanticAIGenerator(
        model_for=pag.openai_compatible(
            "http://gateway.test/v1", transport=wire["transport"], settings={"temperature": 0.25, "seed": 7}
        )
    )
    assert (await gen.generate(REQ, _Note)).outcome == "ok"
    override = GenRequest(model="small-chat", prompt="n", settings={"temperature": 0.9})
    assert (await gen.generate(override, _Note)).outcome == "ok"
    first, second = wire["sent"]
    assert (first["temperature"], first["seed"]) == (0.25, 7)
    assert (second["temperature"], second["seed"]) == (0.9, 7), "run settings merge over model settings"


async def test_record_response_is_a_no_op_with_no_capture_open() -> None:
    await _generator._record_response(httpx.Response(200, json=_completion("x")))
