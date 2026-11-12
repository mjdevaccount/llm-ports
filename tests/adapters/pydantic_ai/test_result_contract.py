"""PydanticAIGenerator's result contract and failure kinds, end to end through pydantic-ai.

FunctionModel drives the agent loop with no wire; `openai_compatible` over an httpx MockTransport
drives a real OpenAI wire with headers. Numbers in the test names are the specs' acceptance tests:
RC = result contract, F = failures.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llm_ports.adapters import litellm_proxy
from llm_ports.adapters import pydantic_ai as pag
from llm_ports.adapters.pydantic_ai import PydanticAIGenerator
from llm_ports.ports import Answered, Cost, GenRequest, Usage
from tests.adapters.pydantic_ai._scripted import scripted_model as _replies


class _Note(BaseModel):
    headline: str
    confidence: float


REQ = GenRequest(model="small-chat", prompt="narrate", output_retries=2)


def _j(headline: str, c: float = 0.5) -> str:
    return json.dumps({"headline": headline, "confidence": c})


class _Events:
    def __init__(self) -> None:
        self.seen: list[tuple[int, str, dict[str, Any]]] = []

    def __call__(self, level: int, event: str, fields: Any) -> None:
        self.seen.append((level, event, dict(fields)))

    def named(self, event: str) -> list[dict[str, Any]]:
        return [f for _, e, f in self.seen if e == event]


class _Pricer:
    def __init__(self, usd: Decimal | None = Decimal("0.5"), *, boom: bool = False) -> None:
        self.usd, self.boom, self.calls = usd, boom, 0

    def price(self, usage: Usage, answered: Answered) -> Cost:
        self.calls += 1
        if self.boom:
            raise RuntimeError("pricer bug")
        if self.usd is None:
            return Cost(usd=None, source="unknown", priced_model=answered.model, reason="unknown_model")
        return Cost(usd=self.usd, source="price_table", priced_model=answered.model)


# ── the wire ─────────────────────────────────────────────────────────────────────────────────────


def _completion(content: str | None, *, finish: str = "stop", usage: Any = "default", **kw: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if "refusal" in kw:
        message["refusal"] = kw.pop("refusal")
    body: dict[str, Any] = {
        "id": kw.pop("id", "resp-1"),
        "object": "chat.completion",
        "created": 0,
        "model": kw.pop("model", "small-chat"),
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        **kw,
    }
    if usage == "default":
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    elif usage is not None:
        body["usage"] = usage
    return body


@pytest.fixture(name="wire")
def _wire() -> dict[str, Any]:
    """`wire["replies"]` is a list of `(status, body, headers)`; the last repeats."""
    state: dict[str, Any] = {"replies": [], "sent": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["sent"].append(json.loads(request.content))
        code, body, headers = state["replies"][min(len(state["sent"]) - 1, len(state["replies"]) - 1)]
        return httpx.Response(code, json=body, headers=headers)

    state["transport"] = httpx.MockTransport(handler)
    return state


def _gen(wire: dict[str, Any], **kw: Any) -> PydanticAIGenerator:
    return PydanticAIGenerator(
        model_for=pag.openai_compatible("http://gateway.test/v1", "k", transport=wire["transport"]), **kw
    )


# ── RC2, RC3: usage is the sum over every request, unknown stays None ───────────────────────────


async def test_rc2_two_re_asks_report_the_sum_of_three_requests() -> None:
    model = _replies(_j(""), _j(""), _j("third"))
    res = await PydanticAIGenerator(model).generate(
        REQ, _Note, validate=lambda n: [] if n.headline else ["headline: empty"], retry_message="; ".join
    )
    assert res.outcome == "ok" and res.retries == 2
    assert res.usage_v2 is not None and res.usage_v2.requests == 3 and res.usage_v2.unmeasured == 0
    one = await PydanticAIGenerator(_replies(_j("first"))).generate(REQ, _Note)
    assert one.usage_v2 is not None and one.usage_v2.input is not None and res.usage_v2.input is not None
    assert res.usage_v2.input > one.usage_v2.input, "every request's tokens are counted, not the last one's"


async def test_rc2_two_re_asks_on_the_wire_sum_every_reply(wire) -> None:
    wire["replies"] = [(200, _completion(_j("")), {}), (200, _completion(_j("")), {}), (200, _completion(_j("x")), {})]
    res = await _gen(wire).generate(
        REQ, _Note, validate=lambda n: [] if n.headline else ["headline: empty"], retry_message="; ".join
    )
    assert res.outcome == "ok" and len(wire["sent"]) == 3
    assert res.usage_v2 == Usage(input=33, output=21, requests=3)
    assert res.usage == {"input": 33, "output": 21}, "the 1.x field is derived from the sum (1.22 kept the last reply)"


async def test_rc3_cached_and_reasoning_tokens_fill_their_buckets(wire) -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "prompt_tokens_details": {"cached_tokens": 64, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 25},
    }
    wire["replies"] = [(200, _completion(_j("a"), usage=usage), {})]
    u = (await _gen(wire).generate(REQ, _Note)).usage_v2
    assert u is not None and (u.input, u.output, u.cache_read, u.reasoning) == (100, 40, 64, 25)
    assert u.details == {"prompt_tokens_details.audio_tokens": 0}


async def test_rc3_a_llama_cpp_shaped_body_fills_cache_read_from_the_same_field(wire) -> None:
    usage = {
        "prompt_tokens": 50,
        "completion_tokens": 9,
        "total_tokens": 59,
        "prompt_tokens_details": {"cached_tokens": 48},
    }
    timings = {"prompt_n": 2, "predicted_n": 9}
    wire["replies"] = [(200, _completion(_j("a"), usage=usage, timings=timings), {})]
    u = (await _gen(wire).generate(REQ, _Note)).usage_v2
    assert u is not None and (u.cache_read, u.reasoning, u.cache_write) == (48, None, None)


async def test_a_gateway_relaying_anthropic_fills_cache_read_and_write(wire) -> None:
    usage = {
        "prompt_tokens": 70,
        "completion_tokens": 5,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 12,
    }
    wire["replies"] = [(200, _completion(_j("a"), usage=usage), {})]
    u = (await _gen(wire).generate(REQ, _Note)).usage_v2
    assert u is not None and (u.cache_read, u.cache_write) == (30, 12) and u.details == {}


async def test_a_reply_with_no_usage_is_one_unmeasured_request_and_its_cost_is_unknown(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a"), usage=None), {})]
    res = await _gen(wire, pricer=_Pricer(), on_event=events).generate(REQ, _Note)
    assert res.outcome == "ok" and res.usage_v2 == Usage(requests=1, unmeasured=1)
    assert res.usage == {"input": None, "output": None}, "unknown is None, never 0"
    assert len(events.named("llm_usage_unmeasured")) == 1


async def test_a_measured_call_emits_no_unmeasured_event(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a")), {})]
    await _gen(wire, on_event=events).generate(REQ, _Note)
    assert events.named("llm_usage_unmeasured") == []


# ── RC4, RC5: cost precedence and unknown cost ───────────────────────────────────────────────────


async def test_rc4_an_unknown_model_with_usage_is_unknown_cost_and_emits_once(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a"), model="rc4-unpriced-model"), {})]
    gen = _gen(wire, pricer=_Pricer(usd=None), on_event=events)
    res = await gen.generate(REQ, _Note)
    assert res.cost == Cost(usd=None, source="unknown", priced_model="rc4-unpriced-model", reason="unknown_model")
    await gen.generate(REQ, _Note)
    fired = events.named("llm_cost_unknown")
    assert len(fired) == 1 and fired[0]["model"] == "rc4-unpriced-model", "once per model per process"


async def test_rc4_a_known_model_is_a_decimal_and_no_event(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a")), {})]
    res = await _gen(wire, pricer=_Pricer(Decimal("0.002")), on_event=events).generate(REQ, _Note)
    assert res.cost is not None and res.cost.usd == Decimal("0.002") and res.cost.source == "price_table"
    assert events.named("llm_cost_unknown") == []


async def test_rc5_the_gateway_cost_wins_over_the_pricer(wire) -> None:
    pricer = _Pricer(Decimal("0.5"))
    wire["replies"] = [(200, _completion(_j("a")), {"x-litellm-response-cost": "0.0123"})]
    res = await _gen(wire, pricer=pricer, gateway=litellm_proxy).generate(REQ, _Note)
    assert res.cost is not None and res.cost.usd == Decimal("0.0123") and res.cost.source == "gateway"
    assert pricer.calls == 0, "the pricer is asked only when the gateway gave nothing"


async def test_with_no_gateway_header_the_pricer_answers(wire) -> None:
    wire["replies"] = [(200, _completion(_j("a")), {})]
    res = await _gen(wire, pricer=_Pricer(Decimal("0.5")), gateway=litellm_proxy).generate(REQ, _Note)
    assert res.cost is not None and res.cost.usd == Decimal("0.5")


async def test_an_invalid_gateway_header_is_absent_and_reported(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a")), {"x-litellm-response-cost": "NaN"})]
    res = await _gen(wire, pricer=_Pricer(Decimal("0.5")), gateway=litellm_proxy, on_event=events).generate(REQ, _Note)
    assert res.cost is not None and res.cost.usd == Decimal("0.5")
    assert len(events.named("llm_gateway_header_invalid")) == 1


async def test_a_pricer_that_raises_is_unknown_cost_and_an_error_event(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a")), {})]
    res = await _gen(wire, pricer=_Pricer(boom=True), on_event=events).generate(REQ, _Note)
    assert res.outcome == "ok" and res.cost is not None and (res.cost.usd, res.cost.reason) == (None, "pricer_error")
    assert [lvl for lvl, e, _ in events.seen if e == "llm_pricer_failed"] == [logging.ERROR]


async def test_with_no_pricer_and_no_gateway_cost_is_none(wire) -> None:
    wire["replies"] = [(200, _completion(_j("a")), {})]
    assert (await _gen(wire).generate(REQ, _Note)).cost is None


async def test_a_failed_call_that_spent_tokens_still_reports_usage_and_cost(wire) -> None:
    wire["replies"] = [(200, _completion("not json"), {"x-litellm-response-cost": "0.001"})]
    res = await _gen(wire, gateway=litellm_proxy).generate(REQ, _Note)
    assert res.outcome == "invalid" and res.usage_v2 is not None and res.usage_v2.requests == 3
    assert res.cost is not None and res.cost.usd == Decimal("0.003")


# ── RC6: who answered ────────────────────────────────────────────────────────────────────────────


async def test_rc6_answered_is_what_the_body_and_the_gateway_said(wire) -> None:
    headers = {"x-litellm-model-id": "dep-42"}
    wire["replies"] = [(200, _completion(_j("a"), model="vendor/model-2026-10", id="chatcmpl-9"), headers)]
    res = await _gen(wire, gateway=litellm_proxy).generate(REQ, _Note)
    assert res.answered == Answered(
        requested="small-chat", model="vendor/model-2026-10", deployment="dep-42", response_id="chatcmpl-9"
    )
    assert res.model == "small-chat", "the 1.x field stays the requested id"


async def test_without_a_gateway_reader_there_is_no_deployment(wire) -> None:
    wire["replies"] = [(200, _completion(_j("a")), {"x-litellm-model-id": "dep-42"})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.answered is not None and res.answered.deployment is None


async def test_the_framework_path_reads_answered_from_the_response() -> None:
    res = await PydanticAIGenerator(_replies(_j("a"))).generate(REQ, _Note)
    # FunctionModel names itself `function:<fn>:`; what matters is that it is not the requested id echoed.
    assert res.answered is not None and res.answered.model == "function:fn:" and res.answered.requested == "small-chat"


# ── RC7: finish ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("raw", "unified"), [("stop", "stop"), ("eos", "other")])
async def test_rc7_finish_keeps_the_raw_string(wire, raw: str, unified: str) -> None:
    wire["replies"] = [(200, _completion(_j("a"), finish=raw), {})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.finish is not None and (res.finish.unified, res.finish.raw) == (unified, raw)
    assert res.finish_reason == raw


# ── RC8: a setting the model ignores is a warning on the result ──────────────────────────────────


class _NoTemperature(FunctionModel):
    unsupported_settings = frozenset({"temperature"})


async def test_rc8_an_ignored_setting_yields_exactly_one_warning() -> None:
    model = _NoTemperature(lambda _m, _i: ModelResponse(parts=[TextPart("ok")]))
    req = GenRequest(model="m", prompt="p", settings={"temperature": 0.2, "max_tokens": 50})
    res = await PydanticAIGenerator(model).generate(req, str)
    assert res.outcome == "ok"
    assert [(w.kind, w.setting) for w in res.warnings] == [("unsupported", "temperature")]


async def test_a_model_that_ignores_nothing_warns_nothing() -> None:
    req = GenRequest(model="m", prompt="p", settings={"temperature": 0.2})
    res = await PydanticAIGenerator(_replies("ok")).generate(req, str)
    assert res.warnings == ()


# ── F3: truncation ───────────────────────────────────────────────────────────────────────────────


async def test_f3_a_length_finish_that_parsed_is_truncated_not_ok(wire) -> None:
    # 2.0 behaviour fix 1: a cut reply is never `ok`, even when the cut text happened to parse.
    wire["replies"] = [(200, _completion(_j("cut short"), finish="length"), {})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "truncated" and res.failure is not None and res.failure.kind == "truncated"
    assert res.value is not None and res.value.headline == "cut short", "the cut value is kept for the caller"


async def test_f3_a_length_finish_that_did_not_parse_is_truncated(wire) -> None:
    wire["replies"] = [(200, _completion('{"headline": "cu', finish="length"), {})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "truncated" and res.value is None


async def test_f3_accept_truncated_restores_ok_with_a_warning(wire) -> None:
    wire["replies"] = [(200, _completion(_j("cut short"), finish="length"), {})]
    req = GenRequest(model="small-chat", prompt="n", accept_truncated=True)
    res = await _gen(wire).generate(req, _Note)
    assert res.outcome == "ok" and res.failure is None and res.value is not None
    assert [w.setting for w in res.warnings] == ["accept_truncated"]


async def test_f3_a_stop_finish_is_ok_and_carries_no_failure(wire) -> None:
    wire["replies"] = [(200, _completion(_j("whole")), {})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "ok" and res.failure is None and res.warnings == ()


# ── F4: a bug is internal ────────────────────────────────────────────────────────────────────────


async def test_f4_a_validator_bug_is_internal_with_an_event_and_one_request(wire) -> None:
    # 2.0 behaviour fix 2: a caller's bug is `internal` (request_fault), never a model outage.
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a")), {})]

    def validate(n: _Note) -> list[str]:
        return [{"x": 1}["missing"]]

    res = await _gen(wire, on_event=events).generate(REQ, _Note, validate=validate)
    assert res.outcome == "request_fault" and res.failure is not None
    assert (res.failure.kind, res.failure.retryable) == ("internal", False)
    assert "KeyError" in res.failure.reason and len(wire["sent"]) == 1
    fired = events.named("llm_call_internal_error")
    assert len(fired) == 1 and "KeyError" in fired[0]["traceback"]
    assert [lvl for lvl, e, _ in events.seen if e == "llm_call_internal_error"] == [logging.ERROR]


async def test_a_model_failure_emits_no_internal_error_event(wire) -> None:
    events = _Events()
    wire["replies"] = [(500, {"error": {"message": "boom"}}, {})]
    res = await _gen(wire, on_event=events).generate(REQ, _Note)
    assert res.outcome == "unavailable" and events.named("llm_call_internal_error") == []


async def test_a_model_for_that_raises_a_bug_is_internal() -> None:
    def broken(_req: GenRequest) -> Any:
        raise AttributeError("no such attribute")

    events = _Events()
    res = await PydanticAIGenerator(model_for=broken, on_event=events).generate(REQ, _Note)
    assert res.outcome == "request_fault" and res.failure is not None and res.failure.kind == "internal"
    assert res.answered == Answered(requested="small-chat") and len(events.named("llm_call_internal_error")) == 1


# ── F5: refusal, three ways ──────────────────────────────────────────────────────────────────────


async def test_f5_a_content_filter_finish_is_refused(wire) -> None:
    wire["replies"] = [(200, _completion(None, finish="content_filter"), {})]
    res = await _gen(wire).generate(REQ, str)
    assert res.outcome == "refused" and res.failure is not None and res.failure.kind == "refused"


async def test_f5_a_refusal_field_is_refused(wire) -> None:
    wire["replies"] = [(200, _completion(None, refusal="I can't help with that."), {})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "refused" and res.value is None


async def test_f5_a_content_filter_error_from_the_model_is_refused() -> None:
    from pydantic_ai.exceptions import ContentFilterError

    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        raise ContentFilterError("blocked by the provider's filter")

    res = await PydanticAIGenerator(FunctionModel(fn)).generate(REQ, str)
    assert res.outcome == "refused"


# ── F6, F7: gateway 429s and context overflow on the wire ───────────────────────────────────────


async def test_f6_a_gateway_budget_429_is_quota(wire) -> None:
    body = {"error": {"message": "Budget has been exceeded! Max budget: 5.0", "type": "budget_exceeded", "code": "429"}}
    wire["replies"] = [(429, body, {})]
    res = await _gen(wire, extra=(litellm_proxy.classify_gateway,)).generate(REQ, _Note)
    assert res.failure is not None and (res.failure.kind, res.failure.retryable) == ("quota", False)


async def test_f6_a_parallel_limit_429_is_rate_limited_with_its_wait(wire) -> None:
    body = {"error": {"message": "Max parallel request limit reached", "type": "None", "code": "429"}}
    wire["replies"] = [(429, body, {"retry-after": "3"})]
    res = await _gen(wire, extra=(litellm_proxy.classify_gateway,)).generate(REQ, _Note)
    assert res.outcome == "unavailable" and res.failure is not None
    assert (res.failure.kind, res.failure.retryable, res.failure.retry_after_s) == ("rate_limited", True, 3.0)


async def test_f7_a_context_overflow_body_is_request_fault(wire) -> None:
    body = {"error": {"message": "the request exceeds the available context size", "type": "exceed_context_size_error"}}
    wire["replies"] = [(400, body, {})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "request_fault" and res.failure is not None and res.failure.kind == "context_overflow"


async def test_a_failure_reason_is_the_1x_reason(wire) -> None:
    wire["replies"] = [(500, {"error": {"message": "boom"}}, {})]
    res = await _gen(wire).generate(REQ, _Note)
    assert res.failure is not None and res.reason == res.failure.reason and res.failure.status == 500


# ── a cost header is a report, never a reason to fail the call ────────────────────────────────────

#: (header, the gateway cost it yields): an implausible magnitude is absent; a signed zero is 0.
COST_HEADERS = [("1e9999999", None), ("1e7", None), ("-0", Decimal(0)), ("1e-9999999", None)]


@pytest.mark.parametrize(("header", "usd"), COST_HEADERS, ids=[h for h, _ in COST_HEADERS])
async def test_an_implausible_cost_header_never_fails_generate(wire, header: str, usd: Decimal | None) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a")), {"x-litellm-response-cost": header})]
    res = await _gen(wire, gateway=litellm_proxy, on_event=events).generate(REQ, _Note)
    assert res.outcome == "ok" and res.value is not None and res.failure is None
    assert (res.cost.usd if res.cost is not None else None) == usd
    assert len(events.named("llm_gateway_header_invalid")) == (1 if usd is None else 0)


class _HugeGateway:
    """A custom gateway reader that passes an overflowing cost through, or raises."""

    def __init__(self, *, boom: bool = False) -> None:
        self.boom = boom

    def cost_from_headers(self, _headers: Any) -> Decimal | None:
        if self.boom:
            raise RuntimeError("reader bug")
        return Decimal("9e999999")

    def answered_from_headers(self, _headers: Any) -> tuple[str | None, str | None]:
        if self.boom:
            raise RuntimeError("reader bug")
        return None, None


async def test_a_cost_sum_that_overflows_is_unknown_not_a_failure(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("")), {}), (200, _completion(_j("b")), {})]
    gen = _gen(wire, gateway=_HugeGateway(), pricer=_Pricer(Decimal("0.5")), on_event=events)
    res = await gen.generate(REQ, _Note, validate=lambda n: [] if n.headline else ["empty"], retry_message="; ".join)
    assert res.outcome == "ok" and res.retries == 1
    assert res.cost is not None and res.cost.source == "price_table", "the pricer answers once the gateway sum is lost"
    (fields,) = events.named("llm_gateway_header_invalid")
    assert fields["detail"].startswith("Overflow")


async def test_a_gateway_reader_that_raises_is_unknown_cost_and_deployment_with_error_events(wire) -> None:
    events = _Events()
    wire["replies"] = [(200, _completion(_j("a")), {})]
    res = await _gen(wire, gateway=_HugeGateway(boom=True), on_event=events).generate(REQ, _Note)
    assert res.outcome == "ok" and res.failure is None and res.cost is None and res.answered.deployment is None
    assert [lvl for lvl, e, _ in events.seen if e == "llm_gateway_header_invalid"] == [logging.ERROR] * 2


# ── step 9: a reply the provider client cannot read is the provider's fault ──────────────────────


async def test_a_200_with_no_choices_is_malformed_response_and_retryable(wire) -> None:
    events = _Events()
    body = _completion(_j("a"))
    body["choices"] = []
    wire["replies"] = [(200, body, {})]
    res = await _gen(wire, on_event=events).generate(REQ, _Note)
    assert res.failure is not None
    assert (res.outcome, res.failure.kind, res.failure.retryable) == ("unavailable", "malformed_response", True)
    assert events.named("llm_call_internal_error") == [], "a provider's bad reply does not page as a code bug"


async def test_a_validator_bug_after_a_good_reply_stays_internal(wire) -> None:
    wire["replies"] = [(200, _completion(_j("a")), {})]

    def validate(_n: _Note) -> list[str]:
        return [{}["missing"]]

    res = await _gen(wire).generate(REQ, _Note, validate=validate)
    assert res.failure is not None and (res.outcome, res.failure.kind) == ("request_fault", "internal")


@pytest.mark.parametrize(
    ("url", "mock"), [("http://[::1/v1", True), ("ftp://gateway.test/v1", False)], ids=["bad_host", "ftp"]
)
async def test_a_url_no_client_can_send_is_internal_with_no_request(wire, url: str, mock: bool) -> None:
    # A MockTransport serves any scheme; the real transport refuses ftp before it connects.
    events = _Events()
    kw: dict[str, Any] = {"transport": wire["transport"]} if mock else {}
    gen = PydanticAIGenerator(model_for=pag.openai_compatible(url, "k", **kw), on_event=events)
    res = await gen.generate(REQ, _Note)
    assert res.failure is not None
    assert (res.outcome, res.failure.kind, res.failure.retryable) == ("request_fault", "internal", False)
    assert wire["sent"] == [] and len(events.named("llm_call_internal_error")) == 1


async def test_an_output_mode_the_model_lacks_is_internal_with_no_request(wire) -> None:
    events = _Events()
    model_for = pag.openai_compatible(
        "http://gateway.test/v1", "k", transport=wire["transport"], profile={"supports_json_schema_output": False}
    )
    res = await PydanticAIGenerator(model_for=model_for, on_event=events).generate(REQ, _Note)
    assert res.failure is not None
    assert (res.outcome, res.failure.kind, res.failure.retryable) == ("request_fault", "internal", False)
    assert res.reason.startswith("UserError")
    assert wire["sent"] == [] and len(events.named("llm_call_internal_error")) == 1
