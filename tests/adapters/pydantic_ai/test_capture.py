"""The wire capture: what each response showed, kept as unknown where the wire said nothing.

Unit tests over `record_response` with hand-built httpx2 responses, and the two generator paths
that read a capture differently: a 1.x capture (finish reason only) and a refusal next to content.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llm_ports.adapters.pydantic_ai import PydanticAIGenerator, parse_usage
from llm_ports.adapters.pydantic_ai._capture import capture, record_response
from llm_ports.ports import GenRequest, Usage


async def _record(status: int, body: Any = None, *, content: bytes | None = None, headers: Any = None) -> Any:
    with capture() as cap:
        if content is None:
            await record_response(httpx2.Response(status, json=body, headers=headers or {}))
        else:
            await record_response(httpx2.Response(status, content=content, headers=headers or {}))
    return cap


async def test_an_error_body_becomes_the_latest_facts_and_no_reply() -> None:
    cap = await _record(429, {"error": {"message": "slow", "type": "rate", "code": 429}}, headers={"Retry-After": "2"})
    assert cap.replies == [] and cap.last is not None
    assert (cap.last.status, cap.last.error_type, cap.last.error_code, cap.last.error_message) == (
        429,
        "rate",
        "429",
        "slow",
    )
    assert cap.last.headers["retry-after"] == "2", "header names are lower-cased"
    assert cap.usage == {"input": None, "output": None}, "1.x recorded an error body's usage as unknown"


@pytest.mark.parametrize(
    ("body", "message", "code"),
    [
        ({"error": "plain text error"}, "plain text error", None),
        ({"message": "top-level", "code": True}, "top-level", None),
    ],
    ids=["string_error", "top_level_fields"],
)
async def test_an_error_body_in_another_shape_still_names_its_message(body: Any, message: str, code: Any) -> None:
    cap = await _record(400, body)
    assert cap.last.error_message == message and cap.last.error_code == code


async def test_an_unreadable_body_is_skipped_and_never_raises() -> None:
    cap = await _record(200, content=b"<html>not json</html>")
    assert cap.replies == [] and cap.last is None
    err = await _record(502, content=b"bad gateway")
    assert err.last is not None and err.last.status == 502 and err.last.error_message is None
    listed = await _record(200, [1, 2])
    assert listed.replies == []


async def test_a_hook_failure_never_breaks_the_call() -> None:
    class Broken(httpx2.Response):
        async def aread(self) -> bytes:
            raise RuntimeError("stream died")

    with capture() as cap:
        await record_response(Broken(200, json={}))
    assert cap.replies == [] and cap.last is None


async def test_a_reply_keeps_model_id_refusal_and_sources() -> None:
    body = {
        "id": 7,
        "model": "m-2026",
        "choices": [{"finish_reason": "stop", "message": {"content": "x", "refusal": "no"}}],
        "sources": ["https://a.example", 3],
    }
    cap = await _record(200, body)
    (reply,) = cap.replies
    assert (reply.model, reply.response_id, reply.refusal, reply.finish_raw) == ("m-2026", "7", True, "stop")
    assert cap.sources == ("https://a.example",) and reply.usage == Usage(requests=1, unmeasured=1)


def test_parse_usage_keeps_unknown_as_none_and_collects_details() -> None:
    assert parse_usage(None) == Usage(requests=1, unmeasured=1)
    got = parse_usage({"prompt_tokens": True, "completion_tokens": 3, "extra_tokens": 2, "nested": {"a": 1, "b": "x"}})
    assert got.input is None and got.output == 3 and got.details == {"extra_tokens": 2, "nested.a": 1}


def test_a_capture_reset_from_another_context_clears_rather_than_leaks() -> None:
    cm = capture()
    cm.__enter__()
    contextvars.copy_context().run(cm.__exit__, None, None, None)  # the token belongs to the outer context


# ── the generator over a capture it did not build ────────────────────────────────────────────────


@dataclass
class _Legacy:
    """A 1.x capture: finish reason, usage and sources only."""

    finish_reason: str | None = None
    usage: Mapping[str, int | None] = field(default_factory=dict)
    sources: tuple[str, ...] | None = None


def _legacy(finish: str) -> Any:
    @contextmanager
    def cm() -> Iterator[_Legacy]:
        yield _Legacy(finish_reason=finish, usage={"input": 3, "output": 2})

    return cm


def _model(text: str) -> FunctionModel:
    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(fn)


async def test_a_1x_capture_still_names_a_cut_reply_truncated() -> None:
    res = await PydanticAIGenerator(_model("cut"), capture=_legacy("length")).generate(GenRequest("m", "p"), str)
    assert res.outcome == "truncated" and res.finish is not None and res.finish.raw == "length"


async def test_a_1x_capture_with_a_content_filter_finish_is_refused_with_no_value() -> None:
    gen = PydanticAIGenerator(_model("partial"), capture=_legacy("content_filter"))
    res = await gen.generate(GenRequest("m", "p"), str)
    assert res.outcome == "refused" and res.value is None


async def test_a_refusal_next_to_content_is_refused_with_no_value() -> None:
    import httpx

    from llm_ports.adapters import pydantic_ai as pag

    body = {
        "id": "r",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x", "refusal": "no"}}
        ],
    }
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json=body))
    gen = PydanticAIGenerator(model_for=pag.openai_compatible("http://gateway.test/v1", transport=transport))
    res = await gen.generate(GenRequest("m", "p"), str)
    assert res.outcome == "refused" and res.value is None
