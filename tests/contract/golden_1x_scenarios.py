"""The scenarios behind the 1.x golden fixture, runnable against any llm-ports version.

`tests/fixtures/golden/result_1x_v1_22_0.json` was written by running this module against the
1.22.0 tree (`PYTHONPATH=<1.22.0 checkout>/src python -m tests.contract.golden_1x_scenarios`).
Each scenario reads only the fields 1.22.0 had. The two 2.0 behaviour fixes (a cut reply that
parsed, a caller's bug) are deliberately not scenarios: they are the changes, tested elsewhere.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llm_ports.adapters import pydantic_ai as pag
from llm_ports.ports import Generated, GenRequest
from llm_ports.testing import StubEmbedder, StubGenerator, StubReranker

FIELDS_1X = ("value", "outcome", "model", "reason", "retries", "problems", "usage", "finish_reason", "sources")


class Note(BaseModel):
    headline: str
    confidence: float


class Refused(Exception):
    pass


REQ = GenRequest(model="small-chat", prompt="narrate", output_retries=1)


def _completion(content: str, *, finish: str = "stop", usage: bool = True, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "small-chat",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        **extra,
    }
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    return body


def _wire(*replies: tuple[int, Any]) -> pag.PydanticAIGenerator:
    sent: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        sent.append(1)
        code, body = replies[min(len(sent) - 1, len(replies) - 1)]
        return httpx.Response(code, json=body)

    return pag.PydanticAIGenerator(
        model_for=pag.openai_compatible("http://gateway.test/v1", "k", transport=httpx.MockTransport(handler)),
        gated=(Refused,),
    )


def _dump(res: Generated[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in FIELDS_1X:
        v = getattr(res, f)
        if isinstance(v, BaseModel):
            v = v.model_dump()
        elif isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, dict):
            v = dict(v)
        out[f] = v
    return out


def _j(h: str) -> str:
    return json.dumps({"headline": h, "confidence": 0.5})


async def run_all() -> dict[str, Any]:
    """Run every scenario; return its 1.x fields by scenario name."""
    out: dict[str, Any] = {}
    out["wire_ok"] = _dump(
        await _wire((200, _completion(_j("a"), sources=["https://a.example/1"]))).generate(REQ, Note)
    )
    out["wire_str_no_usage"] = _dump(await _wire((200, _completion("plain", usage=False))).generate(REQ, str))
    out["wire_reask"] = _dump(
        await _wire((200, _completion(_j(""))), (200, _completion(_j("b")))).generate(
            REQ, Note, validate=lambda n: [] if n.headline else ["headline: empty"], retry_message="; ".join
        )
    )
    out["wire_kept_problems"] = _dump(
        await _wire((200, _completion(_j("")))).generate(REQ, Note, validate=lambda n: ["always"])
    )
    out["wire_invalid"] = _dump(await _wire((200, _completion("not json"))).generate(REQ, Note))
    out["wire_length_unparsed"] = _dump(await _wire((200, _completion('{"head', finish="length"))).generate(REQ, Note))
    out["wire_400"] = _dump(await _wire((400, {"error": {"message": "bad"}})).generate(REQ, Note))
    out["wire_413"] = _dump(await _wire((413, {"error": {"message": "big"}})).generate(REQ, Note))
    out["wire_429"] = _dump(await _wire((429, {"error": {"message": "slow"}})).generate(REQ, Note))
    out["wire_500"] = _dump(await _wire((500, {"error": {"message": "boom"}})).generate(REQ, Note))

    def refusing(_req: GenRequest) -> Any:
        raise Refused("category_disabled: switched off")

    out["gated_model_for"] = _dump(
        await pag.PydanticAIGenerator(model_for=refusing, gated=(Refused,)).generate(REQ, Note)
    )

    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("prose")])

    out["function_ok"] = _dump(await pag.PydanticAIGenerator(FunctionModel(fn)).generate(REQ, str))

    @asynccontextmanager
    async def fixed(_req: GenRequest) -> AsyncIterator[Model]:
        yield FunctionModel(fn)

    out["model_for_ok"] = _dump(await pag.PydanticAIGenerator(model_for=fixed).generate(REQ, str))
    out["stub_ok"] = _dump(await StubGenerator(lambda _r, _o: "x").generate(REQ, str))
    out["stub_raises"] = _dump(await StubGenerator(lambda _r, _o: 1 / 0).generate(REQ, str))
    emb = await StubEmbedder(fail="unavailable").embed(["a"])
    out["stub_embed_fail"] = {"outcome": emb.outcome, "reason": emb.reason, "input_tokens": emb.input_tokens}
    rr = await StubReranker(fail="request_fault").rerank("q", ["d"])
    out["stub_rerank_fail"] = {"outcome": rr.outcome, "reason": rr.reason, "model": rr.model}
    return out


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_all()), indent=2, sort_keys=True))
