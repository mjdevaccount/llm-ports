"""The httpx2 wire under `openai_compatible` and `openai_compatible_embeddings`.

pydantic-ai deprecates a legacy `httpx.AsyncClient` on its OpenAI-compatible providers (removed in
v3), with a `PydanticAIDeprecationWarning`, a `UserWarning` subclass that
`-W error::DeprecationWarning` does not catch. These tests drive both builders with a transport
from each library and fail if pydantic-ai warns, independently of the pytest warning filter.
"""

from __future__ import annotations

import gzip
import json
import warnings
from typing import Any

import httpx
import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import PydanticAIDeprecationWarning

from llm_ports.adapters.pydantic_ai import (
    PydanticAIEmbedder,
    PydanticAIGenerator,
    classify,
    openai_compatible,
    openai_compatible_embeddings,
)
from llm_ports.adapters.pydantic_ai._wire import IndexOrdered, LegacyTransport, as_httpx2
from llm_ports.ports import GenRequest

BASE = "http://gw.test/v1"
REQ = GenRequest(model="small-chat", prompt="narrate")
LIBS = pytest.mark.parametrize("lib", [httpx, httpx2], ids=["legacy-httpx", "httpx2"])


class _Note(BaseModel):
    headline: str


def _completion(content: str) -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "small-chat",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _chat_wire(lib: Any) -> Any:
    def handler(request: Any) -> Any:
        assert request.url.path.endswith("/chat/completions")
        return lib.Response(200, json=_completion(json.dumps({"headline": "hi"})))

    return lib.MockTransport(handler)


def _embed_wire(lib: Any) -> Any:
    def handler(request: Any) -> Any:
        n = len(json.loads(request.content)["input"])
        data = [{"object": "embedding", "index": i, "embedding": [float(i)] * 2} for i in range(n)]
        data.reverse()  # out of order: the re-sort under the client must restore it
        return lib.Response(200, json={"object": "list", "data": data, "model": "e", "usage": {"prompt_tokens": n}})

    return lib.MockTransport(handler)


def _deprecations(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, PydanticAIDeprecationWarning)]


@LIBS
async def test_the_generator_hands_pydantic_ai_an_httpx2_client_with_either_transport(lib: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = await PydanticAIGenerator(model_for=openai_compatible(BASE, "k", transport=_chat_wire(lib))).generate(
            REQ, _Note
        )
    assert _deprecations(caught) == []
    assert res.outcome == "ok" and res.value is not None and res.value.headline == "hi"
    assert res.usage == {"input": 11, "output": 7} and res.finish_reason == "stop", "the capture hook still fires"


@LIBS
async def test_the_embedder_hands_pydantic_ai_an_httpx2_client_with_either_transport(lib: Any) -> None:
    embedder = PydanticAIEmbedder(
        model_for=openai_compatible_embeddings(BASE, "e", "k", transport=_embed_wire(lib)), stamp="e@2", dims=2
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = await embedder.embed(["a", "b", "c"])
    assert _deprecations(caught) == []
    assert res.ok and [v[0] for v in res.vectors] == [0.0, 1.0, 2.0], "index order restored under httpx2"


# ── the legacy bridge ───────────────────────────────────────────────────────────────────────────


async def test_the_bridge_carries_the_whole_request_and_a_decoded_reply() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = gzip.compress(b'{"ok": true}')
        return httpx.Response(201, headers={"content-encoding": "gzip", "x-tag": "a"}, content=body)

    async with httpx2.AsyncClient(transport=as_httpx2(httpx.MockTransport(handler)), timeout=7) as client:
        resp = await client.post(f"{BASE}/x?q=1", json={"a": 1}, headers={"authorization": "Bearer k"})
    req = seen[0]
    assert (req.method, str(req.url)) == ("POST", f"{BASE}/x?q=1")
    assert req.headers["authorization"] == "Bearer k" and json.loads(req.content) == {"a": 1}
    assert req.extensions["timeout"]["read"] == 7, "the client's timeout reaches the legacy transport"
    assert resp.status_code == 201 and resp.json() == {"ok": True} and resp.headers["x-tag"] == "a"
    assert "content-encoding" not in resp.headers, "the body crossed decoded, so its encoding header is dropped"


@pytest.mark.parametrize(
    ("raised", "want"),
    [
        (httpx.ConnectError("refused"), httpx2.ConnectError),
        (httpx.ReadTimeout("slow"), httpx2.ReadTimeout),
        (type("Custom", (httpx.TransportError,), {})("odd"), httpx2.TransportError),
    ],
)
async def test_a_legacy_transport_error_becomes_its_httpx2_twin_chained(raised: Exception, want: type) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise raised

    async with httpx2.AsyncClient(transport=as_httpx2(httpx.MockTransport(handler))) as client:
        with pytest.raises(want) as info:
            await client.get(BASE)
    assert type(info.value) is want and info.value.__cause__ is raised and str(raised) in str(info.value)


async def test_any_other_exception_crosses_the_bridge_unchanged() -> None:
    class Refused(Exception):
        pass

    def handler(_r: httpx.Request) -> httpx.Response:
        raise Refused("category off")

    async with httpx2.AsyncClient(transport=as_httpx2(httpx.MockTransport(handler))) as client:
        with pytest.raises(Refused):
            await client.get(BASE)


async def test_closing_the_bridge_closes_the_legacy_transport() -> None:
    closed: list[bool] = []

    class Inner(httpx.AsyncBaseTransport):
        async def aclose(self) -> None:
            closed.append(True)

    await LegacyTransport(Inner()).aclose()
    assert closed == [True]


def test_as_httpx2_keeps_none_and_an_httpx2_transport() -> None:
    mine = httpx2.MockTransport(lambda r: httpx2.Response(200))
    assert as_httpx2(None) is None and as_httpx2(mine) is mine
    assert isinstance(as_httpx2(httpx.MockTransport(lambda r: httpx.Response(200))), LegacyTransport)


# ── the httpx2 index-order re-sort ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "path", "raw"),
    [
        (500, "/embeddings", b'{"data": [{"index": 1}, {"index": 0}]}'),
        (200, "/chat/completions", b'{"data": [{"index": 1}, {"index": 0}]}'),
        (200, "/embeddings", b"not json"),
        (200, "/embeddings", b'{"data": [{"index": 0}, {"index": 1}]}'),
    ],
    ids=["non-200", "other-path", "unreadable", "already-ordered"],
)
async def test_index_ordered_returns_these_replies_unchanged(status: int, path: str, raw: bytes) -> None:
    inner = httpx2.MockTransport(lambda _r: httpx2.Response(status, content=raw))
    async with httpx2.AsyncClient(transport=IndexOrdered(inner)) as client:
        resp = await client.get(f"{BASE}{path}")
    assert resp.status_code == status and resp.content == raw


def test_index_ordered_defaults_to_httpx2s_own_transport() -> None:
    assert isinstance(IndexOrdered()._inner, httpx2.AsyncHTTPTransport)


async def test_closing_index_ordered_closes_the_inner_transport() -> None:
    closed: list[bool] = []

    class Inner(httpx2.AsyncBaseTransport):
        async def aclose(self) -> None:
            closed.append(True)

    await IndexOrdered(Inner()).aclose()
    assert closed == [True]


# ── classify reads an httpx2 status error ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("code", "want"), [(400, "request_fault"), (503, "unavailable")])
def test_classify_maps_an_httpx2_status_error(code: int, want: str) -> None:
    req = httpx2.Request("POST", BASE)
    exc = httpx2.HTTPStatusError("bad", request=req, response=httpx2.Response(code, request=req))
    assert classify(exc)[0] == want
