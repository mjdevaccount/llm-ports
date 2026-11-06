"""PydanticAIEmbedder: the Embedder port on pydantic-ai's embeddings.

`openai_compatible_embeddings` over an httpx MockTransport drives the real OpenAI `/embeddings`
wire (batching, order, the query prefix, usage, status mapping) with no network.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any

import httpx
import pytest

from llm_ports import ports
from llm_ports.adapters.pydantic_ai import (
    IndexOrderedTransport,
    PydanticAIEmbedder,
    openai_compatible_embeddings,
)
from llm_ports.call.resilience import RetryPolicy
from llm_ports.testing import StubEmbedder

BASE = "http://gw.test/v1"


def _vec(seed: int, dims: int) -> list[float]:
    return [float(seed) + i / 1000 for i in range(dims)]


def _wire(
    calls: list[dict[str, Any]],
    *,
    dims: int = 4,
    status: int = 200,
    drop: int = 0,
    usage: bool = True,
    reverse: bool = False,
) -> httpx.MockTransport:
    """An OpenAI /embeddings server: vector i of a batch = _vec(<global input number>)."""
    seen = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "nope"}})
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = []
        for i, _ in enumerate(inputs[: len(inputs) - drop]):
            vec = _vec(seen["n"] + i, dims)
            emb: Any = vec
            if body.get("encoding_format") == "base64":
                emb = base64.b64encode(struct.pack(f"<{dims}f", *vec)).decode()
            data.append({"object": "embedding", "index": i, "embedding": emb})
        seen["n"] += len(inputs)
        if reverse:  # a server that answers out of order; each item still carries its index
            data.reverse()
        out: dict[str, Any] = {"object": "list", "data": data, "model": body["model"]}
        if usage:
            out["usage"] = {"prompt_tokens": 3 * len(inputs), "total_tokens": 3 * len(inputs)}
        return httpx.Response(200, json=out)

    return httpx.MockTransport(handler)


def _embedder(transport: httpx.MockTransport, **kw: Any) -> PydanticAIEmbedder:
    kw.setdefault("stamp", "box-embed@4")
    kw.setdefault("dims", 4)
    return PydanticAIEmbedder(
        model_for=openai_compatible_embeddings(BASE, "embeddings", "k", transport=transport), **kw
    )


def test_the_shipped_embedders_satisfy_the_port() -> None:
    assert isinstance(_embedder(_wire([])), ports.Embedder)
    assert isinstance(StubEmbedder(), ports.Embedder)


async def test_batches_keep_input_order_and_sum_usage() -> None:
    calls: list[dict[str, Any]] = []
    res = await _embedder(_wire(calls), batch_size=2).embed(["a", "b", "c", "d", "e"])
    assert res.ok and res.stamp == "box-embed@4"
    assert [len(c["input"]) for c in calls] == [2, 2, 1]
    assert [round(v[0]) for v in res.vectors] == [0, 1, 2, 3, 4]
    assert res.input_tokens == 15
    assert {c["model"] for c in calls} == {"embeddings"}


async def test_an_out_of_order_reply_lands_each_vector_on_its_own_input() -> None:
    # MUST FIRE: without IndexOrderedTransport pydantic-ai keeps the listed order and this reads
    # [1, 0, 3, 2, 4] -- every vector on its neighbour's row, the count law still satisfied.
    res = await _embedder(_wire([], reverse=True), batch_size=2).embed(["a", "b", "c", "d", "e"])
    assert res.ok
    assert [round(v[0]) for v in res.vectors] == [0, 1, 2, 3, 4]


async def test_the_index_transport_leaves_what_it_cannot_read_alone() -> None:
    raw = b'{"data": "not a list"}'
    inner = httpx.MockTransport(lambda _r: httpx.Response(200, content=raw))
    async with httpx.AsyncClient(transport=IndexOrderedTransport(inner)) as client:
        resp = await client.post(f"{BASE}/embeddings", json={})
    assert resp.content == raw, "an unreadable reply passes through for the count law to judge"
    inner = httpx.MockTransport(lambda _r: httpx.Response(500, content=raw))
    async with httpx.AsyncClient(transport=IndexOrderedTransport(inner)) as client:
        assert (await client.post(f"{BASE}/embeddings", json={})).status_code == 500


async def test_query_carries_the_instruct_prefix_documents_stay_plain() -> None:
    calls: list[dict[str, Any]] = []
    emb = _embedder(_wire(calls), query_instruct="find emails")
    await emb.embed(["doc"])
    await emb.embed(["what"], input_type="query")
    assert calls[0]["input"] == ["doc"]
    assert calls[1]["input"] == ["Instruct: find emails\nQuery: what"]


async def test_a_wrong_dimension_is_invalid_never_stored() -> None:
    res = await _embedder(_wire([], dims=3)).embed(["a"])
    assert res.outcome == "invalid" and res.vectors == ()
    assert "3-d" in (res.reason or "") and "box-embed@4" in (res.reason or "")


async def test_a_short_batch_is_invalid_not_a_shifted_store() -> None:
    res = await _embedder(_wire([], drop=1)).embed(["a", "b"])
    assert res.outcome == "invalid" and res.vectors == ()


@pytest.mark.parametrize(
    ("status", "outcome"), [(400, "request_fault"), (413, "request_fault"), (500, "unavailable"), (503, "unavailable")]
)
async def test_status_maps_to_a_named_outcome_never_a_raise(status: int, outcome: str) -> None:
    res = await _embedder(_wire([], status=status)).embed(["a"])
    assert res.outcome == outcome and res.vectors == () and res.reason


async def test_a_dead_wire_is_unavailable() -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    res = await _embedder(httpx.MockTransport(boom)).embed(["a"])
    assert res.outcome == "unavailable" and "refused" in (res.reason or "")


async def test_a_gate_refusal_from_the_transport_is_gated() -> None:
    class Closed(Exception):
        pass

    def shut(_req: httpx.Request) -> httpx.Response:
        raise Closed("category off")

    res = await _embedder(httpx.MockTransport(shut), gated=(Closed,)).embed(["a"])
    assert res.outcome == "gated"


async def test_no_usage_on_the_wire_is_unknown_not_zero() -> None:
    res = await _embedder(_wire([], usage=False)).embed(["a"])
    assert res.ok and res.input_tokens is None


async def test_empty_input_is_ok_and_sends_nothing() -> None:
    calls: list[dict[str, Any]] = []
    res = await _embedder(_wire(calls)).embed([])
    assert res.ok and res.vectors == () and calls == []


def test_construction_refuses_an_unnamed_space_or_two_models() -> None:
    with pytest.raises(ValueError, match="stamp"):
        PydanticAIEmbedder(model_for=openai_compatible_embeddings(BASE, "m"), stamp="")
    with pytest.raises(ValueError, match="exactly one"):
        PydanticAIEmbedder(stamp="s")


async def test_stub_is_deterministic_and_can_fail() -> None:
    s = StubEmbedder(dims=6)
    a, b = await s.embed(["x", "y"]), await s.embed(["x"])
    assert a.vectors[0] == b.vectors[0] != a.vectors[1] and len(a.vectors[0]) == 6
    assert (await StubEmbedder(fail="unavailable").embed(["x"])).outcome == "unavailable"


def _flaky(calls: list[int], fail: set[int], status: int = 503) -> httpx.MockTransport:
    """An /embeddings server whose request number n (from 1) fails with `status` when n is in `fail`."""
    ok = _wire([])

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(len(json.loads(req.content)["input"]))
        if len(calls) in fail:
            return httpx.Response(status, json={"error": {"message": "busy"}})
        return ok.handler(req)  # type: ignore[attr-defined]

    return httpx.MockTransport(handler)


NO_WAIT = RetryPolicy(attempts=3, wait_initial_s=0.0, wait_max_s=0.0)


async def test_a_failed_request_is_resent_alone_and_counted() -> None:
    calls: list[int] = []
    res = await _embedder(_flaky(calls, {2}), batch_size=2, retry=NO_WAIT).embed(["a", "b", "c", "d"])
    assert res.ok and len(res.vectors) == 4 and res.retries == 1
    assert calls == [2, 2, 2]  # batch 1 once, batch 2 failed then re-sent; batch 1 is not redone


async def test_retries_run_out_as_unavailable_with_the_count() -> None:
    calls: list[int] = []
    res = await _embedder(_flaky(calls, {1, 2, 3}), retry=NO_WAIT).embed(["a"])
    assert res.outcome == "unavailable" and res.retries == 2 and len(calls) == 3


async def test_a_request_fault_is_never_retried() -> None:
    calls: list[int] = []
    res = await _embedder(_flaky(calls, {1}, status=400), retry=NO_WAIT).embed(["a"])
    assert res.outcome == "request_fault" and res.retries == 0 and calls == [1]


async def test_without_a_policy_a_request_is_sent_once() -> None:
    calls: list[int] = []
    res = await _embedder(_flaky(calls, {1})).embed(["a"])
    assert res.outcome == "unavailable" and res.retries == 0 and calls == [1]
