"""HttpReranker -- the `Reranker` port over the `/rerank` wire: index order, the laws, named outcomes."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_ports import ports
from llm_ports.adapters.http.index_order import IndexOrderedTransport
from llm_ports.adapters.http.reranker import HttpReranker, payload, scores_from
from llm_ports.testing import StubReranker

URL = "http://rerank.test/v1/rerank"
DOCS = ["alpha", "beta", "gamma"]


def _server(results, *, status: int = 200, seen: list | None = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if isinstance(results, Exception):
            raise results
        if isinstance(results, bytes):
            return httpx.Response(status, content=results)
        return httpx.Response(status, json={"results": results})

    return httpx.MockTransport(handle)


def _rr(results, **kw) -> HttpReranker:
    return HttpReranker(URL, "rr-model", transport=_server(results, **kw))


async def test_a_score_ordered_reply_lands_on_its_input_document() -> None:
    # MUST FIRE: a reply listed by SCORE (2, 0, 1) read as listed puts gamma's score on alpha.
    # Without IndexOrderedTransport's `/rerank` entry this is "reply is not in index order".
    res = await _rr(
        [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": -3.2},
        ]
    ).rerank("q", DOCS)
    assert res.ok and res.scores == (0.5, -3.2, 0.9) and res.model == "rr-model"


async def test_the_request_is_the_cohere_jina_shape_with_the_bearer_key() -> None:
    seen: list[httpx.Request] = []
    rr = HttpReranker(
        URL, "m", api_key="k", transport=_server([{"index": i, "relevance_score": 1.0} for i in range(3)], seen=seen)
    )
    await rr.rerank("what moved", DOCS)
    (req,) = seen
    assert json.loads(req.content) == {"model": "m", "query": "what moved", "documents": DOCS}
    assert req.headers["authorization"] == "Bearer k"
    assert payload("m", "q", ("a",)) == {"model": "m", "query": "q", "documents": ["a"]}


async def test_no_documents_is_an_ok_with_no_scores_and_no_call() -> None:
    seen: list = []
    res = await HttpReranker(URL, "m", transport=_server([], seen=seen)).rerank("q", [])
    assert res.ok and res.scores == () and seen == []


@pytest.mark.parametrize(
    ("results", "why"),
    [
        ([{"index": 0, "relevance_score": 1.0}, {"index": 1, "relevance_score": 1.0}], "sent 3 documents, got 2"),
        # MUST FIRE: a duplicate index passes a count check and leaves document 2 unscored.
        (
            [
                {"index": 0, "relevance_score": 1.0},
                {"index": 0, "relevance_score": 2.0},
                {"index": 1, "relevance_score": 1.0},
            ],
            "not exactly 0..2",
        ),
        (
            [
                {"index": 0, "relevance_score": 1.0},
                {"index": 1, "relevance_score": 1.0},
                {"index": 5, "relevance_score": 1.0},
            ],
            "not exactly 0..2",
        ),
        (
            [
                {"index": 0, "relevance_score": 1.0},
                {"index": 1, "relevance_score": "NaN"},
                {"index": 2, "relevance_score": 1.0},
            ],
            "not a finite number",
        ),
        ([{"index": 0}], "unreadable rerank reply"),
    ],
    ids=["short", "duplicate", "out_of_range", "nan", "no_score"],
)
async def test_a_reply_that_breaks_a_law_is_invalid_never_partial(results, why) -> None:
    res = await _rr(results).rerank("q", DOCS)
    assert res.outcome == "invalid" and res.scores == () and why in (res.reason or "")


async def test_a_reply_that_is_not_json_is_invalid() -> None:
    res = await _rr(b"<html>oops</html>").rerank("q", DOCS)
    assert res.outcome == "invalid" and "not JSON" in (res.reason or "")


@pytest.mark.parametrize(
    ("status", "outcome"), [(500, "unavailable"), (503, "unavailable"), (422, "request_fault"), (413, "request_fault")]
)
async def test_a_non_200_is_named_never_raised(status: int, outcome: str) -> None:
    res = await _rr([], status=status).rerank("q", DOCS)
    assert res.outcome == outcome and res.reason == f"rerank HTTP {status}" and res.scores == ()


async def test_a_transport_failure_is_unavailable_never_raised() -> None:
    res = await _rr(httpx.ConnectError("refused")).rerank("q", DOCS)
    assert res.outcome == "unavailable" and "ConnectError" in (res.reason or "")


def test_scores_from_refuses_an_unordered_body_by_name() -> None:
    body = {"results": [{"index": 1, "relevance_score": 1.0}, {"index": 0, "relevance_score": 2.0}]}
    with pytest.raises(ValueError, match="not in index order"):
        scores_from(body, 2)


async def test_the_transport_orders_rerank_results_and_leaves_other_paths_alone() -> None:
    body = {"results": [{"index": 1, "relevance_score": 1.0}, {"index": 0, "relevance_score": 2.0}]}
    inner = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    async with httpx.AsyncClient(transport=IndexOrderedTransport(inner)) as c:
        assert [r["index"] for r in (await c.post("http://x/v1/rerank")).json()["results"]] == [0, 1]
        assert [r["index"] for r in (await c.post("http://x/v1/other")).json()["results"]] == [1, 0]


def test_construction_needs_a_url_and_a_model() -> None:
    with pytest.raises(ValueError):
        HttpReranker("", "m")
    with pytest.raises(ValueError):
        HttpReranker(URL, "")


async def test_the_shipped_rerankers_satisfy_the_port() -> None:
    assert isinstance(HttpReranker(URL, "m"), ports.Reranker)
    assert isinstance(StubReranker(), ports.Reranker)
    res = await StubReranker().rerank("spy gamma", ["spy gamma flip", "weather", "gamma"])
    assert res.scores == (2.0, 0.0, 1.0)
    assert (await StubReranker(fail="unavailable").rerank("q", ["d"])).outcome == "unavailable"
