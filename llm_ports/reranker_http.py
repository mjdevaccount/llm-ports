"""HttpReranker -- the `Reranker` port over the Cohere/Jina-shaped `/rerank` wire.

    reranker = HttpReranker("http://localhost:8080/v1/rerank", "bge-reranker-v2-m3")
    res = await reranker.rerank("what moved the index", [doc.text for doc in docs])
    if res.ok:
        ranked = sorted(zip(res.scores, docs), key=lambda t: -t[0])

The wire is `POST {url}` with `{model, query, documents}` -> `{results: [{index, relevance_score}]}`,
which llama.cpp's server, LiteLLM, vLLM, Jina and Cohere all speak. A reply lists results in SCORE
order (or any order); `IndexOrderedTransport` puts them back in input order under the client --
the same transport that orders an `/embeddings` reply, so the index is read in one place.

THE LAWS, each `invalid` when broken, never a partial ranking: the reply scores every document
exactly once (indices are exactly 0..n-1 -- a duplicate index would otherwise pass a count check
and leave one document unscored), and every score is a finite number (a NaN sorts nowhere).

NEVER RAISES for a model, wire or gate failure: a transport error is `unavailable`, a refused
request (400/413/422) is `request_fault`, any other non-200 is `unavailable`. The caller's
fallback is the order it already had; a reranker that is down degrades ranking, never search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from llm_ports.failure import failure_reason
from llm_ports.index_order import IndexOrderedTransport
from llm_ports.ports import Reranked
from llm_ports.pydantic_ai_generator import REQUEST_FAULT


def payload(model: str, query: str, documents: Sequence[str]) -> dict[str, Any]:
    """The request body (the shape every `/rerank` server above accepts)."""
    return {"model": model, "query": query, "documents": list(documents)}


def scores_from(body: Any, n: int) -> tuple[float, ...]:
    """Pure. The scores of an INDEX-ORDERED reply body (what `IndexOrderedTransport` hands up) for
    `n` documents, in input order. Raises
    ValueError naming the broken law (count, index set, non-finite score) or the unreadable shape."""
    try:
        results = body["results"]
        indices = [int(r["index"]) for r in results]
        scores = tuple(float(r["relevance_score"]) for r in results)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unreadable rerank reply: {type(exc).__name__}: {exc}"[:200]) from exc
    if len(indices) != n:
        raise ValueError(f"sent {n} documents, got {len(indices)} scores")
    if sorted(indices) != list(range(n)):
        raise ValueError(f"reply indices are not exactly 0..{n - 1}: {sorted(indices)[:8]}")
    if indices != list(range(n)):
        raise ValueError("reply is not in index order (read it through IndexOrderedTransport)")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("a score is not a finite number")
    return scores


class HttpReranker:
    """`Reranker` over one `/rerank` endpoint. One httpx client per call; `transport` replaces
    httpx's own (a consumer's counting transport, or `httpx.MockTransport` in a test) and always
    sits under `IndexOrderedTransport`."""

    def __init__(self, url: str, model: str, *, api_key: str | None = None, timeout_s: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not url or not model:
            raise ValueError("HttpReranker: url and model are required")
        self.url = url
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout_s = timeout_s
        self._transport = transport

    def _failed(self, outcome: str, reason: str) -> Reranked:
        return Reranked(scores=(), outcome=outcome, model=self.model, reason=reason[:200])  # type: ignore[arg-type]

    async def rerank(self, query: str, documents: Sequence[str]) -> Reranked:
        if not documents:
            return Reranked(scores=(), outcome="ok", model=self.model)
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s,
                                         transport=IndexOrderedTransport(self._transport)) as client:
                r = await client.post(self.url, json=payload(self.model, query, documents),
                                      headers=self._headers)
                body = r.json() if r.status_code == 200 else None
        except httpx.HTTPError as exc:
            return self._failed("unavailable", f"rerank transport: {failure_reason(exc)}")
        except ValueError as exc:             # a 200 whose body is not JSON
            return self._failed("invalid", f"rerank reply is not JSON: {failure_reason(exc)}")
        if r.status_code != 200:
            outcome = "request_fault" if r.status_code in REQUEST_FAULT else "unavailable"
            return self._failed(outcome, f"rerank HTTP {r.status_code}")
        try:
            scores = scores_from(body, len(documents))
        except ValueError as exc:
            return self._failed("invalid", str(exc))
        return Reranked(scores=scores, outcome="ok", model=self.model)


__all__ = ["HttpReranker", "payload", "scores_from"]
