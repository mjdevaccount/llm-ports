"""The `Reranker` port over the Cohere/Jina-shaped `/rerank` wire.

The wire is `POST {url}` with `{model, query, documents}` -> `{results: [{index, relevance_score}]}`,
which llama.cpp's server, LiteLLM, vLLM, Jina and Cohere all speak. A reply may list its results
in score order or any other order; `IndexOrderedTransport` puts them back in input order under the
client. It is the same transport that orders an `/embeddings` reply, so the index is read in one
place.

On a failure the caller keeps the order it already had: a reranker that is down degrades ranking,
never search.

Invariants:
    - An `ok` reply scores every document exactly once: its indices are exactly 0..n-1, so a
      duplicate index cannot pass a count check and leave one document unscored.
    - Every `ok` score is a finite number.
    - A reply that breaks either rule is `invalid`, never a partial ranking.
    - A call never raises for a model, wire or gate failure. Every failure is a `Failure` from the
      shared classification table (`adapters._classify`), the one the generator and the embedder
      use: a transport error is `transport` or `timeout`, a status maps by `kind_for_status`, a
      context overflow is `context_overflow` whatever its status.
    - `answered` names the model the reply reported, when it reported one.

Retry: `policy=WirePolicy(...)` re-sends a request whose failure kind the policy retries, through
the loop every adapter shares (`adapters._retry`, the generator's and the embedder's); each
re-sent request is an `Attempt` on the result. None (the default) sends one request, as 1.x did.

Rationale: ADR-0001, ADR-0020, ADR-0021.

Example:
    reranker = HttpReranker("http://localhost:8080/v1/rerank", "bge-reranker-v2-m3")
    res = await reranker.rerank("late shipment refund", [doc.text for doc in docs])
    if res.ok:
        ranked = sorted(zip(res.scores, docs), key=lambda t: -t[0])
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters._classify import ERROR_MESSAGE_MAX, WireFacts, classify_core
from llm_ports.adapters._retry import Judged, WireRetry
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import Answered, Failure, Reranked, WirePolicy, failure
from llm_ports.ports.outcomes import REASON_MAX

with extra("http"):
    import httpx

    from llm_ports.adapters._index_order import IndexOrderedTransport

#: The per-call timeout when the caller names none, in seconds.
DEFAULT_TIMEOUT_S = 30.0


def payload(model: str, query: str, documents: Sequence[str]) -> dict[str, Any]:
    """Return the `/rerank` request body."""
    return {"model": model, "query": query, "documents": list(documents)}


def scores_from(body: Any, n: int) -> tuple[float, ...]:
    """Return the scores of an index-ordered reply body for `n` documents, in input order.

    Pure. `body` is what `IndexOrderedTransport` hands up.

    Raises:
        ValueError: The body is unreadable, or breaks a rule (score count, index set, index order,
            non-finite score); the message names which.
    """
    try:
        results = body["results"]
        indices = [int(r["index"]) for r in results]
        scores = tuple(float(r["relevance_score"]) for r in results)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unreadable rerank reply: {type(exc).__name__}: {exc}"[:REASON_MAX]) from exc
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
    """A `Reranker` over one `/rerank` endpoint.

    Each call opens its own httpx client, and keeps it across the call's retries.

    Args:
        url: The full `/rerank` endpoint URL.
        model: The model name sent on every request and reported on every result.
        api_key: Sent as a bearer token when set.
        timeout_s: The per-call httpx timeout, in seconds.
        transport: Replaces httpx's own transport (a consumer's counting transport, or
            `httpx.MockTransport` in a test). It always sits under `IndexOrderedTransport`.
        policy: Which failed requests are re-sent, and how; None sends each request once.
        on_event: Receives `llm_wire_retry`; without one it goes to the stdlib logger
            `llm_ports.adapters.http`.

    Raises:
        ValueError: At construction, when `url` or `model` is empty.
    """

    def __init__(
        self,
        url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        policy: WirePolicy | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        if not url or not model:
            raise ValueError("HttpReranker: url and model are required")
        self.url = url
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout_s = timeout_s
        self._transport = transport
        self._policy = policy if policy is not None else WirePolicy.none()
        self._emit: EventHook = on_event or log_events("llm_ports.adapters.http")

    def _failed(self, got: Failure, answered: Answered) -> Reranked:
        return Reranked(
            scores=(), outcome=got.outcome, model=self.model, reason=got.reason, answered=answered, failure=got
        )

    async def rerank(self, query: str, documents: Sequence[str]) -> Reranked:
        """Score `documents` against `query`, in input order.

        An empty `documents` returns `ok` with no scores and sends no request. A failure is a named
        outcome, never an exception (see the module invariants).
        """
        asked = Answered(requested=self.model)
        if not documents:
            return Reranked(scores=(), outcome="ok", model=self.model, answered=asked)
        wire = WireRetry(self._policy, model=self.model, deadline_at=None, emit=self._emit, label="rerank")
        seen: list[Mapping[str, str]] = [{}]  # the latest request's response headers

        def judge(res: Any) -> Judged:
            return (res.failure, seen[0]) if isinstance(res, Reranked) else (None, {})

        async with httpx.AsyncClient(
            timeout=self._timeout_s, transport=IndexOrderedTransport(self._transport)
        ) as client:

            async def send() -> Reranked:
                res, seen[0] = await self._once(client, query, documents, asked)
                return res

            res = await wire.run(send, judge)
        return replace(res, attempts=tuple(wire.attempts)) if wire.attempts else res

    async def _once(
        self, client: httpx.AsyncClient, query: str, documents: Sequence[str], asked: Answered
    ) -> tuple[Reranked, Mapping[str, str]]:
        """One wire request: the result, and the response headers (empty without a response)."""
        try:
            r = await client.post(self.url, json=payload(self.model, query, documents), headers=self._headers)
            body = r.json() if r.status_code == 200 else _error_body(r)
        except httpx.HTTPError as exc:
            got = classify_core(exc, WireFacts())
            assert got is not None  # an exception always classifies
            return self._failed(replace(got, reason=f"rerank transport: {got.reason}"[:REASON_MAX]), asked), {}
        except ValueError as exc:  # a 200 whose body is not JSON
            return self._failed(failure("invalid_output", reason=f"rerank reply is not JSON: {exc}"), asked), {}
        headers = {k.lower(): v for k, v in r.headers.items()}
        if r.status_code != 200:
            got = classify_core(None, _facts(r.status_code, headers, body))
            assert got is not None  # a non-200 status always classifies
            return self._failed(replace(got, reason=f"rerank HTTP {r.status_code}"), asked), headers
        return self._scored(body, len(documents)), headers

    def _scored(self, body: Any, n: int) -> Reranked:
        answered = _answered(self.model, body)
        try:
            scores = scores_from(body, n)
        except ValueError as exc:
            return self._failed(failure("invalid_output", reason=str(exc)), answered)
        return Reranked(scores=scores, outcome="ok", model=self.model, answered=answered)


def _error_body(r: httpx.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return None


def _facts(status: int, headers: dict[str, str], body: Any) -> WireFacts:
    """The facts of a non-200 reply: its status, headers and the provider's error fields, no body."""
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, str):
        err = {"message": err}
    if not isinstance(err, dict):
        err = {}
    message = err.get("message")
    code, kind = err.get("code"), err.get("type")
    return WireFacts(
        status=status,
        headers=headers,
        error_code=str(code) if code is not None else None,
        error_type=kind if isinstance(kind, str) else None,
        error_message=message[:ERROR_MESSAGE_MAX] if isinstance(message, str) else None,
    )


def _answered(requested: str, body: Any) -> Answered:
    model = body.get("model") if isinstance(body, dict) else None
    rid = body.get("id") if isinstance(body, dict) else None
    return Answered(
        requested=requested,
        model=model if isinstance(model, str) else None,
        response_id=rid if isinstance(rid, str) else None,
    )


__all__ = ["DEFAULT_TIMEOUT_S", "HttpReranker", "payload", "scores_from"]
