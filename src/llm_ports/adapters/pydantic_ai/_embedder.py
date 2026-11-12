"""The `Embedder` port implemented on pydantic-ai's embeddings.

pydantic-ai's `OpenAIEmbeddingModel` speaks the wire (`/v1/embeddings` on any OpenAI-compatible
endpoint: a LiteLLM gateway, llama.cpp's server, OpenAI itself). This adapter adds what a vector
store needs and the framework does not promise:

- A stamp: the name of the space every vector lands in (model identity and dimensions), constant
  per embedder.
- Count and dimension checks: one vector per input, in input order, each `dims` long; otherwise
  the whole call is `invalid`.
- Batching (`batch_size` inputs per request) and an asymmetric query prefix (`query_instruct`,
  sent as an `Instruct: {task}` line before `Query: {text}`); documents embed plain.
- Per-request retry (`policy=WirePolicy(...)`, the loop every adapter shares,
  `adapters._retry`): a request whose failure kind the policy retries is re-sent after a backoff,
  and only that request, so a call of many batches does not redo the ones that landed. A spent
  quota, a refused key, an unknown model, a bad request or a bug is never retried. The 1.x
  `retry=RetryPolicy(...)` is the same thing (`RetryPolicy.wire()`). `Embedded.retries` counts
  the re-sends. Without either, each request is sent once (the 1.x default).
- The result contract (ADR-0019): `usage_v2` summed over every batch, `answered`, and `cost`
  (the gateway's report, else the `pricer=`). Over `openai_compatible_embeddings` the wire capture
  reads each reply's usage exactly; over a fixed `model=` pydantic-ai's usage cannot tell an
  absent block from 0, so a batch reporting 0 input tokens reads as unmeasured there.
- Index order: pydantic-ai (2.59, `embeddings/openai.py`) reads a reply's `data` in listed order
  and drops each item's `index`, so a server that answers out of order would put vectors on the
  wrong rows while the count check still passes. `IndexOrderedTransport` (`llm_ports.adapters.http`,
  shared with the rerank wire) re-sorts `data` by `index` beneath the SDK.
  `openai_compatible_embeddings` always installs the same re-sort under its httpx2 client; a caller
  that builds its own httpx provider wraps its transport in `IndexOrderedTransport`.

The model source is the caller's, as for the generator:

- `model=`: a fixed pydantic-ai `EmbeddingModel` (tests pass `TestEmbeddingModel`).
- `model_for=`: `() -> async context manager yielding an EmbeddingModel`, called per request, so
  a caller can put the model on its own transport (admission control, leases, a work queue).
- `openai_compatible_embeddings(base_url, model_name, ...)` builds a `model_for` for any
  OpenAI-wire endpoint.

Invariants:
    - A model, wire or gate failure never raises; `classify_failure` names the `Failure`, with one
      table for the generator, the embedder and the reranker.

Rationale: ADR-0001, ADR-0019, ADR-0020, ADR-0021; the httpx2 client, ADR-0009.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import replace
from typing import Any

from llm_ports._extras import extra
from llm_ports._headers import HEADER_VALUE_RULE, header_value_ok
from llm_ports.adapters._classify import WireFacts
from llm_ports.adapters._retry import WireRetry
from llm_ports.adapters.pydantic_ai._capture import capture, record_response
from llm_ports.adapters.pydantic_ai._classify import classify_failure
from llm_ports.adapters.pydantic_ai._result import gateway_cost, gateway_deployment, price
from llm_ports.call.resilience import RetryPolicy
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import (
    Answered,
    Embedded,
    EmbedInput,
    Failure,
    GatewayReader,
    Pricer,
    Usage,
    WirePolicy,
    failure,
)
from llm_ports.ports.outcomes import exception_chain, failure_reason

with extra("pydantic-ai"):
    import httpx
    import httpx2
    from pydantic_ai.embeddings import EmbeddingModel

    from llm_ports.adapters._index_order import IndexOrderedTransport
    from llm_ports.adapters.pydantic_ai._resilient import judge_for
    from llm_ports.adapters.pydantic_ai._wire import IndexOrdered, as_httpx2

EmbedModelFor = Callable[[], AbstractAsyncContextManager[EmbeddingModel]]

#: The cap on an `unavailable` reason that carries the wrapper's reason and the chain root's.
_CHAINED_REASON_MAX = 240


def openai_compatible_embeddings(
    base_url: str,
    model_name: str,
    api_key: str | None = None,
    *,
    timeout_s: float = 120.0,
    max_retries: int = 0,
    transport: httpx2.AsyncBaseTransport | httpx.AsyncBaseTransport | None = None,
) -> EmbedModelFor:
    """Build a `model_for` over any OpenAI-wire `/embeddings` endpoint.

    Each call opens one `httpx2.AsyncClient` (the client pydantic-ai's OpenAI provider takes).

    Args:
        base_url: The URL up to where `/embeddings` starts, e.g. `http://localhost:4000/v1`.
        model_name: The embedding model the endpoint serves.
        api_key: The bearer key; None sends a placeholder, for servers that need no key.
        timeout_s: The client timeout per call.
        max_retries: The OpenAI SDK's own retry count. Defaults to 0, so retries are not stacked
            on those of a gateway in front.
        transport: Replaces httpx2's own transport (a caller's admitting or counting transport, or
            a `MockTransport` in a test). An httpx2 transport is used as is; a legacy httpx one is
            carried across a bridge. It always sits under the index-order re-sort.

    Returns:
        An `EmbedModelFor` that yields an `OpenAIEmbeddingModel`.

    Raises:
        ValueError: `api_key` holds whitespace, a control or a non-ASCII character. The value is
            not shown.

    Rationale: ADR-0009.
    """
    if api_key is not None and not header_value_ok(api_key):
        raise ValueError(f"api_key {HEADER_VALUE_RULE}")
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

    @asynccontextmanager
    async def model_for() -> AsyncIterator[EmbeddingModel]:
        ordered = IndexOrdered(as_httpx2(transport))
        async with httpx2.AsyncClient(
            timeout=timeout_s, transport=ordered, event_hooks={"response": [record_response]}
        ) as client:
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key", http_client=client)
            provider = OpenAIProvider(openai_client=provider.client.with_options(max_retries=max_retries))
            yield OpenAIEmbeddingModel(model_name, provider=provider)

    return model_for


class PydanticAIEmbedder:
    """The `Embedder` port on pydantic-ai; the module docstring states the contract.

    Args:
        model: A fixed pydantic-ai `EmbeddingModel`.
        stamp: The name of the space the vectors live in; an unnamed space cannot be stored.
        dims: When given, every vector must be this long, or the call is `invalid`.
        model_for: A per-call model factory.
        batch_size: Inputs per request.
        query_instruct: The task line prefixed to `input_type="query"` inputs; empty sends queries
            plain.
        gated: Exception types that classify as `gated`.
        retry: Deprecated: `policy=retry.wire()` is the same. Pass at most one of the two.
        policy: Which failed requests are re-sent, and how; None (and no `retry`) sends each
            request once.
        pricer: Prices a call the gateway did not price; None leaves `cost` None unless a gateway
            reported one.
        gateway: Reads a gateway's response headers (e.g. the module `adapters.litellm_proxy`).
        on_event: Receives `llm_pricer_failed`, `llm_gateway_header_invalid` and `llm_wire_retry`;
            without one they go to the stdlib logger `llm_ports.adapters.pydantic_ai`.

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given, both `retry` and
            `policy` are given, `stamp` is empty, or `batch_size` is below 1.
    """

    def __init__(
        self,
        model: EmbeddingModel | None = None,
        *,
        stamp: str,
        dims: int | None = None,
        model_for: EmbedModelFor | None = None,
        batch_size: int = 16,
        query_instruct: str = "",
        gated: tuple[type[BaseException], ...] = (),
        retry: RetryPolicy | None = None,
        pricer: Pricer | None = None,
        gateway: GatewayReader | None = None,
        on_event: EventHook | None = None,
        policy: WirePolicy | None = None,
    ) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticAIEmbedder: pass exactly one of model= or model_for=")
        if retry is not None and policy is not None:
            raise ValueError("PydanticAIEmbedder: pass at most one of retry= (deprecated) or policy=")
        if not stamp:
            raise ValueError("PydanticAIEmbedder: a stamp is required -- an unnamed space cannot be stored")
        if batch_size < 1:
            raise ValueError("PydanticAIEmbedder: batch_size must be >= 1")
        self._model = model
        self._model_for = model_for
        self._stamp = stamp
        self._dims = dims
        self._batch = batch_size
        self._instruct = query_instruct
        self._gated = gated
        self._policy = policy if policy is not None else (retry.wire() if retry is not None else WirePolicy.none())
        self._pricer = pricer
        self._gateway = gateway
        self._emit: EventHook = on_event or log_events("llm_ports.adapters.pydantic_ai")

    @property
    def stamp(self) -> str:
        """The name of the space this embedder's vectors live in."""
        return self._stamp

    def _inputs(self, texts: Sequence[str], input_type: EmbedInput) -> list[str]:
        if input_type == "query" and self._instruct:
            return [f"Instruct: {self._instruct}\nQuery: {t}" for t in texts]
        return list(texts)

    async def _embed_on(
        self, model: EmbeddingModel, inputs: list[str], input_type: EmbedInput, tally: _Tally
    ) -> list[tuple[float, ...]]:
        from pydantic_ai.embeddings import Embedder as _PaiEmbedder

        tally.requested = getattr(model, "model_name", "") or self._stamp
        embedder = _PaiEmbedder(model)
        out: list[tuple[float, ...]] = []
        for i in range(0, len(inputs), self._batch):
            batch = inputs[i : i + self._batch]
            res = await self._send(embedder, batch, input_type, tally)
            tally.framework.append(_framework_usage(res))
            tally.model = getattr(res, "model_name", None) or tally.model
            tally.response_id = getattr(res, "provider_response_id", None) or tally.response_id
            if len(res.embeddings) != len(batch):
                raise _Invalid(f"sent {len(batch)} inputs, got {len(res.embeddings)} vectors")
            for vec in res.embeddings:
                if self._dims is not None and len(vec) != self._dims:
                    raise _Invalid(f"vector is {len(vec)}-d, the {self._stamp} space is {self._dims}-d")
                out.append(tuple(float(x) for x in vec))
        return out

    async def _send(self, embedder: Any, batch: list[str], input_type: EmbedInput, tally: _Tally) -> Any:
        """One request, re-sent while the policy retries its failure (the shared loop)."""
        wire = WireRetry(self._policy, model=tally.requested, deadline_at=None, emit=self._emit, label=self._stamp)
        cap = tally.cap

        async def send() -> Any:
            cap.last = None  # this try's facts only
            return await embedder.embed(batch, input_type=input_type)

        try:
            return await wire.run(send, judge_for(cap, extra=(), gated=self._gated))
        finally:
            tally.retries += len(wire.attempts)

    async def embed(self, texts: Sequence[str], *, input_type: EmbedInput = "document") -> Embedded:
        """Embed `texts` in order; a model, wire or gate failure returns a named outcome, never raises.

        `input_type="query"` applies `query_instruct`. Empty `texts` returns `ok` with no vectors
        and zero input tokens. A reply that breaks the count or dimension check is `invalid`. A
        failure the generator would call `invalid_output`, `truncated` or `refused` is `server`
        (`unavailable`) here, since an embeddings reply has no parse step, token ceiling or
        refusal; its reason ends with the root cause of the exception chain (refused, timed out,
        DNS).

        Returns:
            An `Embedded` stamped with this embedder's space. `input_tokens` is None when any
            batch did not report usage, never 0.

        Rationale: ADR-0001, ADR-0007, ADR-0019.
        """
        if not texts:
            return Embedded(vectors=(), outcome="ok", stamp=self._stamp, input_tokens=0)
        inputs = self._inputs(texts, input_type)
        tally = _Tally()
        with capture() as cap:
            tally.cap = cap
            try:
                if self._model is not None:
                    vecs = await self._embed_on(self._model, inputs, input_type, tally)
                else:
                    assert self._model_for is not None
                    async with self._model_for() as model:
                        vecs = await self._embed_on(model, inputs, input_type, tally)
            except _Invalid as exc:
                return self._done((), failure("invalid_output", reason=str(exc)), tally, cap.replies)
            except Exception as exc:  # the port never raises for a model, wire or gate failure
                return self._done((), self._failure(exc, cap.last), tally, cap.replies)
        return self._done(tuple(vecs), None, tally, cap.replies)

    def _failure(self, exc: BaseException, last: WireFacts | None) -> Failure:
        got = classify_failure(exc, last or WireFacts(), gated=self._gated)
        assert got is not None  # an exception always classifies
        if got.kind in ("invalid_output", "truncated", "refused"):
            got = failure("server", status=got.status, reason=got.reason)  # no parse step, ceiling or refusal
        root = list(exception_chain(exc))[-1]
        if got.outcome == "unavailable" and root is not exc:
            # pydantic-ai -> openai -> httpx2: the wrappers say "Connection error."; the root
            # of the chain names the cause (refused, timed out, DNS).
            reason = f"{got.reason} <- {failure_reason(root)}"[:_CHAINED_REASON_MAX]
            got = replace(got, reason=reason)
        return got

    def _done(
        self, vecs: tuple[tuple[float, ...], ...], got: Failure | None, tally: _Tally, replies: Sequence[Any]
    ) -> Embedded:
        """Assemble the result contract for one `embed` call."""
        if replies:
            parts = [
                r.usage if r.usage.input is None or r.usage.output is not None else replace(r.usage, output=0)
                for r in replies
            ]
        else:
            parts = tally.framework
        usage: Usage | None = None
        for part in parts:
            usage = part if usage is None else usage + part
        last = replies[-1] if replies else None
        answered = Answered(
            requested=tally.requested or self._stamp,
            model=(last.model if last is not None else None) or tally.model,
            deployment=gateway_deployment(self._gateway, replies, self._emit),
            response_id=(last.response_id if last is not None else None) or tally.response_id,
        )
        gw = gateway_cost(self._gateway, replies, self._emit)
        cost = price(usage, answered, gateway_usd=gw, pricer=self._pricer, emit=self._emit)
        tokens = usage.input if usage is not None and usage.unmeasured == 0 else None
        return Embedded(
            vectors=vecs,
            outcome=got.outcome if got is not None else "ok",
            stamp=self._stamp,
            reason=got.reason if got is not None else None,
            input_tokens=tokens if got is None else None,
            retries=tally.retries,
            usage_v2=usage,
            cost=cost,
            answered=answered,
            failure=got,
        )


def _framework_usage(res: Any) -> Usage:
    """One batch's usage as pydantic-ai recorded it.

    Embeddings produce no output tokens, so `output` is 0 whenever the input was measured.
    pydantic-ai's embeddings usage cannot tell an absent block from 0, so 0 reads as unmeasured.
    """
    used = getattr(getattr(res, "usage", None), "input_tokens", None)
    if not isinstance(used, int) or used <= 0:
        return Usage(requests=1, unmeasured=1)
    return Usage(input=used, output=0, requests=1)


class _Invalid(Exception):
    """A reply that broke the count or dimension check."""


class _Tally:
    """One `embed` call's re-sends and what each batch reported (per call, never shared)."""

    __slots__ = ("cap", "framework", "model", "requested", "response_id", "retries")

    def __init__(self) -> None:
        self.cap: Any = None
        self.retries = 0
        self.framework: list[Usage] = []
        self.requested: str = ""
        self.model: str | None = None
        self.response_id: str | None = None


__all__ = ["EmbedModelFor", "IndexOrderedTransport", "PydanticAIEmbedder", "openai_compatible_embeddings"]
