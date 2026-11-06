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
- Per-request retry (`retry=RetryPolicy(...)`, tenacity underneath): a request that fails
  `unavailable` (timed out, refused, 5xx) is re-sent after a backoff, and only that request, so a
  call of many batches does not redo the ones that landed. A `request_fault` or `gated` failure is
  never retried. `Embedded.retries` counts the re-sends.
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
    - A model, wire or gate failure never raises; `classify` names the outcome, with one
      taxonomy for the generator and the embedder.

Rationale: ADR-0001; the httpx2 client, ADR-0009.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters.pydantic_ai._generator import classify
from llm_ports.call.resilience import RetryPolicy
from llm_ports.ports import Embedded, EmbedInput
from llm_ports.ports.outcomes import exception_chain, failure_reason

with extra("pydantic-ai"):
    import httpx
    import httpx2
    import tenacity
    from pydantic_ai.embeddings import EmbeddingModel

    from llm_ports.adapters._index_order import IndexOrderedTransport
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

    Rationale: ADR-0009.
    """
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

    @asynccontextmanager
    async def model_for() -> AsyncIterator[EmbeddingModel]:
        ordered = IndexOrdered(as_httpx2(transport))
        async with httpx2.AsyncClient(timeout=timeout_s, transport=ordered) as client:
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
        retry: Re-send a request that failed `unavailable`; None sends each request once.

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given, `stamp` is empty,
            or `batch_size` is below 1.
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
    ) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticAIEmbedder: pass exactly one of model= or model_for=")
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
        self._retry = retry

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
    ) -> tuple[list[tuple[float, ...]], int | None]:
        from pydantic_ai.embeddings import Embedder as _PaiEmbedder

        embedder = _PaiEmbedder(model)
        out: list[tuple[float, ...]] = []
        tokens: int | None = 0
        for i in range(0, len(inputs), self._batch):
            batch = inputs[i : i + self._batch]
            res = await self._send(embedder, batch, input_type, tally)
            if len(res.embeddings) != len(batch):
                raise _Invalid(f"sent {len(batch)} inputs, got {len(res.embeddings)} vectors")
            for vec in res.embeddings:
                if self._dims is not None and len(vec) != self._dims:
                    raise _Invalid(f"vector is {len(vec)}-d, the {self._stamp} space is {self._dims}-d")
                out.append(tuple(float(x) for x in vec))
            used = getattr(res.usage, "input_tokens", None)
            tokens = None if (tokens is None or not used) else tokens + int(used)
        return out, tokens

    def _transient(self, exc: BaseException) -> bool:
        return isinstance(exc, Exception) and classify(exc, gated=self._gated)[0] == "unavailable"

    async def _send(self, embedder: Any, batch: list[str], input_type: EmbedInput, tally: _Tally) -> Any:
        """One request, re-sent per `self._retry` while it fails `unavailable`."""
        if self._retry is None or self._retry.attempts == 1:
            return await embedder.embed(batch, input_type=input_type)
        p = self._retry
        retrying = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(p.attempts),
            wait=tenacity.wait_exponential(multiplier=p.wait_initial_s, max=p.wait_max_s),
            retry=tenacity.retry_if_exception(self._transient),
            reraise=True,
        )
        async for attempt in retrying:
            if attempt.retry_state.attempt_number > 1:
                tally.retries += 1
            with attempt:
                return await embedder.embed(batch, input_type=input_type)
        raise AssertionError("unreachable: tenacity re-raises the last failure")  # pragma: no cover

    async def embed(self, texts: Sequence[str], *, input_type: EmbedInput = "document") -> Embedded:
        """Embed `texts` in order; a model, wire or gate failure returns a named outcome, never raises.

        `input_type="query"` applies `query_instruct`. Empty `texts` returns `ok` with no vectors
        and zero input tokens. A reply that breaks the count or dimension check is `invalid`. A
        failure the generator would call `invalid` or `truncated` is `unavailable` here, since an
        embeddings reply has no parse step and no token ceiling; its reason ends with the root
        cause of the exception chain (refused, timed out, DNS).

        Returns:
            An `Embedded` stamped with this embedder's space. `input_tokens` is None when any
            batch did not report usage, never 0.

        Rationale: ADR-0001, ADR-0007.
        """
        if not texts:
            return Embedded(vectors=(), outcome="ok", stamp=self._stamp, input_tokens=0)
        inputs = self._inputs(texts, input_type)
        tally = _Tally()
        try:
            if self._model is not None:
                vecs, tokens = await self._embed_on(self._model, inputs, input_type, tally)
            else:
                assert self._model_for is not None
                async with self._model_for() as model:
                    vecs, tokens = await self._embed_on(model, inputs, input_type, tally)
        except _Invalid as exc:
            return Embedded(vectors=(), outcome="invalid", stamp=self._stamp, reason=str(exc), retries=tally.retries)
        except Exception as exc:  # the port never raises for a model, wire or gate failure
            outcome, reason = classify(exc, gated=self._gated)
            if outcome == "invalid" or outcome == "truncated":
                outcome = "unavailable"  # an embeddings reply has no parse step or token ceiling
            root = list(exception_chain(exc))[-1]
            if outcome == "unavailable" and root is not exc:
                # pydantic-ai -> openai -> httpx2: the wrappers say "Connection error."; the root
                # of the chain names the cause (refused, timed out, DNS).
                reason = f"{reason} <- {failure_reason(root)}"[:_CHAINED_REASON_MAX]
            return Embedded(vectors=(), outcome=outcome, stamp=self._stamp, reason=reason, retries=tally.retries)
        return Embedded(
            vectors=tuple(vecs), outcome="ok", stamp=self._stamp, input_tokens=tokens, retries=tally.retries
        )


class _Invalid(Exception):
    """A reply that broke the count or dimension check."""


class _Tally:
    """One `embed` call's re-sent requests (per call, so concurrent calls never share a count)."""

    __slots__ = ("retries",)

    def __init__(self) -> None:
        self.retries = 0


__all__ = ["EmbedModelFor", "IndexOrderedTransport", "PydanticAIEmbedder", "openai_compatible_embeddings"]
