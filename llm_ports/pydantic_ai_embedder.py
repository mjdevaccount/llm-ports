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
- Index order: pydantic-ai (2.59, `embeddings/openai.py`) reads a reply's `data` in listed order
  and drops each item's `index`, so a server that answers out of order would put vectors on the
  wrong rows while the count check still passes. `IndexOrderedTransport` (`llm_ports.index_order`,
  shared with the rerank wire) re-sorts `data` by `index` beneath the SDK.
  `openai_compatible_embeddings` always installs it; a caller that builds its own provider wraps
  its transport the same way.

The model source is the caller's, as for the generator:

- `model=`: a fixed pydantic-ai `EmbeddingModel` (tests pass `TestEmbeddingModel`).
- `model_for=`: `() -> async context manager yielding an EmbeddingModel`, called per request, so
  a caller can put the model on its own transport (admission control, leases, a work queue).
- `openai_compatible_embeddings(base_url, model_name, ...)` builds a `model_for` for any
  OpenAI-wire endpoint.

Invariants:
    - A model, wire or gate failure never raises; `classify` names the outcome, with one
      taxonomy for the generator and the embedder.

Rationale: ADR-0001.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
from pydantic_ai.embeddings import EmbeddingModel

from llm_ports.failure import exception_chain, failure_reason
from llm_ports.index_order import IndexOrderedTransport
from llm_ports.ports import Embedded, EmbedInput
from llm_ports.pydantic_ai_generator import classify

EmbedModelFor = Callable[[], AbstractAsyncContextManager[EmbeddingModel]]


def openai_compatible_embeddings(
    base_url: str,
    model_name: str,
    api_key: str | None = None,
    *,
    timeout_s: float = 120.0,
    max_retries: int = 0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> EmbedModelFor:
    """Build a `model_for` over any OpenAI-wire `/embeddings` endpoint.

    Each call opens one httpx client.

    Args:
        base_url: The URL up to where `/embeddings` starts, e.g. `http://localhost:4000/v1`.
        model_name: The embedding model the endpoint serves.
        api_key: The bearer key; None sends a placeholder, for servers that need no key.
        timeout_s: The httpx timeout per call.
        max_retries: The OpenAI SDK's own retry count. Defaults to 0, so retries are not stacked
            on those of a gateway in front.
        transport: Replaces httpx's own transport (a caller's admitting or counting transport, or
            `httpx.MockTransport` in a test). It always sits under `IndexOrderedTransport`.

    Returns:
        An `EmbedModelFor` that yields an `OpenAIEmbeddingModel`.
    """
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

    @asynccontextmanager
    async def model_for() -> AsyncIterator[EmbeddingModel]:
        ordered = IndexOrderedTransport(transport)
        async with httpx.AsyncClient(timeout=timeout_s, transport=ordered) as client:
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

    @property
    def stamp(self) -> str:
        """The name of the space this embedder's vectors live in."""
        return self._stamp

    def _inputs(self, texts: Sequence[str], input_type: EmbedInput) -> list[str]:
        if input_type == "query" and self._instruct:
            return [f"Instruct: {self._instruct}\nQuery: {t}" for t in texts]
        return list(texts)

    async def _embed_on(
        self, model: EmbeddingModel, inputs: list[str], input_type: EmbedInput
    ) -> tuple[list[tuple[float, ...]], int | None]:
        from pydantic_ai.embeddings import Embedder as _PaiEmbedder

        embedder = _PaiEmbedder(model)
        out: list[tuple[float, ...]] = []
        tokens: int | None = 0
        for i in range(0, len(inputs), self._batch):
            batch = inputs[i : i + self._batch]
            res = await embedder.embed(batch, input_type=input_type)
            if len(res.embeddings) != len(batch):
                raise _Invalid(f"sent {len(batch)} inputs, got {len(res.embeddings)} vectors")
            for vec in res.embeddings:
                if self._dims is not None and len(vec) != self._dims:
                    raise _Invalid(f"vector is {len(vec)}-d, the {self._stamp} space is {self._dims}-d")
                out.append(tuple(float(x) for x in vec))
            used = getattr(res.usage, "input_tokens", None)
            tokens = None if (tokens is None or not used) else tokens + int(used)
        return out, tokens

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
        try:
            if self._model is not None:
                vecs, tokens = await self._embed_on(self._model, inputs, input_type)
            else:
                assert self._model_for is not None
                async with self._model_for() as model:
                    vecs, tokens = await self._embed_on(model, inputs, input_type)
        except _Invalid as exc:
            return Embedded(vectors=(), outcome="invalid", stamp=self._stamp, reason=str(exc))
        except Exception as exc:  # the port never raises for a model, wire or gate failure
            outcome, reason = classify(exc, gated=self._gated)
            if outcome == "invalid" or outcome == "truncated":
                outcome = "unavailable"  # an embeddings reply has no parse step or token ceiling
            root = list(exception_chain(exc))[-1]
            if outcome == "unavailable" and root is not exc:
                # pydantic-ai -> openai -> httpx: the wrappers say "Connection error."; the root
                # of the chain names the cause (refused, timed out, DNS).
                reason = f"{reason} <- {failure_reason(root)}"[:240]
            return Embedded(vectors=(), outcome=outcome, stamp=self._stamp, reason=reason)
        return Embedded(vectors=tuple(vecs), outcome="ok", stamp=self._stamp, input_tokens=tokens)


class _Invalid(Exception):
    """A reply that broke the count or dimension check."""


__all__ = ["EmbedModelFor", "IndexOrderedTransport", "PydanticAIEmbedder", "openai_compatible_embeddings"]
