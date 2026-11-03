"""PydanticAIEmbedder -- the `Embedder` port on pydantic-ai's embeddings (OUR seam, THEIR impl).

pydantic-ai's `OpenAIEmbeddingModel` does the wire (`/v1/embeddings` on any OpenAI-compatible
endpoint: a LiteLLM gateway, llama.cpp's server, OpenAI itself). This adapter adds only what a
store needs and the framework does not promise:

  * the STAMP -- the space every vector lands in (model identity + dims), constant per embedder;
  * the LAWS -- one vector per input in input order, every vector `dims` long, or the whole call
    is `invalid` (a 512-d vector in a 1024 column cosines against garbage; a short batch shifts
    every later vector onto the wrong row);
  * batching (`batch_size` inputs per request) and the asymmetric QUERY prefix (`query_instruct`,
    e.g. Qwen3-Embedding's `Instruct: {task}\\nQuery: {text}`; documents embed plain);
  * NEVER RAISES for a model, wire or gate failure: `classify` names the outcome, exactly as the
    Generator does (one taxonomy for both ports);
  * INDEX ORDER (`IndexOrderedTransport`): pydantic-ai reads a reply's `data` as listed and drops
    each item's `index` (embeddings/openai.py, 2.59), so a server that answers out of order lands
    every vector on the wrong row with the count law still satisfied. The transport puts `data`
    back in `index` order under the SDK (`llm_ports.index_order`, shared with the rerank wire);
    `openai_compatible_embeddings` always wraps with it, and a consumer that builds its own
    provider wraps its transport the same way.

WHERE THE MODEL COMES FROM is the consumer's, as for the Generator:
  * `model=`      a fixed pydantic-ai `EmbeddingModel` (tests pass `TestEmbeddingModel`);
  * `model_for=`  `() -> async context manager yielding an EmbeddingModel`, per call -- how a
                  consumer puts the model on its own transport (admission, leases, a work row);
  * `openai_compatible_embeddings(base_url, model_name, ...)` builds a `model_for` for any
                  OpenAI-wire endpoint.
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


def openai_compatible_embeddings(base_url: str, model_name: str, api_key: str | None = None, *,
                                 timeout_s: float = 120.0, max_retries: int = 0,
                                 transport: httpx.AsyncBaseTransport | None = None) -> EmbedModelFor:
    """A `model_for` over any OpenAI-wire `/embeddings` endpoint (`base_url` ends where
    `/embeddings` starts, e.g. `http://litellm:4000/v1`). One httpx client per call. The SDK's
    own retries default OFF: a gateway in front retries, and two stacked policies multiply.
    `transport` replaces httpx's own (a consumer's admitting/counting transport, or
    `httpx.MockTransport` in a test); it always sits under `IndexOrderedTransport`."""
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
    """The `Embedder` port. `stamp` names the space; `dims` (when given) is enforced per vector."""

    def __init__(self, model: EmbeddingModel | None = None, *, stamp: str, dims: int | None = None,
                 model_for: EmbedModelFor | None = None, batch_size: int = 16,
                 query_instruct: str = "", gated: tuple[type[BaseException], ...] = ()) -> None:
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
        return self._stamp

    def _inputs(self, texts: Sequence[str], input_type: EmbedInput) -> list[str]:
        if input_type == "query" and self._instruct:
            return [f"Instruct: {self._instruct}\nQuery: {t}" for t in texts]
        return list(texts)

    async def _embed_on(self, model: EmbeddingModel, inputs: list[str],
                        input_type: EmbedInput) -> tuple[list[tuple[float, ...]], int | None]:
        from pydantic_ai.embeddings import Embedder as _PaiEmbedder

        embedder = _PaiEmbedder(model)
        out: list[tuple[float, ...]] = []
        tokens: int | None = 0
        for i in range(0, len(inputs), self._batch):
            batch = inputs[i:i + self._batch]
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
                # pydantic-ai -> openai -> httpx: the wrappers say "Connection error."; the
                # bottom of the chain says WHICH (refused, timed out, DNS) -- what names the fix.
                reason = f"{reason} <- {failure_reason(root)}"[:240]
            return Embedded(vectors=(), outcome=outcome, stamp=self._stamp, reason=reason)
        return Embedded(vectors=tuple(vecs), outcome="ok", stamp=self._stamp, input_tokens=tokens)


class _Invalid(Exception):
    """A reply that broke the count or dimension law."""


__all__ = ["EmbedModelFor", "IndexOrderedTransport", "PydanticAIEmbedder", "openai_compatible_embeddings"]
