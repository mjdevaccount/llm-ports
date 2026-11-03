"""Index order restored to an OpenAI-compatible batch reply's index-keyed list.

Two wire shapes answer a batch with one item per input and an `index` on each item:

- `/embeddings` -> `{"data": [{"index": i, "embedding": [...]}, ...]}`
- `/rerank` -> `{"results": [{"index": i, "relevance_score": s}, ...]}` (the Cohere/Jina shape,
  which llama.cpp's server, LiteLLM and vLLM also speak)

Neither promises the list is in index order: a rerank reply is usually sorted by score, and an
embeddings server may answer out of order. A reader that takes the list as listed lands every
value on the wrong input while every count check still holds. `IndexOrderedTransport` sorts the
list by `index` under the client, so whatever reads the body above it (an SDK, or `HttpReranker`)
sees input order.

Invariants:
    - A body the transport cannot read passes through untouched; the reader's own checks judge it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import httpx

#: Maps a reply path suffix to the key of that reply's index-keyed list.
INDEXED_LISTS: Mapping[str, str] = {"/embeddings": "data", "/rerank": "results"}

_HOP_HEADERS = ("content-length", "content-encoding", "transfer-encoding")


def _list_key(path: str) -> str | None:
    return next((key for suffix, key in INDEXED_LISTS.items() if path.endswith(suffix)), None)


class IndexOrderedTransport(httpx.AsyncBaseTransport):
    """An httpx transport that re-sorts a 200 reply's index-keyed list (`INDEXED_LISTS`) by `index`.

    Args:
        inner: The transport that sends the request; None uses httpx's own.

    Invariants:
        - A non-200 reply, a path not in `INDEXED_LISTS`, an unreadable body, or a list already in
          index order is returned unchanged.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send `request` through the inner transport and return the reply with its list in index order."""
        resp = await self._inner.handle_async_request(request)
        key = _list_key(request.url.path)
        if resp.status_code != 200 or key is None:
            return resp
        raw = await resp.aread()
        try:
            body = json.loads(raw)
            items = body[key]
            ordered = sorted(items, key=lambda d: int(d["index"]))
        except (ValueError, KeyError, TypeError):
            return resp
        if ordered == items:
            return resp
        body[key] = ordered
        headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in _HOP_HEADERS]
        return httpx.Response(200, headers=headers, json=body, request=request)

    async def aclose(self) -> None:
        """Close the inner transport."""
        await self._inner.aclose()


__all__ = ["INDEXED_LISTS", "IndexOrderedTransport"]
