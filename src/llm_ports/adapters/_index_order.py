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

Both the `http` and the `pydantic-ai` extras install httpx, so each importer imports this module
inside its own `extra(...)` block and a missing httpx names the extra the caller asked for. The
re-sort itself (`ordered_reply`) reads only what both httpx and httpx2 replies share, so the
pydantic-ai adapter's httpx2 wire shares it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import httpx

#: Maps a reply path suffix to the key of that reply's index-keyed list.
INDEXED_LISTS: Mapping[str, str] = {"/embeddings": "data", "/rerank": "results"}

#: Headers that describe the original body's framing and encoding, dropped from a rebuilt reply.
HOP_HEADERS: tuple[str, ...] = ("content-length", "content-encoding", "transfer-encoding")


def index_key(path: str) -> str | None:
    """Return the key of the index-keyed list a reply on `path` carries, or None for any other path."""
    return next((key for suffix, key in INDEXED_LISTS.items() if path.endswith(suffix)), None)


def reordered(raw: bytes, key: str) -> dict[str, Any] | None:
    """Return the JSON body `raw` with its `key` list sorted by `index`, or None to keep the reply as is.

    Pure; never raises.

    Args:
        raw: The reply body.
        key: The key of the index-keyed list (`index_key`).

    Returns:
        The re-sorted body, or None when the body is unreadable or the list is already in order.
    """
    try:
        body = json.loads(raw)
        items = body[key]
        ordered = sorted(items, key=lambda d: int(d["index"]))
    except (ValueError, KeyError, TypeError):
        return None
    if ordered == items:
        return None
    body[key] = ordered
    return dict(body)


class _Reply(Protocol):
    """What `ordered_reply` reads from a reply; an httpx and an httpx2 `Response` both have it."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    async def aread(self) -> bytes: ...


async def ordered_reply[R: _Reply](
    resp: R, path: str, rebuild: Callable[[list[tuple[str, str]], dict[str, Any]], R]
) -> R:
    """Return `resp` with its index-keyed list in index order, rebuilt by `rebuild(headers, body)`.

    Never raises past `resp.aread()`.

    Args:
        resp: The reply the inner transport returned.
        path: The request's URL path, which names the list (`index_key`).
        rebuild: Builds a 200 reply of the caller's library from the kept headers and the body.

    Returns:
        `resp` itself for a non-200 reply, a path not in `INDEXED_LISTS`, an unreadable body or a
        list already in order; otherwise the rebuilt reply, without the original framing headers.
    """
    key = index_key(path)
    if resp.status_code != 200 or key is None:
        return resp
    body = reordered(await resp.aread(), key)
    if body is None:
        return resp
    return rebuild([(k, v) for k, v in resp.headers.items() if k.lower() not in HOP_HEADERS], body)


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
        return await ordered_reply(
            resp,
            request.url.path,
            lambda headers, body: httpx.Response(200, headers=headers, json=body, request=request),
        )

    async def aclose(self) -> None:
        """Close the inner transport."""
        await self._inner.aclose()


__all__ = ["INDEXED_LISTS", "IndexOrderedTransport"]
