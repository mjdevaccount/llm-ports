"""The httpx2 wire under the adapter's OpenAI-compatible clients.

pydantic-ai and the OpenAI SDK speak httpx2, and a legacy `httpx.AsyncClient` handed to an
OpenAI-compatible provider is deprecated there. `openai_compatible` and
`openai_compatible_embeddings` therefore always build an `httpx2.AsyncClient`, and still accept a
caller's transport from either library: `as_httpx2` passes an httpx2 transport through and carries
a legacy httpx one across `LegacyTransport`.

Invariants:
    - A legacy transport sees every request in full (method, URL, headers, body, extensions) and
      its reply reaches the client with status, headers and the decoded body. The body is read
      whole: the adapter's calls do not stream.
    - A legacy `httpx.TransportError` re-raises as the httpx2 error of the same name
      (`httpx2.TransportError` when httpx2 has none), chained from the original, so the SDK's
      connection and timeout handling sees its own types. Any other exception passes through
      unchanged, so a caller's gate refusal still classifies as `gated`.

Rationale: ADR-0009.
"""

from __future__ import annotations

from llm_ports._extras import extra

with extra("pydantic-ai"):
    import httpx
    import httpx2

    from llm_ports.adapters._index_order import HOP_HEADERS, index_key, reordered


def as_httpx2(
    transport: httpx2.AsyncBaseTransport | httpx.AsyncBaseTransport | None,
) -> httpx2.AsyncBaseTransport | None:
    """Return `transport` as one an `httpx2.AsyncClient` can use; None stays None (httpx2's own)."""
    if transport is None or isinstance(transport, httpx2.AsyncBaseTransport):
        return transport
    return LegacyTransport(transport)


def _as_httpx2_error(exc: httpx.TransportError, request: httpx2.Request) -> httpx2.TransportError:
    cls = getattr(httpx2, type(exc).__name__, None)
    if not (isinstance(cls, type) and issubclass(cls, httpx2.TransportError)):
        cls = httpx2.TransportError
    return cls(str(exc), request=request)


class LegacyTransport(httpx2.AsyncBaseTransport):
    """An httpx2 transport that sends each request through a legacy httpx transport.

    Args:
        inner: The legacy transport that sends the request.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send `request` through the legacy transport and return its reply as an httpx2 response."""
        legacy = httpx.Request(
            request.method,
            str(request.url),
            headers=request.headers.raw,
            content=await request.aread(),
            extensions=request.extensions,
        )
        try:
            resp = await self._inner.handle_async_request(legacy)
            try:
                content = await resp.aread()
            finally:
                await resp.aclose()
        except httpx.TransportError as exc:
            raise _as_httpx2_error(exc, request) from exc
        headers = [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in HOP_HEADERS]
        return httpx2.Response(resp.status_code, headers=headers, content=content, request=request)

    async def aclose(self) -> None:
        """Close the legacy transport."""
        await self._inner.aclose()


class IndexOrdered(httpx2.AsyncBaseTransport):
    """`IndexOrderedTransport` for an httpx2 client: a 200 reply's index-keyed list, re-sorted by `index`.

    Args:
        inner: The transport that sends the request; None uses httpx2's own.

    Invariants:
        - A non-200 reply, a path not in `INDEXED_LISTS`, an unreadable body, or a list already in
          index order is returned unchanged.
    """

    def __init__(self, inner: httpx2.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx2.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send `request` through the inner transport and return the reply with its list in index order."""
        resp = await self._inner.handle_async_request(request)
        key = index_key(request.url.path)
        if resp.status_code != 200 or key is None:
            return resp
        body = reordered(await resp.aread(), key)
        if body is None:
            return resp
        headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in HOP_HEADERS]
        return httpx2.Response(200, headers=headers, json=body, request=request)

    async def aclose(self) -> None:
        """Close the inner transport."""
        await self._inner.aclose()


__all__: list[str] = []
