"""One reused `httpx2.AsyncClient` per event loop, under the adapter's OpenAI-compatible clients.

A client per call means no keep-alive and a TLS handshake per request to a hosted API. The client
here is built lazily, once per event loop (an `httpx2.AsyncClient` is bound to the loop that first
used it), with explicit pool limits and a short connect timeout; the per-request timeout is
passed per call. `aclose()` closes every client it built.

Invariants:
    - Two event loops never share a client; one loop never gets two.
    - The capture hook (`record_response`) is on every client, so a response is recorded into the
      capture of the call that sent it (a `ContextVar`), never another's.
    - A client the caller passed in is used as given and never closed here.

Rationale: ADR-0021; the httpx2 client, ADR-0009.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager

from llm_ports._extras import extra
from llm_ports.ports import GenRequest

with extra("pydantic-ai"):
    import httpx
    import httpx2
    from pydantic_ai.models import Model
    from pydantic_ai.profiles import ModelProfileSpec
    from pydantic_ai.settings import ModelSettings

    from llm_ports.adapters.pydantic_ai._capture import record_response
    from llm_ports.adapters.pydantic_ai._wire import as_httpx2

#: The connect bound every built client uses, in seconds; the request bound is per call.
CONNECT_TIMEOUT_S = 5.0
#: The pool every built client uses.
DEFAULT_LIMITS = httpx2.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0)


class SharedClient:
    """Builds and keeps one `httpx2.AsyncClient` per event loop.

    Args:
        transport: Replaces httpx2's own transport; a legacy httpx one is bridged.
        limits: The connection pool's limits.
        client: A caller's own client, used on every loop as given and never closed here.
        wrap: Applied to the transport before the client is built (the embeddings' index re-sort).
    """

    def __init__(
        self,
        *,
        transport: httpx2.AsyncBaseTransport | httpx.AsyncBaseTransport | None = None,
        limits: httpx2.Limits = DEFAULT_LIMITS,
        client: httpx2.AsyncClient | None = None,
        wrap: Callable[[httpx2.AsyncBaseTransport | None], httpx2.AsyncBaseTransport] | None = None,
    ) -> None:
        self._transport = transport
        self._limits = limits
        self._given = client
        self._wrap = wrap
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx2.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        self._built = 0
        self._requests = 0

    def get(self) -> httpx2.AsyncClient:
        """The running loop's client, built on first use."""
        if self._given is not None:
            return self._given
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None or client.is_closed:
            inner = as_httpx2(self._transport)
            transport = self._wrap(inner) if self._wrap is not None else inner
            client = httpx2.AsyncClient(
                timeout=httpx2.Timeout(None, connect=CONNECT_TIMEOUT_S),
                limits=self._limits,
                transport=transport,
                event_hooks={"request": [self._count], "response": [record_response]},
            )
            self._clients[loop] = client
            self._built += 1
        return client

    async def _count(self, _request: httpx2.Request) -> None:
        self._requests += 1

    def stats(self) -> Mapping[str, int]:
        """`clients_built`, `clients_open` and `requests_sent` so far (a caller gauges them)."""
        open_ = sum(1 for c in self._clients.values() if not c.is_closed)
        return {"clients_built": self._built, "clients_open": open_, "requests_sent": self._requests}

    async def aclose(self) -> None:
        """Close every client built here on the running loop; a client of another loop is dropped. Idempotent."""
        loop = asyncio.get_running_loop()
        for owner, client in list(self._clients.items()):
            if owner is loop and not client.is_closed:
                await client.aclose()
        self._clients.clear()


class OpenAICompatible:
    """A `ModelFor` over one OpenAI-wire chat-completions endpoint, on one reused client per loop.

    Built by `openai_compatible`, which documents the arguments. The provider (and the SDK client
    it wraps) is built once per event loop, with the SDK's own retries at `max_retries`; each
    request gets an `OpenAIChatModel` for its model id on it.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        max_retries: int,
        transport: httpx2.AsyncBaseTransport | httpx.AsyncBaseTransport | None,
        profile: ModelProfileSpec | None,
        settings: ModelSettings | None,
        client: httpx2.AsyncClient | None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._max_retries = max_retries
        self._profile = profile
        self._settings = settings
        self._shared = SharedClient(transport=transport, client=client)
        self._providers: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[httpx2.AsyncClient, object]] = (
            weakref.WeakKeyDictionary()
        )

    def _provider(self) -> object:
        from pydantic_ai.providers.openai import OpenAIProvider

        client = self._shared.get()
        loop = asyncio.get_running_loop()
        held = self._providers.get(loop)
        if held is not None and held[0] is client:
            return held[1]
        # A keyless local endpoint still needs a value for the SDK; it is sent as a bearer nobody reads.
        built = OpenAIProvider(base_url=self._base_url, api_key=self._api_key or "no-key", http_client=client)
        provider = OpenAIProvider(openai_client=built.client.with_options(max_retries=self._max_retries))
        self._providers[loop] = (client, provider)
        return provider

    @asynccontextmanager
    async def __call__(self, req: GenRequest) -> AsyncIterator[Model]:
        """Yield an `OpenAIChatModel` for `req.model` on the running loop's provider."""
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = self._provider()
        assert isinstance(provider, OpenAIProvider)
        yield OpenAIChatModel(req.model, provider=provider, profile=self._profile, settings=self._settings)

    def stats(self) -> Mapping[str, int]:
        """The client's `clients_built`, `clients_open` and `requests_sent`."""
        return self._shared.stats()

    async def aclose(self) -> None:
        """Close the clients built for this endpoint. Idempotent."""
        self._providers.clear()
        await self._shared.aclose()


__all__ = ["OpenAICompatible"]
