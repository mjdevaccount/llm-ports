"""The health probe over HTTP, shared by `adapters.http.probe` and the pydantic-ai provider factory.

`GET` one URL (an OpenAI-wire server's `/v1/models`) under a hard deadline and read the model ids
it lists. Both the `http` and the `pydantic-ai` extras install httpx, so each importer imports this
module inside its own `extra(...)` block and a missing httpx names the extra the caller asked for.

Invariants:
    - `probe` never raises; `asyncio.CancelledError` propagates.
    - `down`: the backend could not be reached, timed out, or answered with an error status.
      `unknown`: the probe itself failed, or the backend has no endpoint to judge it by (404, 405).
      A breaker opens on `down` only.

Rationale: ADR-0022.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import httpx

from llm_ports.ports import Health
from llm_ports.ports.outcomes import REASON_MAX, failure_reason

#: The probe's deadline when the caller names none, in seconds.
PROBE_DEADLINE_S = 5.0
#: Statuses that mean the backend has no endpoint the probe can judge it by.
_UNJUDGED = frozenset({404, 405})


def _ids(body: Any) -> tuple[str, ...] | None:
    """The `data[].id` strings of an OpenAI-wire model list, or None when the body is not one."""
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return None
    return tuple(m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str))


class HttpProbe:
    """A `Probe` that `GET`s one URL.

    Args:
        url: The URL to ask, e.g. `http://localhost:4000/v1/models`.
        secret: Sent as a bearer token; None sends no `Authorization` header.
        deadline_s: The whole probe's deadline, connection included.
        transport: Replaces httpx's own transport (a test's `MockTransport`).
        clock: Monotonic seconds; injectable for tests.
    """

    __slots__ = ("_clock", "_deadline_s", "_secret", "_transport", "url")

    def __init__(
        self,
        url: str,
        *,
        secret: str | None = None,
        deadline_s: float = PROBE_DEADLINE_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url: str = url
        self._secret = secret
        self._deadline_s = deadline_s
        self._transport = transport
        self._clock = clock

    def __repr__(self) -> str:
        return f"HttpProbe({self.url!r})"  # never the secret

    async def probe(self) -> Health:
        """Ask the URL once. Never raises.

        Returns:
            `up` with the listed model ids (None when the body lists none); `down` when the backend
            could not be reached, timed out or answered with an error status; `unknown` when the
            probe failed or the backend has no endpoint to judge it by.
        """
        start = self._clock()
        headers = {"Authorization": f"Bearer {self._secret}"} if self._secret else {}
        try:
            try:
                async with (
                    asyncio.timeout(self._deadline_s),
                    httpx.AsyncClient(timeout=self._deadline_s, transport=self._transport) as client,
                ):
                    resp = await client.get(self.url, headers=headers)
            except (httpx.TransportError, TimeoutError) as exc:
                return Health("down", reason=failure_reason(exc)[:REASON_MAX])
            latency = self._clock() - start
            if resp.status_code in _UNJUDGED:
                return Health("unknown", reason=f"HTTP {resp.status_code}: no endpoint to judge by", latency_s=latency)
            if resp.status_code >= 400:
                return Health("down", reason=f"HTTP {resp.status_code}", latency_s=latency)
            try:
                models = _ids(resp.json())
            except ValueError:
                models = None
            return Health("up", models=models, latency_s=latency)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the probe broke: say so, never claim the backend is down
            return Health("unknown", reason=f"probe failed: {failure_reason(exc)}"[:REASON_MAX])


__all__ = ["PROBE_DEADLINE_S", "HttpProbe"]
