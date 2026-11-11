"""`adapters.http.probe.HttpProbe`: up, down or unknown, never raising (spec routing test 9)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from llm_ports.adapters.http import PROBE_DEADLINE_S, HttpProbe
from llm_ports.ports import Health, Probe

URL = "http://box.test/v1/models"


def _probe(handler: Callable[[httpx.Request], httpx.Response], **kw: object) -> HttpProbe:
    ticks = iter([10.0, 10.25])
    return HttpProbe(URL, transport=httpx.MockTransport(handler), clock=lambda: next(ticks), **kw)  # type: ignore[arg-type]


async def test_9_a_200_is_up_with_the_listed_model_ids() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"data": [{"id": "small-chat"}, {"id": "small-web"}, {"x": 1}]})

    health = await _probe(handler, secret="sk-test").probe()
    assert health == Health("up", models=("small-chat", "small-web"), latency_s=0.25)
    assert seen[0].headers["Authorization"] == "Bearer sk-test" and str(seen[0].url) == URL


async def test_9_connection_refused_is_down() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    health = await _probe(handler).probe()
    assert health == Health("down", reason="ConnectError"), "the type only, never the message"


async def test_9_a_probe_that_raises_inside_is_unknown_never_down() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise RuntimeError("the probe broke")

    health = await _probe(handler).probe()
    assert health == Health("unknown", reason="probe failed: RuntimeError")


@pytest.mark.parametrize(
    ("status", "state"),
    [(404, "unknown"), (405, "unknown"), (401, "unknown"), (403, "unknown"), (500, "down"), (503, "down")],
)
async def test_an_error_status_is_down_and_a_missing_endpoint_or_refused_credential_is_unknown(
    status: int, state: str
) -> None:
    health = await _probe(lambda _r: httpx.Response(status)).probe()
    assert health.state == state and health.reason is not None and str(status) in health.reason
    assert health.latency_s == 0.25


@pytest.mark.parametrize("status", [401, 403])
async def test_a_refused_credential_is_auth_not_liveness(status: int) -> None:
    health = await _probe(lambda _r: httpx.Response(status)).probe()
    assert health == Health("unknown", reason=f"auth: HTTP {status}", latency_s=0.25)


@pytest.mark.parametrize("body", [b"not json", b'{"data": "x"}', b"[1, 2]", b"{}"])
async def test_a_200_whose_body_is_not_a_model_list_is_unknown(body: bytes) -> None:
    health = await _probe(lambda _r: httpx.Response(200, content=body)).probe()
    assert health == Health("unknown", reason="HTTP 200: body is not a model list", latency_s=0.25)


async def test_an_empty_model_list_is_up_with_no_models() -> None:
    health = await _probe(lambda _r: httpx.Response(200, json={"data": []})).probe()
    assert health == Health("up", models=(), latency_s=0.25)


async def test_the_deadline_bounds_the_whole_probe() -> None:
    async def slow(_req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    probe = HttpProbe(URL, transport=httpx.MockTransport(slow), deadline_s=0.01)
    health = await probe.probe()
    assert health.state == "down"


async def test_cancellation_propagates() -> None:
    started = asyncio.Event()

    async def hang(_req: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(60)
        return httpx.Response(200)

    task = asyncio.create_task(HttpProbe(URL, transport=httpx.MockTransport(hang)).probe())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_it_is_a_probe_and_its_repr_never_shows_the_secret() -> None:
    probe = HttpProbe(URL, secret="sk-test-hidden")
    assert isinstance(probe, Probe)
    assert "sk-test-hidden" not in repr(probe) and URL in repr(probe)
    assert PROBE_DEADLINE_S == 5.0
