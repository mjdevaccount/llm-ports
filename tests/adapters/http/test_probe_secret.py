"""A secret never reaches `Health` or a log: a header httpx refuses is quoted, secret and all.

The live socket case is the real failure: h11 refuses a header holding a newline and names it in
`LocalProtocolError`, so a secret read from a file with its trailing newline reached
`Health.reason`.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Iterator

import httpx
import pytest

from llm_ports.adapters.http import HttpProbe
from llm_ports.ports import Health

SECRET = "sk-SECRETVALUE123"
URL = "http://box.test/v1/models"


@pytest.fixture
def server() -> Iterator[str]:
    """A loopback HTTP server that answers every request with a one-model list."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    port = sock.getsockname()[1]

    def serve() -> None:
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:
                return
            with conn:
                conn.recv(65536)
                body = b'{"data":[{"id":"m"}]}'
                head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                conn.sendall(head % len(body) + body)

    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1/models"
    sock.close()


@pytest.mark.parametrize("tail", ["\n", "\r\n", " ", "\t", "\x00", "é"])
async def test_a_secret_no_header_can_carry_never_reaches_health_or_a_log(
    server: str, tail: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        health = await HttpProbe(server, secret=SECRET + tail).probe()
    assert health == Health("unknown", reason="probe failed: the secret is not a valid header value")
    assert SECRET not in repr(health) and SECRET not in caplog.text


async def test_the_clean_secret_is_up_against_the_same_server(server: str) -> None:
    health = await HttpProbe(server, secret=SECRET).probe()
    assert health.state == "up" and health.models == ("m",), "must fire both ways: the server is fine"


@pytest.mark.parametrize(
    ("exc", "health"),
    [
        (httpx.LocalProtocolError, Health("unknown", reason="probe failed: LocalProtocolError")),
        (httpx.UnsupportedProtocol, Health("unknown", reason="probe failed: UnsupportedProtocol")),
        (httpx.ConnectError, Health("down", reason="ConnectError")),
        (httpx.RemoteProtocolError, Health("down", reason="RemoteProtocolError")),
    ],
)
async def test_a_transport_error_is_named_by_its_type_never_its_message(
    exc: type[httpx.TransportError], health: Health
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc(f"Illegal header value b'Bearer {SECRET}'", request=req)

    got = await HttpProbe(URL, secret=SECRET, transport=httpx.MockTransport(handler)).probe()
    assert got == health and SECRET not in repr(got)


async def test_a_probe_that_breaks_is_named_by_type_never_its_message() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"Bearer {SECRET}")

    got = await HttpProbe(URL, secret=SECRET, transport=httpx.MockTransport(handler)).probe()
    assert got == Health("unknown", reason="probe failed: RuntimeError")
