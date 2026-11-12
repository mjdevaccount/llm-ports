"""Blocking live model requests in tests: `block_live_requests()` and `LiveRequestBlocked`.

Two guards, so a request is stopped whichever way it is made:

- pydantic-ai's own switch, `models.ALLOW_MODEL_REQUESTS = False`, refuses a request from any of
  its provider models (when pydantic-ai is installed);
- a transport guard: a request that reaches a REAL network transport (`httpx2` or `httpx`,
  async or sync) for any host but loopback raises `LiveRequestBlocked` naming the host. A test
  transport (`MockTransport`, a recorded transport, an in-process ASGI app) is not a network
  transport, and a server on loopback (`localhost`, `127.0.0.0/8`, `::1`) is the test's own: neither
  is blocked.

`block_live_requests()` returns the function that undoes both. The pytest plugin applies it to
every test that does not opt in (`@pytest.mark.llm_live` plus `--llm-live`).
"""

from __future__ import annotations

import functools
import importlib
import ipaddress
from collections.abc import Callable
from typing import Any


class LiveRequestBlocked(RuntimeError):
    """A test sent a request to a live host without opting in.

    Attributes:
        host: The host the request was for.
    """

    def __init__(self, host: str) -> None:
        """Name the host the request was for."""
        super().__init__(
            f"live request to {host!r} blocked: mark the test @pytest.mark.llm_live and pass --llm-live to allow it"
        )
        self.host = host


def _host(request: Any) -> str:
    try:
        return str(request.url.host)
    except AttributeError:
        return "<unknown>"


def is_loopback(host: str) -> bool:
    """Whether `host` is this machine: `localhost`, or a loopback address. Pure."""
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _guard_async(original: Any) -> Any:
    async def guarded(self: Any, request: Any) -> Any:
        host = _host(request)
        if not is_loopback(host):
            raise LiveRequestBlocked(host)
        return await original(self, request)

    return guarded


def _guard_sync(original: Any) -> Any:
    def guarded(self: Any, request: Any) -> Any:
        host = _host(request)
        if not is_loopback(host):
            raise LiveRequestBlocked(host)
        return original(self, request)

    return guarded


# (module, class, method, guard): every real network transport the library's adapters can reach.
_TARGETS: tuple[tuple[str, str, str, Callable[..., Any]], ...] = (
    ("httpx2", "AsyncHTTPTransport", "handle_async_request", _guard_async),
    ("httpx2", "HTTPTransport", "handle_request", _guard_sync),
    ("httpx", "AsyncHTTPTransport", "handle_async_request", _guard_async),
    ("httpx", "HTTPTransport", "handle_request", _guard_sync),
)


def block_live_requests() -> Callable[[], None]:
    """Block every live model request until the returned function is called; see the module docstring."""
    undo: list[Callable[[], None]] = []
    for mod_name, cls_name, meth, guard in _TARGETS:
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except (ImportError, AttributeError):  # not installed: nothing of that kind to block
            continue
        original = cls.__dict__[meth]
        setattr(cls, meth, guard(original))
        undo.append(functools.partial(setattr, cls, meth, original))
    try:
        models = importlib.import_module("pydantic_ai.models")
    except ImportError:
        pass
    else:
        was = getattr(models, "ALLOW_MODEL_REQUESTS", True)
        setattr(models, "ALLOW_MODEL_REQUESTS", False)  # noqa: B010  # a module read by name
        undo.append(functools.partial(setattr, models, "ALLOW_MODEL_REQUESTS", was))

    def release() -> None:
        for fn in reversed(undo):
            fn()

    return release


__all__ = ["LiveRequestBlocked", "block_live_requests", "is_loopback"]
