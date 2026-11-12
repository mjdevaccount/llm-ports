"""The `Chat` port's value types refuse what could never run, and the call path's edge exits."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from llm_ports import call
from llm_ports.adapters.http import HttpTokenCounter
from llm_ports.adapters.pydantic_ai._core import call_settings
from llm_ports.ports import ChatRequest, Counted, Deadline, Message, Part, Tool, ToolOutput
from llm_ports.testing import StubChat, Turn
from tests.call.test_chat import R, _user

OBJ = {"type": "object", "properties": {}}


async def _ok(_a: Mapping[str, Any]) -> ToolOutput:
    return ToolOutput("ok")


def _sync(_a: Mapping[str, Any]) -> ToolOutput:
    return ToolOutput("ok")


class _Callable:
    async def __call__(self, _a: Mapping[str, Any]) -> ToolOutput:
        return ToolOutput("ok")


def test_a_tool_and_a_request_refuse_what_could_never_run() -> None:
    with pytest.raises(TypeError, match="coroutine function"):
        Tool("t", "d", OBJ, _sync)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="type 'object'"):
        Tool("t", "d", {"type": "string"}, _ok)
    with pytest.raises(ValueError, match="timeout_s"):
        Tool("t", "d", OBJ, _ok, timeout_s=0)
    assert Tool("t", "d", OBJ, _Callable()).name == "t", "an async __call__ is a coroutine handler"
    with pytest.raises(ValueError, match="max_steps"):
        ChatRequest(model="m", messages=(), max_steps=0)
    assert ChatRequest(model="m", messages=(), timeout_s=7).clocks.total_s == 7
    msg = Message("user", (Part("text", text="a"), Part("image", data=b"x"), Part("text", text="b")))
    assert msg.text == "ab"


def test_call_settings_names_each_clock_and_the_cache_prefix() -> None:
    clocks = Deadline(total_s=10, attempt_s=4, first_event_s=2)
    out = call_settings({}, clocks, streamed=True, cache=True)
    assert out == {"timeout": 4, "first_event_timeout": 2, "cache": {"messages": False}}
    assert "first_event_timeout" not in call_settings({}, clocks, streamed=False, cache=False)


async def test_the_preflight_count_is_bounded_by_the_deadline() -> None:
    class Slow:
        async def count(self, *_a: Any, **_k: Any) -> Counted:
            await asyncio.sleep(5)
            return Counted(1, "provider")  # pragma: no cover

    role = R("c.slow", tier="chat", context="check", timeout_s=0.05)
    stub = StubChat([])
    res = await call.chat(role, messages=[_user("q")], chat=stub, counter=Slow())
    assert (
        res.failure is not None
        and (res.failure.kind, res.failure.phase) == ("timeout", "total")
        and stub.requests == []
    )


async def test_a_chat_whose_stream_never_finishes_is_internal() -> None:
    class Silent(StubChat):
        async def stream(self, req: Any, output: Any = str, **kw: Any) -> AsyncIterator[Any]:  # type: ignore[override]
            if req is None:
                yield None  # pragma: no cover

    res = await call.chat(R("c.silent", tier="chat"), messages=[_user("q")], chat=Silent([]))
    assert res.failure is not None and res.failure.kind == "internal"


async def test_a_trace_hook_that_swallows_the_call_is_internal() -> None:
    @asynccontextmanager
    async def swallow(*_a: Any) -> AsyncIterator[Any]:
        try:
            yield _Obs()
        except Exception:
            return

    class Raises(StubChat):
        async def stream(self, req: Any, output: Any = str, **kw: Any) -> AsyncIterator[Any]:  # type: ignore[override]
            if req is not None:
                raise RuntimeError("boom")
            yield None  # pragma: no cover

    hooks = call.Hooks(trace=swallow)
    res = await call.chat(R("c.swallow", tier="chat"), messages=[_user("q")], chat=Raises([]), hooks=hooks)
    assert res.failure is not None and res.failure.kind == "internal"


class _Obs:
    def update(self, **_k: Any) -> None:
        return None

    def score(self, **_k: Any) -> None:
        return None


async def test_a_count_reply_that_is_not_json_is_a_failed_count() -> None:
    c = HttpTokenCounter(
        "http://gw/c", transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"<html>"))
    )
    got = await c.count("m", system="", messages=())
    assert got.tokens is None and got.failure is not None and got.failure.kind == "invalid_output"


async def test_the_stub_runs_tool_budgets_and_validators() -> None:
    from llm_ports.testing import run_tool

    long = Tool("l", "d", OBJ, _ok)
    out, cut, _ = await run_tool(long, {}, 1)
    assert cut and out.content != "ok"
    req = ChatRequest(model="m", messages=(Message("user", (Part("text", text="q"),)),))

    def bad(_v: str) -> list[str]:
        raise RuntimeError("validator bug")

    res = await StubChat([Turn.reply("x", reasoning="think")]).chat(req, validate=bad)
    assert res.failure is not None and res.failure.kind == "internal"
    res = await StubChat([Turn.reply("x")]).chat(req, validate=lambda v: ["no"])
    assert res.outcome == "ok" and res.problems == ("no",)
