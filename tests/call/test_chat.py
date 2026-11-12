"""`call.chat` / `call.stream` and the pre-flight both call paths run (chat spec, context spec)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from llm_ports import call, ports
from llm_ports.call import roles
from llm_ports.call.context import TrimPolicy
from llm_ports.ports import Counted, Finished, Message, Part, TextDelta, Tool, ToolOutput
from llm_ports.testing import RecordingTracer, StubChat, StubGenerator, ToolCall, Turn

RAW: dict[str, Any] = {
    "schema_version": 2,
    "ladder": {"default_s": 30.0},
    "providers": {"local": {"kind": "openai_compatible", "base_url": "local"}},
    "models": {
        "chat": {
            "id": "small-chat",
            "provider": "local",
            "context_window": 120,
            "max_output": 20,
            "supports": ["tools", "stream", "count_tokens", "cache", "json_schema"],
        },
        "plain": {"id": "plain", "provider": "local", "supports": ["json_schema"]},
    },
    "tiers": {"chat": {"model": "chat", "timeout": "default_s"}, "plain": {"model": "plain", "timeout": "default_s"}},
}
TABLE = roles.load_tiers(RAW)


class R(roles.Role):
    TIERS: ClassVar = TABLE.tiers


Events = list[tuple[int, str, dict[str, Any]]]


def _hooks(**kw: Any) -> tuple[call.Hooks, Events]:
    events: Events = []
    return call.Hooks(on_event=lambda lvl, ev, f: events.append((lvl, ev, dict(f))), **kw), events


def _names(events: Events) -> list[str]:
    return [e[1] for e in events]


def _user(text: str) -> Message:
    return Message("user", (Part("text", text=text),))


class CharCounter:
    """Counts one token per character of every message: exact for the test's purposes."""

    def __init__(self) -> None:
        self.calls = 0

    async def count(
        self, model: str, *, system: str, messages: Sequence[Message], tools: Sequence[Tool] = ()
    ) -> Counted:
        self.calls += 1
        return Counted(len(system) + sum(len(m.text) for m in messages), "provider")


async def _echo(args: Mapping[str, Any]) -> ToolOutput:
    return ToolOutput(f"echo:{args['x']}")


async def _boom(_a: Mapping[str, Any]) -> ToolOutput:
    raise RuntimeError("down")


async def _long(_a: Mapping[str, Any]) -> ToolOutput:
    return ToolOutput("y" * 300)


OBJ = {"type": "object", "properties": {"x": {"type": "string"}}}
TOOLS = (
    Tool("echo", "e", OBJ, _echo),
    Tool("boom", "b", OBJ, _boom),
    Tool("long", "l", OBJ, _long, max_output_chars=50),
)


async def test_chat_equals_the_drained_stream() -> None:
    script = [Turn.tool("echo", {"x": "a"}), Turn.reply("done")]
    events = [
        ev
        async for ev in call.stream(R("c.chat", tier="chat"), messages=[_user("q")], chat=StubChat(script), tools=TOOLS)
    ]
    assert isinstance(events[-1], Finished) and sum(isinstance(e, Finished) for e in events) == 1
    streamed = events[-1].result
    res = await call.chat(R("c.chat", tier="chat"), messages=[_user("q")], chat=StubChat(script), tools=TOOLS)
    assert (res.outcome, res.value, res.messages, res.steps) == (streamed.outcome, streamed.value, streamed.messages, 2)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "done"


async def test_preflight_check_refuses_an_overflow_before_the_wire() -> None:
    stub, counter = StubChat([]), CharCounter()
    hooks, events = _hooks()
    role = R("c.check", tier="chat", context="check")
    res = await call.chat(role, messages=[_user("x" * 200)], chat=stub, counter=counter, hooks=hooks)
    assert res.outcome == "request_fault" and res.failure is not None
    assert (res.failure.kind, res.failure.provider_code) == ("context_overflow", "preflight")
    assert stub.requests == [] and res.stamp["count_requests"] == 1
    assert "llm_context_overflow" in _names(events)


async def test_preflight_check_sends_what_fits() -> None:
    stub, counter = StubChat([Turn.reply("ok")]), CharCounter()
    res = await call.chat(
        R("c.check", tier="chat", context="check"), messages=[_user("short")], chat=stub, counter=counter
    )
    assert res.outcome == "ok" and len(stub.requests) == 1 and res.stamp["count_requests"] == 1
    assert "trimmed" not in res.stamp and not res.warnings


def _history(turns: int) -> list[Message]:
    out = [_user("system-ish first")]
    for i in range(turns):
        out += [_user(f"q{i} " + "z" * 20), Message("assistant", (Part("text", text=f"a{i} " + "z" * 20),))]
    return [*out, _user("last")]


async def test_preflight_trim_drops_oldest_turns_until_it_fits() -> None:
    stub, counter = StubChat([Turn.reply("ok")]), CharCounter()
    hooks, events = _hooks()
    role = R("c.trim", tier="chat", context="trim", trim=TrimPolicy(keep_first=1, keep_last=1))
    res = await call.chat(role, messages=_history(4), chat=stub, counter=counter, hooks=hooks)
    assert res.outcome == "ok"
    sent = stub.requests[0].messages
    assert sent[0].text == "system-ish first" and sent[-1].text == "last" and len(sent) < len(_history(4))
    assert res.stamp["trimmed"]["dropped_messages"] > 0 and res.stamp["count_requests"] >= 2
    assert any(w.kind == "clamped" and w.setting == "messages" for w in res.warnings)
    assert "llm_trimmed" in _names(events)


async def test_preflight_trim_refuses_what_still_does_not_fit() -> None:
    stub = StubChat([])
    role = R("c.trim", tier="chat", context="trim", trim=TrimPolicy(keep_first=1, keep_last=1))
    res = await call.chat(role, messages=[_user("w" * 300)], chat=stub, counter=CharCounter())
    assert res.failure is not None and res.failure.kind == "context_overflow" and stub.requests == []


async def test_preflight_unknown_sends_and_warns() -> None:
    stub = StubChat([Turn.reply("ok")])
    hooks, events = _hooks()
    res = await call.chat(R("c.unk", tier="chat", context="check"), messages=[_user("x" * 500)], chat=stub, hooks=hooks)
    assert res.outcome == "ok" and res.stamp["count_requests"] == 0
    assert any(w.setting == "context" for w in res.warnings)

    class Broken:
        async def count(self, *a: Any, **k: Any) -> Counted:
            raise RuntimeError("counter bug")

    res = await call.chat(
        R("c.unk2", tier="chat", context="check"),
        messages=[_user("q")],
        chat=StubChat([Turn.reply("ok")]),
        counter=Broken(),
        hooks=hooks,
    )
    assert res.outcome == "ok" and "llm_count_failed" in _names(events)


async def test_generate_runs_the_preflight_both_ways() -> None:
    counter = CharCounter()
    gen = StubGenerator("fine")
    over = await call.generate(
        R("g.check", tier="chat", context="check"), prompt="x" * 200, generator=gen, counter=counter
    )
    assert over.failure is not None and over.failure.provider_code == "preflight" and gen.requests == []
    ok = await call.generate(R("g.check", tier="chat", context="check"), prompt="fits", generator=gen, counter=counter)
    assert ok.outcome == "ok" and ok.stamp["count_requests"] == 1 and len(gen.requests) == 1
    off = await call.generate(R("g.off", tier="chat"), prompt="x" * 200, generator=gen, counter=counter)
    assert off.outcome == "ok" and "count_requests" not in off.stamp


async def test_generate_reports_an_optimistic_count() -> None:
    hooks, events = _hooks()
    gen = StubGenerator([ports.failure("context_overflow")])
    res = await call.generate(
        R("g.opt", tier="chat", context="check"), prompt="fits", generator=gen, counter=CharCounter(), hooks=hooks
    )
    assert res.failure is not None and res.failure.kind == "context_overflow"
    assert "llm_count_optimistic" in _names(events)


async def test_capability_and_gate_refusals_send_nothing() -> None:
    stub = StubChat([])
    res = await call.chat(R("c.plain", tier="plain"), messages=[_user("q")], chat=stub, tools=TOOLS)
    assert res.outcome == "request_fault" and res.failure is not None and res.failure.kind == "unsupported"
    hooks, _ = _hooks(admit=lambda _r: "switched off")
    res = await call.chat(R("c.gated", tier="chat"), messages=[_user("q")], chat=stub, hooks=hooks)
    assert res.outcome == "gated" and stub.requests == []


async def test_cache_and_stream_follow_the_stated_capabilities() -> None:
    stub = StubChat([Turn.reply("a")])
    await call.chat(R("c.cache", tier="chat", cache_prefix=True), messages=[_user("q")], chat=stub)
    assert stub.requests[0].streamed is True
    plain = StubChat([Turn.reply("a")])
    res = await call.chat(R("c.plain", tier="plain", cache_prefix=True), messages=[_user("q")], chat=plain)
    assert plain.requests[0].streamed is False and plain.requests[0].cache_system is False
    assert {w.setting for w in res.warnings} >= {"cache", "stream"}


async def test_tools_become_child_spans_and_events() -> None:
    tracer = RecordingTracer()
    hooks, events = _hooks(trace=call.tracer_trace(tracer))
    script = [Turn.tools(ToolCall("boom", {"x": "1"}), ToolCall("long", {"x": "2"})), Turn.tool("echo", {"x": "3"})]
    res = await call.chat(
        R("c.tools", tier="chat", max_steps=2), messages=[_user("q")], chat=StubChat(script), tools=TOOLS, hooks=hooks
    )
    assert res.failure is not None and res.failure.provider_code == "max_steps"
    gen = tracer.generations[-1]
    assert [t["name"] for t in gen.tools] == ["boom", "long", "echo"]
    assert [t["is_error"] for t in gen.tools] == [True, False, False]
    names = _names(events)
    assert {"llm_tool_failed", "llm_tool_truncated", "llm_max_steps", "llm_call_failed"} <= set(names)


async def test_closing_the_facade_stream_stops_the_call() -> None:
    stub = StubChat([Turn.tool("echo", {"x": "a"}), Turn.reply("never")])
    events = call.stream(R("c.close", tier="chat"), messages=[_user("q")], chat=stub, tools=TOOLS)
    async for _ in events:
        break
    await events.aclose()
    assert len(stub.unused) == 1


def test_chat_request_for_resolves_the_role() -> None:
    req = call.chat_request_for(
        R("c.req", tier="chat", max_steps=3, tool_output_chars=10), messages=[_user("q")], max_steps=None
    )
    assert (req.model, req.max_steps, req.tool_output_chars, req.label) == ("small-chat", 3, 10, "c.req")
    assert call.chat_request_for(R("c.req", tier="chat"), messages=[], max_steps=5).max_steps == 5
