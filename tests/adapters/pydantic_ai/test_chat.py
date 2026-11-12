"""`PydanticAIChat`: the adapter's own paths beyond `check_chat` (chat spec 1-11)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, DeltaThinkingPart, DeltaToolCall, FunctionModel

from llm_ports import ports
from llm_ports.adapters.pydantic_ai import PydanticAIChat
from llm_ports.adapters.pydantic_ai._chat_messages import from_model_messages, to_model_messages, wrap_untrusted
from llm_ports.call.resilience import LocalLimiter
from llm_ports.ports import (
    ChatRequest,
    Finished,
    Generated,
    Message,
    Part,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCallFinished,
    ToolOutput,
)

NONE = ports.WirePolicy.none()
OBJ = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
USER = (Message("user", (Part("text", text="q"),)),)


def _req(**kw: Any) -> ChatRequest:
    return ChatRequest(**{"model": "m", "messages": USER, **kw})


def _turns(*turns: Any, seen: list[list[ModelMessage]] | None = None) -> FunctionModel:
    """A streaming FunctionModel: each turn is a str, a list of stream chunks, an exception, or a callable."""
    left = list(turns)

    async def stream(messages: list[ModelMessage], _i: AgentInfo) -> AsyncIterator[Any]:
        if seen is not None:
            seen.append(list(messages))
        t = left.pop(0)
        for chunk in t if isinstance(t, list) else [t]:
            if isinstance(chunk, BaseException):
                raise chunk
            if callable(chunk):
                await chunk()
                continue
            yield chunk

    return FunctionModel(stream_function=stream)


def _call(name: str, args: Mapping[str, Any], cid: str = "c1") -> dict[int, DeltaToolCall]:
    return {0: DeltaToolCall(name=name, json_args=json.dumps(dict(args)), tool_call_id=cid)}


async def _events(chat: PydanticAIChat, req: ChatRequest) -> tuple[list[StreamEvent], Generated[Any]]:
    evs = [e async for e in chat.stream(req)]
    last = evs[-1]
    assert isinstance(last, Finished)
    return evs, last.result


async def _echo(args: Mapping[str, Any]) -> ToolOutput:
    return ToolOutput(f"echo:{args['x']}")


ECHO = Tool("echo", "e", OBJ, _echo)


async def test_reasoning_streams_as_reasoning_deltas() -> None:
    model = _turns([{0: DeltaThinkingPart(content="hmm")}, "an", "swer"])
    evs, res = await _events(PydanticAIChat(model, policy=NONE), _req())
    assert [e.text for e in evs if isinstance(e, ReasoningDelta)] == ["hmm"]
    assert res.value == "answer" and [p.kind for p in res.messages[-1].parts] == ["reasoning", "text"]


async def test_an_unstreamed_request_still_yields_its_text() -> None:
    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("whole")])

    evs, res = await _events(PydanticAIChat(FunctionModel(fn), policy=NONE), _req(streamed=False))
    assert [e.text for e in evs if isinstance(e, TextDelta)] == ["whole"] and res.value == "whole"


async def test_a_slow_a_mistyped_and_an_unknown_tool_are_error_results() -> None:
    async def slow(_a: Mapping[str, Any]) -> ToolOutput:
        await asyncio.sleep(5)
        return ToolOutput("late")  # pragma: no cover

    async def wrong(_a: Mapping[str, Any]) -> Any:
        return "a bare string"

    tools = (Tool("slow", "s", OBJ, slow, timeout_s=0.01), Tool("wrong", "w", OBJ, wrong))
    model = _turns(_call("slow", {"x": "1"}), _call("wrong", {"x": "1"}, "c2"), _call("ghost", {"x": "1"}, "c3"), "end")
    evs, res = await _events(PydanticAIChat(model, policy=NONE), _req(tools=tools, max_steps=6))
    done = [e for e in evs if isinstance(e, ToolCallFinished)]
    assert [(d.name, d.output.is_error) for d in done] == [("slow", True), ("wrong", True), ("ghost", True)]
    assert "TimeoutError" in done[0].output.content and "not ToolOutput" in done[1].output.content
    assert res.outcome == "ok" and res.value == "end" and (res.retries or 0) >= 1


async def test_a_missing_required_argument_is_re_asked() -> None:
    model = _turns(_call("echo", {}), _call("echo", {"x": "ok"}, "c2"), "fine")
    evs, res = await _events(PydanticAIChat(model, policy=NONE), _req(tools=(ECHO,)))
    done = [e for e in evs if isinstance(e, ToolCallFinished)]
    assert done[0].output.is_error and "needs the argument" in done[0].output.content
    assert done[-1].output.content == "echo:ok" and res.value == "fine"


async def test_the_deadline_ends_the_call_with_what_it_made() -> None:
    async def hang() -> None:
        await asyncio.sleep(5)

    model = _turns(["partial", hang])
    evs, res = await _events(PydanticAIChat(model, policy=NONE), _req(timeout_s=0.2))
    assert res.failure is not None and res.failure.kind == "timeout"
    assert any(isinstance(e, TextDelta) for e in evs)


async def test_an_error_mid_stream_is_interrupted_and_keeps_the_partial_text() -> None:
    model = _turns(["half an ans", RuntimeError("socket closed")])
    _, res = await _events(PydanticAIChat(model, policy=NONE), _req())
    assert res.failure is not None and res.failure.in_stream
    assert res.finish is not None and res.finish.unified == "interrupted"
    assert res.messages and res.messages[-1].text == "half an ans"


async def test_model_for_wrap_and_a_failing_model_for() -> None:
    wrapped: list[Model] = []

    @asynccontextmanager
    async def model_for(_req: ChatRequest) -> AsyncIterator[Model]:
        yield _turns("via model_for")

    def wrap(m: Model) -> Model:
        wrapped.append(m)
        return m

    _, res = await _events(PydanticAIChat(model_for=model_for, wrap=wrap, policy=NONE), _req())
    assert res.value == "via model_for" and len(wrapped) == 1

    @asynccontextmanager
    async def broken(_req: ChatRequest) -> AsyncIterator[Model]:
        if _req is not None:
            raise ValueError("bad url")
        yield _turns()  # pragma: no cover

    _, res = await _events(PydanticAIChat(model_for=broken, policy=NONE), _req())
    assert res.outcome != "ok" and res.failure is not None
    with pytest.raises(ValueError, match="exactly one"):
        PydanticAIChat()


async def test_tool_choice_and_parallel_settings() -> None:
    infos: list[AgentInfo] = []

    async def stream(_m: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        infos.append(info)
        yield "ok"

    chat = PydanticAIChat(FunctionModel(stream_function=stream), policy=NONE)
    _, res = await _events(chat, _req(tools=(ECHO,), tool_choice="required", parallel_tools=False))
    assert any(w.setting == "tool_choice" for w in res.warnings)
    assert infos[-1].model_settings is not None and infos[-1].model_settings.get("parallel_tool_calls") is False
    _, res = await _events(chat, _req(tools=(ECHO,), tool_choice="none"))
    assert res.outcome == "ok"
    _, res = await _events(chat, _req(tools=(ECHO,), tool_choice="echo"))
    assert res.outcome == "ok" and [t.name for t in infos[-1].function_tools] == ["echo"]


async def test_a_slot_wait_past_its_bound_is_rate_limited() -> None:
    limiter = LocalLimiter({"g": 1}, max_wait_s=0.01)
    hold = asyncio.Event()

    async def wait() -> None:
        await hold.wait()

    chat = PydanticAIChat(_turns([wait, "first"], "second"), policy=NONE, limiter=limiter)
    first = asyncio.ensure_future(chat.chat(_req(concurrency_group="g")))
    await asyncio.sleep(0.05)
    second = await chat.chat(_req(concurrency_group="g"))
    hold.set()
    assert second.failure is not None and second.failure.provider_code == "local_limiter"
    assert (await first).value == "first"


async def test_closing_the_stream_cancels_the_call() -> None:
    ran: list[str] = []

    async def tool(_a: Mapping[str, Any]) -> ToolOutput:
        ran.append("tool")
        return ToolOutput("x")

    model = _turns(["a", _call("t", {"x": "1"})], "never")
    events = PydanticAIChat(model, policy=NONE).stream(_req(tools=(Tool("t", "t", OBJ, tool),)))
    async for _ in events:
        break
    await events.aclose()  # type: ignore[attr-defined]
    await asyncio.sleep(0.05)
    assert ran == []


async def test_the_history_maps_both_ways_and_marks_untrusted_tool_output() -> None:
    history = (
        Message(
            "user", (Part("text", text="look"), Part("image", data=b"\x89PNG", media_type="image/png")), cache=True
        ),
        Message(
            "assistant",
            (Part("reasoning", text="hidden"), Part("tool_call", call_id="c1", name="echo", args={"x": "1"})),
        ),
        Message("tool", (Part("tool_result", text="<b>page</b>", call_id="c1", name="echo", untrusted=True),)),
        Message("user", (Part("text", text="and?"),)),
    )
    seen: list[list[ModelMessage]] = []
    _, res = await _events(PydanticAIChat(_turns("done", seen=seen), policy=NONE), _req(messages=history))
    assert res.value == "done"
    sent = seen[0]
    ret = next(p for m in sent for p in getattr(m, "parts", ()) if isinstance(p, ToolReturnPart))
    assert ret.content == wrap_untrusted("echo", "<b>page</b>") and 'untrusted="true"' in str(ret.content)
    assert all("hidden" not in str(p) for m in sent for p in getattr(m, "parts", ()))
    assert len(to_model_messages(history)) == 4

    with pytest.warns(Warning, match="RetryPromptPart"):
        old = [RetryPromptPart("fix it"), RetryPromptPart("bad args", tool_name="echo", tool_call_id="c9")]
    back = from_model_messages([ModelRequest(parts=old), ModelResponse(parts=[TextPart("")])], {})
    assert [m.role for m in back] == ["tool", "user"]
    import pydantic_ai.messages as pm

    if not hasattr(pm, "RetryFeedbackPart"):  # an older pydantic-ai has only RetryPromptPart
        return
    feedback = pm.RetryFeedbackPart(content="fix it", retry=pm.RetryInfo(cause="validation"))
    back = from_model_messages([ModelRequest(parts=[feedback])], {})
    assert [m.role for m in back] == ["user"] and "fix it" in back[0].text


async def test_an_output_re_ask_is_counted_and_kept_in_the_messages() -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        n: int

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0].name if info.output_tools else None
        args = '{"n": "x"}' if len(messages) == 1 else '{"n": 3}'
        from pydantic_ai.messages import ToolCallPart

        return ModelResponse(parts=[ToolCallPart(tool, args)]) if tool else ModelResponse(parts=[TextPart(args)])

    res = await PydanticAIChat(FunctionModel(fn), policy=NONE).chat(_req(streamed=False, output_retries=1), Out)
    assert res.outcome == "ok" and res.value == Out(n=3) and res.retries == 1
