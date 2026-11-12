"""`StubChat`, `check_chat` and `check_token_counter`: the stub and the reference adapter both pass,
and every case fires on a chat (or counter) broken in exactly that one way."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from typing import Any

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from llm_ports import ports
from llm_ports.adapters.http import HttpTokenCounter
from llm_ports.adapters.pydantic_ai import PydanticAIChat
from llm_ports.ports import Counted, Finished, Generated, Message, Part, StreamEvent, TextDelta, ToolCallFinished
from llm_ports.testing import ContractViolation, StubChat, ToolCall, Turn
from llm_ports.testing import conformance as cf


def _pydantic_ai_from(script: Sequence[Turn]) -> PydanticAIChat:
    """The reference adapter over a FunctionModel streaming each request from the next turn."""
    turns = list(script)

    async def stream(_m: list[ModelMessage], _i: AgentInfo) -> AsyncIterator[Any]:
        t = turns.pop(0)
        if t.failed is not None:
            raise ModelHTTPError(status_code=400, model_name="scripted", body={"error": t.failed.reason})
        if t.text:
            yield t.text
        if t.calls:
            yield {
                i: DeltaToolCall(name=c.name, json_args=json.dumps(dict(c.args)), tool_call_id=c.call_id or f"c{i}")
                for i, c in enumerate(t.calls)
            }

    return PydanticAIChat(FunctionModel(stream_function=stream), policy=ports.WirePolicy.none())


async def test_check_chat_passes_for_the_stub() -> None:
    cases = await cf.check_chat(StubChat)
    assert "tool_runs" in cases and "max_steps_truncated" in cases


async def test_check_chat_passes_for_the_pydantic_ai_adapter() -> None:
    assert await cf.check_chat(_pydantic_ai_from) == await cf.check_chat(StubChat)


def _if(cond: Callable[[Generated[Any]], bool], change: Callable[[Generated[Any]], Generated[Any]]) -> Any:
    return lambda r: change(r) if cond(r) else r


def _drop_tool_results(r: Generated[Any]) -> Generated[Any]:
    return replace(r, messages=tuple(m for m in r.messages if m.role != "tool"))


#: One result mutation per case: each breaks exactly the property its case names.
RESULT_MUTANTS: dict[str, Callable[[Generated[Any]], Generated[Any]]] = {
    "ok_reply": _if(lambda r: r.value == "alpha", lambda r: replace(r, steps=2)),
    "tool_runs": _if(lambda r: r.value == "done", _drop_tool_results),
    "tool_error_is_result": _if(lambda r: r.value == "recovered", lambda r: replace(r, outcome="unavailable")),
    "max_steps_truncated": _if(
        lambda r: r.failure is not None and r.failure.kind == "truncated",
        lambda r: replace(r, failure=ports.failure("server")),
    ),
    "failure_never_raises": _if(
        lambda r: r.failure is not None and r.failure.kind == "bad_request",
        lambda r: replace(r, failure=ports.failure("server")),
    ),
}


class _Mutant(StubChat):
    """A stub whose stream is rewritten event by event."""

    def __init__(self, script: Sequence[Turn], edit: Callable[[list[StreamEvent]], list[StreamEvent]]) -> None:
        super().__init__(script)
        self._edit = edit

    async def stream(self, req: Any, output: Any = str, **kw: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        events = [ev async for ev in super().stream(req, output, **kw)]
        for ev in self._edit(events):
            yield ev


def _results(mutate: Callable[[Generated[Any]], Generated[Any]]) -> Callable[[list[StreamEvent]], list[StreamEvent]]:
    return lambda evs: [Finished(mutate(e.result)) if isinstance(e, Finished) else e for e in evs]


#: One stream mutation per stream-level case.
STREAM_MUTANTS: dict[str, Callable[[list[StreamEvent]], list[StreamEvent]]] = {
    "finished_once_last": lambda evs: [*evs, evs[-1]],
    "deltas_are_the_text": lambda evs: [TextDelta("x") if isinstance(e, TextDelta) else e for e in evs],
    "tool_budget_cut": lambda evs: [replace(e, truncated=False) if isinstance(e, ToolCallFinished) else e for e in evs],
    **{case: _results(m) for case, m in RESULT_MUTANTS.items()},
}


@pytest.mark.parametrize("case", sorted(STREAM_MUTANTS))
async def test_check_chat_fires_on_one_mutant_per_property(case: str) -> None:
    with pytest.raises(ContractViolation) as e:
        await cf.check_chat(lambda s: _Mutant(s, STREAM_MUTANTS[case]))
    assert e.value.invariant == case


async def test_check_chat_fires_when_chat_is_not_the_drained_stream() -> None:
    class _Drifts(StubChat):
        async def chat(self, req: Any, output: Any = str, **kw: Any) -> Any:
            return replace(await super().chat(req, output, **kw), value="other")

    with pytest.raises(ContractViolation) as e:
        await cf.check_chat(_Drifts)
    assert e.value.invariant == "chat_is_drained_stream"


async def test_check_chat_fires_on_a_raise_a_bad_close_and_a_non_chat() -> None:
    class _Raises(StubChat):
        async def stream(self, req: Any, output: Any = str, **kw: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
            if req is not None:
                raise RuntimeError("model down")
            yield TextDelta("")  # pragma: no cover

    with pytest.raises(ContractViolation, match="stream raised RuntimeError"):
        await cf.check_chat(_Raises)

    class _BadClose(StubChat):
        async def stream(self, req: Any, output: Any = str, **kw: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
            done = False
            try:
                async for ev in super().stream(req, output, **kw):
                    yield ev
                done = True
            finally:
                if not done:
                    raise RuntimeError("close blew up")

    with pytest.raises(ContractViolation) as e:
        await cf.check_chat(_BadClose)
    assert e.value.invariant == "close_stops"
    with pytest.raises(ContractViolation) as e:
        await cf.check_chat(lambda _s: object())  # type: ignore[arg-type,return-value]
    assert e.value.invariant == "is_chat"


async def test_stub_chat_reports_script_exhaustion_and_leftovers() -> None:
    chat = StubChat([Turn.tool("missing", {})])
    req = ports.ChatRequest(model="m", messages=(Message("user", (Part("text", text="q"),)),))
    with pytest.raises(ContractViolation, match="script_exhausted"):
        await chat.chat(req)
    left = StubChat([Turn.reply("a"), Turn.reply("b")])
    await left.chat(req)
    with pytest.raises(ContractViolation, match="script_unused"):
        left.assert_spent()


async def test_stub_chat_unknown_tool_and_structured_output() -> None:
    req = ports.ChatRequest(model="m", messages=(Message("user", (Part("text", text="q"),)),))
    res = await StubChat([Turn.tools(ToolCall("nope", {}, "c1")), Turn.reply('{"a": 1}')]).chat(req, dict[str, int])
    assert res.value == {"a": 1} and res.messages[1].parts[0].text == "unknown tool 'nope'"
    bad = await StubChat([Turn.reply("not json")]).chat(req, dict[str, int])
    assert bad.failure is not None and bad.failure.kind == "invalid_output"


class _Counter:
    def __init__(self, tokens: int | None, method: Any = "provider", failure: Any = None) -> None:
        self.tokens, self.method, self.failure = tokens, method, failure

    async def count(
        self, model: str, *, system: str, messages: Sequence[Message], tools: Sequence[Any] = ()
    ) -> Counted:
        if model == "uncountable":
            return Counted(None, "unsupported")
        n = self.tokens if self.tokens is None else self.tokens + sum(len(m.text) for m in messages)
        return Counted(n, self.method, self.failure)


async def test_check_token_counter_passes_and_fires() -> None:
    assert "uncountable_is_unsupported" in await cf.check_token_counter(
        _Counter(3), model="m", uncountable="uncountable"
    )
    cases = {
        "count_shape": _Counter(3, method="unsupported"),
        "is_counter": object(),
    }
    for case, counter in cases.items():
        with pytest.raises(ContractViolation) as e:
            await cf.check_token_counter(counter, model="m")  # type: ignore[arg-type]
        assert e.value.invariant == case

    class _Shrinks(_Counter):
        async def count(
            self, model: str, *, system: str, messages: Sequence[Message], tools: Sequence[Any] = ()
        ) -> Counted:
            return Counted(100 - len(messages), "provider")

    with pytest.raises(ContractViolation) as e:
        await cf.check_token_counter(_Shrinks(0), model="m")
    assert e.value.invariant == "count_grows"

    class _Estimates(_Counter):
        async def count(
            self, model: str, *, system: str, messages: Sequence[Message], tools: Sequence[Any] = ()
        ) -> Counted:
            return Counted(9, "provider")

    with pytest.raises(ContractViolation) as e:
        await cf.check_token_counter(_Estimates(0), model="m", uncountable="uncountable")
    assert e.value.invariant == "uncountable_is_unsupported"


def _count_transport(status: int = 200, body: Any = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        if body is not None or status != 200:
            return httpx.Response(status, json=body)
        return httpx.Response(200, json={"total_tokens": len(json.dumps(sent["messages"])) // 4, "tokenizer_type": "x"})

    return httpx.MockTransport(handle)


async def test_http_token_counter_passes_the_check_and_maps_failures() -> None:
    counter = HttpTokenCounter("http://gw/utils/token_counter", models={"m"}, transport=_count_transport())
    assert "count_grows" in await cf.check_token_counter(counter, model="m", uncountable="other")
    one = (Message("user", (Part("text", text="hi"),)),)
    got = await counter.count("m", system="", messages=one)
    assert got.tokens is not None and got.method == "tokenizer"
    for status, body, method, kind in (
        (404, {}, "unsupported", None),
        (500, {"error": "x"}, "provider", "server"),
        (200, {"nope": 1}, "provider", "invalid_output"),
    ):
        c = HttpTokenCounter("http://gw/c", transport=_count_transport(status, body))
        got = await c.count("m", system="s", messages=one)
        assert got.tokens is None and got.method == method
        assert (got.failure.kind if got.failure else None) == kind


async def test_http_token_counter_transport_error_is_a_failed_count() -> None:
    def refuse(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    c = HttpTokenCounter("http://gw/c", transport=httpx.MockTransport(refuse))
    got = await c.count("m", system="", messages=())
    assert got.tokens is None and got.failure is not None and got.failure.kind == "transport"
    with pytest.raises(ValueError, match="url is required"):
        HttpTokenCounter("")


def test_count_payload_carries_tool_calls_results_and_tools() -> None:
    from llm_ports.adapters.http.token_count import count_payload
    from llm_ports.testing.conformance import TOOLS

    msgs = (
        Message("assistant", (Part("tool_call", call_id="c1", name="echo", args={"x": "a"}),)),
        Message("tool", (Part("tool_result", text="echo:a", call_id="c1", name="echo"),)),
    )
    body = count_payload("m", "sys", msgs, TOOLS[:1])
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1]["tool_calls"][0]["function"]["name"] == "echo"
    assert body["messages"][2] == {"role": "tool", "tool_call_id": "c1", "content": "echo:a"}
    assert body["tools"][0]["function"]["name"] == "echo"
