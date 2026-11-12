"""`check_chat` and `check_token_counter`: the `Chat` and `TokenCounter` ports' contracts.

`check_chat` takes a factory from a SCRIPT (`testing.Turn`s, one per model request) to a fresh
`Chat`: the stub takes the script as is; a real adapter's test turns it into its wire (a
pydantic-ai `FunctionModel` answering each request from the next turn). The tools are the check's
own; the chat must really run them.

`check_chat` cases (each is the invariant name a `ContractViolation` carries):
    - `is_chat`: the object satisfies the runtime `Chat` check.
    - `finished_once_last`: `stream` yields exactly one `Finished`, as its last event.
    - `ok_reply`: a reply is `ok`, its value the text, `steps == 1`, its last message the reply.
    - `deltas_are_the_text`: the `TextDelta`s concatenate to the value.
    - `chat_is_drained_stream`: `chat` returns what a drained `stream` finished with.
    - `tool_runs`: a tool call runs the handler with the model's arguments; `ToolCallStarted`
      and `ToolCallFinished` carry it; the messages hold the call and the handler's raw output.
    - `tool_error_is_result`: a raising handler is an error result the run continues past.
    - `tool_budget_cut`: an output over `Tool.max_output_chars` is cut and marked truncated.
    - `max_steps_truncated`: a run past `max_steps` is `truncated` / `max_steps`, keeping the
      tool results it made.
    - `failure_never_raises`: a failed request returns its `Failure`; it never raises.
    - `close_stops`: closing the stream after its first event raises nothing.

`check_token_counter` cases:
    - `is_counter`, `count_never_raises`, `count_shape` (an int with no failure and a measured
      method, or None with `unsupported` or a failure), `count_grows` (more history never counts
      fewer tokens), and with `uncountable=` a model id, `uncountable_is_unsupported`.

Rationale: ADR-0016, ADR-0023, ADR-0024.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, get_args

from llm_ports.ports import (
    OUTCOME_FOR,
    Chat,
    ChatRequest,
    Counted,
    CountMethod,
    FailureKind,
    Finished,
    Generated,
    Message,
    Part,
    StreamEvent,
    TextDelta,
    TokenCounter,
    Tool,
    ToolCallFinished,
    ToolCallStarted,
    ToolOutput,
)
from llm_ports.testing._chat import Turn
from llm_ports.testing.conformance._core import call, expect, is_instance, present
from llm_ports.testing.conformance._generator import CHECK_MODEL

_KIND: FailureKind = "bad_request"
_OBJ = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}


async def _echo(args: Mapping[str, Any]) -> ToolOutput:
    return ToolOutput(f"echo:{args.get('x')}")


async def _boom(_args: Mapping[str, Any]) -> ToolOutput:
    raise RuntimeError("tool bug")


async def _long(_args: Mapping[str, Any]) -> ToolOutput:
    return ToolOutput("x" * 400)


#: The check's tools: one that answers, one that raises, one whose output is over its budget.
TOOLS = (
    Tool("echo", "Echo x.", _OBJ, _echo),
    Tool("boom", "Always fails.", _OBJ, _boom),
    Tool("long", "Returns 400 characters.", _OBJ, _long, max_output_chars=100),
)


def _req(max_steps: int = 4) -> ChatRequest:
    user = Message("user", (Part("text", text="conformance"),))
    return ChatRequest(model=CHECK_MODEL, messages=(user,), tools=TOOLS, max_steps=max_steps)


async def _drain(chat: Chat, req: ChatRequest, case: str) -> list[StreamEvent]:
    async def run() -> list[StreamEvent]:
        return [ev async for ev in chat.stream(req, str)]

    events = await call(case, "stream", run())
    n = sum(isinstance(e, Finished) for e in events)
    expect("finished_once_last", n == 1 and isinstance(events[-1], Finished), f"{case}: {n} Finished events")
    return events


async def _run(
    make: Callable[[Sequence[Turn]], Chat], script: Sequence[Turn], case: str, steps: int = 4
) -> tuple[Generated[Any], list[StreamEvent]]:
    chat = make(script)
    is_instance("is_chat", chat, Chat)
    events = await _drain(chat, _req(steps), case)
    last = events[-1]
    assert isinstance(last, Finished)
    return last.result, events


def _tool_results(res: Generated[Any]) -> list[Part]:
    return [p for m in res.messages if m.role == "tool" for p in m.parts if p.kind == "tool_result"]


async def check_chat(make: Callable[[Sequence[Turn]], Chat]) -> tuple[str, ...]:
    """Check a `Chat` against the port's contract; the module docstring lists the cases.

    Args:
        make: Builds a fresh chat that answers its model requests from the script, in order.

    Returns:
        The invariants checked, in order.

    Raises:
        ContractViolation: The first invariant the chat breaks.
    """
    res, events = await _run(make, [Turn.reply("alpha")], "ok_reply")
    expect("ok_reply", res.outcome == "ok" and res.value == "alpha", f"outcome={res.outcome} value={res.value!r}")
    expect("ok_reply", res.steps == 1, f"one model request, steps={res.steps}")
    expect("ok_reply", bool(res.messages) and res.messages[-1].text == "alpha", f"messages={res.messages}")
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    expect("deltas_are_the_text", text == "alpha", f"TextDeltas joined to {text!r}")

    drained = await call("chat_is_drained_stream", "chat", make([Turn.reply("alpha")]).chat(_req(), str))
    expect(
        "chat_is_drained_stream",
        (drained.outcome, drained.value, drained.messages) == (res.outcome, res.value, res.messages),
        f"chat gave {drained.outcome}/{drained.value!r}, stream gave {res.outcome}/{res.value!r}",
    )

    res, events = await _run(make, [Turn.tool("echo", {"x": "a"}), Turn.reply("done")], "tool_runs")
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    expect("tool_runs", res.outcome == "ok" and res.value == "done", f"outcome={res.outcome} value={res.value!r}")
    expect(
        "tool_runs", [e.name for e in started] == ["echo"] and dict(started[0].args) == {"x": "a"}, f"started={started}"
    )
    expect("tool_runs", len(finished) == 1 and finished[0].output.content == "echo:a", f"finished={finished}")
    results = _tool_results(res)
    expect("tool_runs", [p.text for p in results] == ["echo:a"], f"tool results in messages: {results}")
    calls = [p for m in res.messages if m.role == "assistant" for p in m.parts if p.kind == "tool_call"]
    expect("tool_runs", [p.name for p in calls] == ["echo"], f"tool calls in messages: {calls}")

    res, events = await _run(make, [Turn.tool("boom", {"x": "a"}), Turn.reply("recovered")], "tool_error_is_result")
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    expect("tool_error_is_result", len(finished) == 1 and finished[0].output.is_error, f"finished={finished}")
    expect("tool_error_is_result", res.outcome == "ok" and res.value == "recovered", f"outcome={res.outcome}")

    res, events = await _run(make, [Turn.tool("long", {"x": "a"}), Turn.reply("ok")], "tool_budget_cut")
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    expect("tool_budget_cut", len(finished) == 1 and finished[0].truncated, f"finished={finished}")
    expect("tool_budget_cut", len(finished[0].output.content) < 400, "the output was not cut")

    res, _ = await _run(make, [Turn.tool("echo", {"x": "a"}), Turn.reply("late")], "max_steps_truncated", steps=1)
    got = present("max_steps_truncated", res.failure, f"max_steps=1 gave outcome={res.outcome} and no failure")
    expect(
        "max_steps_truncated",
        got.kind == "truncated" and got.provider_code == "max_steps",
        f"kind={got.kind} provider_code={got.provider_code}",
    )
    expect("max_steps_truncated", [p.text for p in _tool_results(res)] == ["echo:a"], f"messages={res.messages}")

    res, _ = await _run(make, [Turn.failure(_KIND)], "failure_never_raises")
    got = present("failure_never_raises", res.failure, f"a failed request gave outcome={res.outcome} and no failure")
    expect("failure_never_raises", got.kind == _KIND and res.outcome == OUTCOME_FOR[_KIND], f"kind={got.kind}")

    async def close_early() -> None:
        events = make([Turn.tool("echo", {"x": "a"}), Turn.reply("never")]).stream(_req(), str)
        async for _ in events:
            break
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()

    await call("close_stops", "closing the stream", close_early())
    return (
        "is_chat",
        "finished_once_last",
        "ok_reply",
        "deltas_are_the_text",
        "chat_is_drained_stream",
        "tool_runs",
        "tool_error_is_result",
        "tool_budget_cut",
        "max_steps_truncated",
        "failure_never_raises",
        "close_stops",
    )


def _shape(got: Any, what: str) -> Counted:
    expect("count_shape", isinstance(got, Counted), f"{what} returned {type(got).__name__}, not Counted")
    expect("count_shape", got.method in get_args(CountMethod), f"{what}: method={got.method!r}")
    if got.tokens is not None:
        ok = isinstance(got.tokens, int) and got.tokens >= 0 and got.failure is None and got.method != "unsupported"
        expect("count_shape", ok, f"{what}: tokens={got.tokens} method={got.method} failure={got.failure}")
    else:
        ok = got.method == "unsupported" or got.failure is not None
        expect("count_shape", ok, f"{what}: no tokens, yet method={got.method} and no failure")
    return got  # type: ignore[no-any-return]


async def check_token_counter(counter: TokenCounter, *, model: str, uncountable: str | None = None) -> tuple[str, ...]:
    """Check a `TokenCounter` against the port's contract; the module docstring lists the cases.

    Args:
        counter: The counter.
        model: A model id it counts.
        uncountable: A model id it cannot count, when it has one.

    Returns:
        The invariants checked, in order.

    Raises:
        ContractViolation: The first invariant the counter breaks.
    """
    is_instance("is_counter", counter, TokenCounter)
    one = (Message("user", (Part("text", text="hello"),)),)
    more = (*one, Message("assistant", (Part("text", text="hello there, a longer reply"),)), *one)
    small = _shape(
        await call("count_never_raises", "count", counter.count(model, system="", messages=one)), "count(one)"
    )
    big = _shape(
        await call("count_never_raises", "count", counter.count(model, system="sys", messages=more, tools=TOOLS[:1])),
        "count(more)",
    )
    if small.tokens is not None and big.tokens is not None:
        expect("count_grows", big.tokens >= small.tokens, f"more history counted {big.tokens} < {small.tokens}")
    names = ["is_counter", "count_never_raises", "count_shape", "count_grows"]
    if uncountable is not None:
        got = _shape(
            await call("count_never_raises", "count", counter.count(uncountable, system="", messages=one)),
            "count(uncountable)",
        )
        expect("uncountable_is_unsupported", got.tokens is None and got.method == "unsupported", f"got {got}")
        names.append("uncountable_is_unsupported")
    return tuple(names)


__all__ = ["TOOLS", "check_chat", "check_token_counter"]
