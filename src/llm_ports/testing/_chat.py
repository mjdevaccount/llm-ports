"""`StubChat`: a `Chat` that plays a SCRIPT of model turns and really runs the tools it is offered.

A script is a sequence of turns, one per model request: `Turn.text("...")` ends the run with that
reply, `Turn.tool("name", {...})` (or `Turn.tools(...)` for parallel calls) asks for tool calls,
and `Turn.failed(kind)` fails the request. A tool call runs the `Tool.handler` from the request
under its timeout and output budget, exactly as the adapter does, so a test sees the handler's
real output, a raising handler as an error result, and a budget cut as `truncated`. A script that
runs out raises `ContractViolation("script_exhausted")`; replies left over fail `assert_spent()`.

Invariants:
    - `stream` yields `Finished` exactly once, last, and never raises for a model or tool failure;
      `chat` is `stream` drained.
    - `max_steps` is checked before each model request: the run ends `truncated`
      (`provider_code="max_steps"`) with the messages so far, tool results included.
    - A tool output is cut to `Tool.max_output_chars`, else `ChatRequest.tool_output_chars`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError

from llm_ports.call.context import cut_middle
from llm_ports.ports import (
    Answered,
    ChatRequest,
    Failure,
    FailureKind,
    Finish,
    Finished,
    Generated,
    Message,
    Part,
    ReasoningDelta,
    StepFinished,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCallFinished,
    ToolCallStarted,
    ToolOutput,
    Usage,
    failure,
)
from llm_ports.ports.outcomes import failure_reason
from llm_ports.testing._violation import ContractViolation

#: The model id a `StubChat` reports as having answered unless told otherwise.
STUB_CHAT_MODEL = "stub-chat"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One scripted tool call.

    Attributes:
        name: The tool's name.
        args: The arguments the model sends.
        call_id: The call id; None numbers it.
    """

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    """One scripted model turn: final text, tool calls, or a failure. Build one with the class methods.

    Attributes:
        text: The reply text (final when there are no calls).
        calls: The tool calls asked for.
        reasoning: Reasoning text streamed before the reply.
        usage: The request's usage; None is unmeasured.
        failed: The failure the request ends with.
    """

    text: str = ""
    calls: tuple[ToolCall, ...] = ()
    reasoning: str = ""
    usage: Usage | None = None
    failed: Failure | None = None

    @classmethod
    def reply(cls, text: str, *, reasoning: str = "", usage: Usage | None = None) -> Turn:
        """A final reply. Pure."""
        return cls(text=text, reasoning=reasoning, usage=usage)

    @classmethod
    def tool(cls, name: str, args: Mapping[str, Any] | None = None, *, call_id: str | None = None) -> Turn:
        """One tool call. Pure."""
        return cls(calls=(ToolCall(name, dict(args or {}), call_id),))

    @classmethod
    def tools(cls, *calls: ToolCall, usage: Usage | None = None) -> Turn:
        """Parallel tool calls in one turn. Pure."""
        return cls(calls=tuple(calls), usage=usage)

    @classmethod
    def failure(cls, kind: FailureKind, **detail: Any) -> Turn:
        """A request that fails with `kind`; `detail` is `ports.failure`'s keywords. Pure."""
        return cls(failed=failure(kind, **detail))


def _plus(total: Usage | None, more: Usage | None) -> Usage | None:
    if more is None:
        return total
    return more if total is None else total + more


async def run_tool(tool: Tool, args: Mapping[str, Any], budget: int | None) -> tuple[ToolOutput, bool, float]:
    """Run one handler under its timeout and cut its output to `budget`; never raises but cancellation.

    Returns:
        The output (an error output for a timeout, a raise, or a non-`ToolOutput` return), whether
        it was cut, and the seconds the handler ran.
    """
    started = time.perf_counter()
    try:
        async with asyncio.timeout(tool.timeout_s):
            got: Any = await tool.handler(args)
    except TimeoutError:
        got = ToolOutput(f"tool {tool.name!r} timed out after {tool.timeout_s:g} s", is_error=True)
    except Exception as exc:  # a broken tool is an error result the model sees, never a crash
        got = ToolOutput(f"tool {tool.name!r} failed: {failure_reason(exc)}", is_error=True)
    if not isinstance(got, ToolOutput):
        got = ToolOutput(f"tool {tool.name!r} returned {type(got).__name__}, not ToolOutput", is_error=True)
    duration = time.perf_counter() - started
    if budget is None:
        return got, False, duration
    text, removed = cut_middle(got.content, budget)
    return (
        (ToolOutput(text, is_error=got.is_error, meta=got.meta), True, duration) if removed else (got, False, duration)
    )


class StubChat:
    """A `Chat` with no wire; the module docstring states the contract.

    Args:
        script: The model turns, one per model request.
        model: The model id the stub reports in `answered.model`.

    Attributes:
        requests: Every `ChatRequest` received, in call order.
    """

    def __init__(self, script: Sequence[Turn], *, model: str = STUB_CHAT_MODEL) -> None:
        self._script = list(script)
        self._played = 0
        self._model = model
        self.requests: list[ChatRequest] = []

    @property
    def unused(self) -> tuple[Turn, ...]:
        """The scripted turns no request has played yet."""
        return tuple(self._script[self._played :])

    def assert_spent(self) -> None:
        """Raise `ContractViolation("script_unused")` when scripted turns were never played."""
        if self.unused:
            raise ContractViolation("script_unused", f"{len(self.unused)} of {len(self._script)} turns never played")

    def _next(self) -> Turn:
        if self._played >= len(self._script):
            raise ContractViolation(
                "script_exhausted", f"request {self._played + 1} has no turn; the script holds {len(self._script)}"
            )
        self._played += 1
        return self._script[self._played - 1]

    async def chat[T](
        self,
        req: ChatRequest,
        output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> Generated[T]:
        """Run `stream` to its end and return `Finished.result`."""
        result: Generated[T] | None = None
        async for ev in self.stream(req, output, validate=validate, retry_message=retry_message):
            if isinstance(ev, Finished):
                result = ev.result
        assert result is not None  # stream always yields Finished
        return result

    async def stream[T](
        self,
        req: ChatRequest,
        output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """Play the script for `req`, running its tools; `retry_message` is unused (no re-ask)."""
        del retry_message
        self.requests.append(req)
        tools = {t.name: t for t in req.tools}
        made: list[Message] = []
        total: Usage | None = None
        answered = Answered(requested=req.model, model=self._model)
        step = 0
        while True:
            if step >= req.max_steps:
                f = failure("truncated", provider_code="max_steps", reason=f"max_steps: {req.max_steps} model requests")
                yield Finished(self._ended(req, f, made, total, answered, step))
                return
            turn = self._next()
            step += 1
            total = _plus(total, turn.usage)
            if turn.failed is not None:
                yield Finished(self._ended(req, turn.failed, made, total, answered, step))
                return
            if turn.reasoning:
                yield ReasoningDelta(turn.reasoning)
            if turn.text:
                yield TextDelta(turn.text)
            if not turn.calls:
                made.append(Message("assistant", (Part("text", text=turn.text),)))
                yield StepFinished(step, Finish("stop", "stop"), turn.usage)
                yield Finished(self._final(req, output, validate, turn.text, made, total, answered, step))
                return
            async for ev in self._calls(req, turn, tools, made, step):
                yield ev

    async def _calls(
        self, req: ChatRequest, turn: Turn, tools: Mapping[str, Tool], made: list[Message], step: int
    ) -> AsyncGenerator[StreamEvent]:
        calls = [(c.call_id or f"call_{step}_{i}", c) for i, c in enumerate(turn.calls)]
        parts = [Part("text", text=turn.text)] if turn.text else []
        parts += [Part("tool_call", call_id=cid, name=c.name, args=c.args) for cid, c in calls]
        made.append(Message("assistant", tuple(parts)))
        results: list[Part] = []
        for cid, c in calls:
            yield ToolCallStarted(cid, c.name, c.args)
            tool = tools.get(c.name)
            if tool is None:
                out, cut, took = ToolOutput(f"unknown tool {c.name!r}", is_error=True), False, 0.0
            else:
                out, cut, took = await run_tool(tool, c.args, tool.max_output_chars or req.tool_output_chars)
            yield ToolCallFinished(cid, c.name, out, took, cut)
            untrusted = tool.untrusted_output if tool is not None else False
            results.append(Part("tool_result", text=out.content, call_id=cid, name=c.name, untrusted=untrusted))
        made.append(Message("tool", tuple(results)))
        yield StepFinished(step, Finish("tool_call", "tool_calls"), turn.usage)

    def _ended(
        self, req: ChatRequest, f: Failure, made: list[Message], total: Usage | None, answered: Answered, step: int
    ) -> Generated[Any]:
        return Generated(
            value=None,
            outcome=f.outcome,
            model=req.model,
            reason=f.reason,
            usage_v2=total,
            answered=answered,
            failure=f,
            messages=tuple(made),
            steps=step,
        )

    def _final[T](
        self,
        req: ChatRequest,
        output: type[T],
        validate: Callable[[T], Sequence[Any]] | None,
        text: str,
        made: list[Message],
        total: Usage | None,
        answered: Answered,
        step: int,
    ) -> Generated[T]:
        try:
            value: Any = text if output is str else TypeAdapter(output).validate_json(text)
        except ValidationError as exc:
            f = failure("invalid_output", reason=f"output does not parse: {exc.error_count()} errors")
            return self._ended(req, f, made, total, answered, step)
        try:
            problems = tuple(validate(value)) if validate is not None else ()
        except Exception as exc:  # a bug in the caller's validator is never a model outage
            return self._ended(req, failure("internal", reason=failure_reason(exc)), made, total, answered, step)
        return Generated(
            value=value,
            outcome="ok",
            model=req.model,
            retries=0,
            problems=problems,
            usage_v2=total,
            finish=Finish("stop", "stop"),
            answered=answered,
            messages=tuple(made),
            steps=step,
        )


__all__ = ["STUB_CHAT_MODEL", "StubChat", "ToolCall", "Turn", "run_tool"]
