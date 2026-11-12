"""The `Chat` port implemented on pydantic-ai.

One `Agent` runs per call, with the request's tools as function tools (`Tool.from_schema`), the
system prompt as its instructions, and the history as its message history. The agent is driven
with `agent.iter(...)` for both `chat` and `stream`: each model request is streamed with
`node.stream(...)` and its text and reasoning deltas become `TextDelta` / `ReasoningDelta` as they
arrive; each tool round becomes `ToolCallStarted` / `ToolCallFinished`; each model request ends
with a `StepFinished`; the stream ends with exactly one `Finished`. `chat()` is `stream()` drained
(ADR-0023). The model loop, tool dispatch, the unknown-tool and argument re-ask and the stream
parse are pydantic-ai's; this module owns the bounds, the tool-output boundary and budget, and the
mapping. Client, capture, classification, wire retry, limiter, usage and cost are the generator's,
shared through `_core.AdapterCore`.

The call runs in its own task and hands events over a queue, so the call's deadline and limiter
slot are scoped to the call and not to whoever iterates the stream. Closing the iterator (or
cancelling the consumer) cancels that task: the HTTP stream is closed and the slot released.

Invariants:
    - Bounded three ways: `ChatRequest.max_steps` model requests (reaching it with a tool call
      pending is `truncated` / `provider_code="max_steps"`, with the messages so far), the call's
      `Deadline`, and `Tool.timeout_s` per tool call.
    - A tool that raises or times out does not fail the call: the model is shown
      `ToolOutput(content="<type>: <message>", is_error=True)` and may recover. An unknown tool or
      missing required arguments are returned to the model as an error and count against
      `output_retries`; past them the call is `invalid` / `invalid_output`.
    - A tool output over its budget is cut in the middle with `[... N chars omitted]`,
      `ToolCallFinished.truncated=True` and a `CallWarning`; an untrusted output is wrapped in
      `<tool_output name=... untrusted="true">` and sent in the tool-result channel.
    - `stream` never raises for a model, wire or gate failure. A failure after a step's output
      started is `in_stream` and not retried (the wire policy re-opens a stream only before its
      first event); the partial text is kept in `messages` and `finish.unified` is `interrupted`.
    - Usage and cost are summed over every step; `StepFinished` carries each step's own.
    - `asyncio.CancelledError` from the caller propagates.

Rationale: ADR-0001, ADR-0021, ADR-0023.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, aclosing
from dataclasses import replace
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters._classify import Classifier, WireFacts
from llm_ports.adapters.pydantic_ai._capture import CaptureLike, capture
from llm_ports.adapters.pydantic_ai._chat_messages import (
    ToolRecord,
    from_model_messages,
    from_response,
    is_retry,
    to_model_messages,
    wrap_untrusted,
)
from llm_ports.adapters.pydantic_ai._classify import classify_failure
from llm_ports.adapters.pydantic_ai._core import WIRE, AdapterCore, build_agent, call_settings, settled
from llm_ports.adapters.pydantic_ai._result import finish_of, unsupported, usage_of
from llm_ports.call.context import cut_middle
from llm_ports.observability import EventHook
from llm_ports.ports import (
    Answered,
    CallWarning,
    ChatRequest,
    Failure,
    Finish,
    Finished,
    GatewayReader,
    Generated,
    Limiter,
    Message,
    Part,
    Pricer,
    ReasoningDelta,
    SlotWaitExceeded,
    StepFinished,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCallFinished,
    ToolCallStarted,
    ToolOutput,
    WirePolicy,
    failure,
)

with extra("pydantic-ai"):
    from pydantic_ai import Agent, ModelRetry, RunContext, ToolFailed, ToolOrOutput, UsageLimits, capture_run_messages
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        ModelMessage,
        ModelResponse,
        PartDeltaEvent,
        PartStartEvent,
        RetryPromptPart,
        TextPart,
        TextPartDelta,
        ThinkingPart,
        ThinkingPartDelta,
        ToolReturnPart,
    )
    from pydantic_ai.models import Model
    from pydantic_ai.tools import Tool as AITool

ChatModelFor = Callable[[ChatRequest], AbstractAsyncContextManager[Model]]


class _Run:
    """One call's live state: the events handed out, the step count, the tool records, the partial text."""

    def __init__(self, put: Callable[[StreamEvent], None], loop: asyncio.AbstractEventLoop) -> None:
        self._put = put
        self._loop = loop
        self.started = loop.time()
        self.first_event_s: float | None = None
        self.steps = 0
        self.step_output = False
        self.step_done = False
        self.step_text = ""
        self.tools: dict[str, ToolRecord] = {}
        self.warnings: list[CallWarning] = []
        self.history_len = 0
        self.messages: list[ModelMessage] = []
        self.pending: list[ModelMessage] = []

    def emit(self, ev: StreamEvent) -> None:
        if self.first_event_s is None:
            self.first_event_s = self._loop.time() - self.started
        self._put(ev)

    def begin_step(self) -> None:
        self.steps += 1
        self.step_output = False
        self.step_done = False
        self.step_text = ""

    def model_event(self, ev: Any) -> None:
        """A text or reasoning piece of the reply being streamed."""
        if isinstance(ev, PartStartEvent):
            if isinstance(ev.part, TextPart) and ev.part.content:
                self._text(ev.part.content)
            elif isinstance(ev.part, ThinkingPart) and ev.part.content:
                self._reasoning(ev.part.content)
        elif isinstance(ev, PartDeltaEvent):
            if isinstance(ev.delta, TextPartDelta) and ev.delta.content_delta:
                self._text(ev.delta.content_delta)
            elif isinstance(ev.delta, ThinkingPartDelta) and ev.delta.content_delta:
                self._reasoning(ev.delta.content_delta)

    def _text(self, text: str) -> None:
        self.step_output = True
        self.step_text += text
        self.emit(TextDelta(text))

    def _reasoning(self, text: str) -> None:
        self.step_output = True
        self.emit(ReasoningDelta(text))

    def end_step(self, response: ModelResponse, *, streamed: bool) -> None:
        """The step's reply is complete: its whole text when it was not streamed, then `StepFinished`."""
        self.step_done = True
        if not streamed:
            got = from_response(response)
            if got is not None and got.text:
                self._text(got.text)
        self.emit(StepFinished(self.steps, finish_of(None, [response]), usage_of((), [response])))

    def tool_event(self, ev: Any) -> None:
        if isinstance(ev, FunctionToolCallEvent):
            part = ev.part
            try:
                args = part.args_as_dict()
            except Exception:  # malformed arguments: the result event reports the error
                args = {}
            self.emit(ToolCallStarted(part.tool_call_id, part.tool_name, args))
        elif isinstance(ev, FunctionToolResultEvent):
            ret = ev.part
            rec = self.tools.get(ret.tool_call_id)
            if rec is None:  # the handler never ran: an unknown tool or arguments it could not take
                out = ToolOutput(content=_text_of(ret), is_error=True)
                self.emit(ToolCallFinished(ret.tool_call_id, ret.tool_name, out, 0.0, False))
            else:
                self.emit(ToolCallFinished(ret.tool_call_id, ret.tool_name, rec.output, rec.duration_s, rec.truncated))

    def produced(self) -> tuple[Message, ...]:
        """The messages the call produced so far, the interrupted step's partial text included."""
        new = [*self.messages[self.history_len :], *self.pending]
        out = from_model_messages(new, self.tools)
        kept = any(isinstance(m, ModelResponse) and m.state == "interrupted" for m in new)
        if self.step_text and not self.step_done and not kept:
            out = (*out, Message("assistant", (Part("text", text=self.step_text),)))
        return out


def _text_of(part: ToolReturnPart | RetryPromptPart) -> str:
    return part.model_response() if isinstance(part, RetryPromptPart) else part.model_response_str()


def _retries(messages: Sequence[ModelMessage]) -> int:
    """Re-asks in a run: output re-ask prompts plus tool calls returned to the model for a retry."""
    n = 0
    for m in messages:
        for p in getattr(m, "parts", ()):
            if is_retry(p):
                n += 1
    return n


async def _invoke(tool: Tool, args: dict[str, Any]) -> ToolOutput:
    """Run one handler under its timeout; an exception or a timeout is an error result, never a raise."""
    try:
        async with asyncio.timeout(tool.timeout_s):
            out = await tool.handler(args)
    except TimeoutError:
        return ToolOutput(content=f"TimeoutError: tool {tool.name!r} ran past {tool.timeout_s:g} s", is_error=True)
    except Exception as exc:
        return ToolOutput(content=f"{type(exc).__name__}: {exc}", is_error=True)
    if not isinstance(out, ToolOutput):
        return ToolOutput(
            content=f"TypeError: tool {tool.name!r} returned {type(out).__name__}, not ToolOutput", is_error=True
        )
    return out


def _required_check(tool: Tool) -> Callable[..., Any]:
    required = tuple(r for r in tool.parameters.get("required", ()) if isinstance(r, str))

    def check(_ctx: RunContext[None], **args: Any) -> None:
        lacking = [r for r in required if r not in args]
        if lacking:
            raise ModelRetry(f"tool {tool.name!r} needs the argument(s) {', '.join(lacking)}")

    return check


class PydanticAIChat(AdapterCore):
    """The `Chat` port on pydantic-ai; the module docstring states the contract.

    Args:
        model: A fixed pydantic-ai `Model`.
        model_for: A per-request model factory (e.g. `openai_compatible(...)`, the same object a
            `PydanticAIGenerator` takes, so both share one client).
        capture: The capture factory; the default is filled by `openai_compatible`'s response hook.
        gated: Exception types that classify as `gated`.
        wrap: Applied to each model `model_for` yields, e.g. a `replay.Recorder`.
        pricer: Prices a call the gateway did not price.
        gateway: Reads a gateway's response headers: its cost and deployment.
        extra: Classifiers consulted before the shared table, in order.
        on_event: Receives the generator's events (`llm_call_internal_error`, `llm_cost_unknown`,
            `llm_usage_unmeasured`, `llm_wire_retry`, `llm_deadline_expired`, ...).
        policy: How a failed model request is retried when the request carries none.
        limiter: Caps concurrent calls per `ChatRequest.concurrency_group` (else per model id).

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given.
    """

    def __init__(
        self,
        model: Model | None = None,
        *,
        model_for: ChatModelFor | None = None,
        capture: Callable[[], AbstractContextManager[CaptureLike]] = capture,
        gated: tuple[type[BaseException], ...] = (),
        wrap: Callable[[Model], Model] | None = None,
        pricer: Pricer | None = None,
        gateway: GatewayReader | None = None,
        extra: Sequence[Classifier] = (),
        on_event: EventHook | None = None,
        policy: WirePolicy | None = None,
        limiter: Limiter | None = None,
    ) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticAIChat needs exactly one of model= / model_for=")
        super().__init__(
            capture=capture,
            gated=gated,
            wrap=wrap,
            pricer=pricer,
            gateway=gateway,
            extra=extra,
            on_event=on_event,
            policy=policy,
            limiter=limiter,
        )
        self._model = model
        self._model_for = model_for
        self._source = model_for

    async def chat[T](
        self,
        req: ChatRequest,
        output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> Generated[T]:
        """Run `stream` to its end and return `Finished.result`; never raises for a model failure."""
        result: Generated[T] | None = None
        async with aclosing(self.stream(req, output, validate=validate, retry_message=retry_message)) as events:
            async for ev in events:
                if isinstance(ev, Finished):
                    result = ev.result
        assert result is not None  # `stream` always ends with `Finished`
        return result

    async def stream[T](
        self,
        req: ChatRequest,
        output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """Run one call, yielding its events as they happen and `Finished` last, exactly once.

        The result's `stamp` carries `first_event_s` (seconds to the first event; None when there
        was none), `limiter_wait_s` and `wire_requests`.

        Raises:
            asyncio.CancelledError: The consumer was cancelled; the call's task is cancelled first.
        """
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        task = asyncio.create_task(self._produce(req, output, validate, retry_message, queue.put_nowait))
        try:
            while True:
                ev = await queue.get()
                yield ev
                if isinstance(ev, Finished):
                    return
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task})

    async def _produce(
        self,
        req: ChatRequest,
        output: type[Any],
        validate: Callable[[Any], Sequence[Any]] | None,
        retry_message: Callable[[Sequence[Any]], str] | None,
        put: Callable[[StreamEvent], None],
    ) -> None:
        loop = asyncio.get_running_loop()
        run = _Run(put, loop)
        clocks = req.clocks.start(run.started)
        req = replace(req, deadline=clocks)
        wire = self._wire(req.policy, req.model, clocks.at, req.label)
        WIRE.set(wire)  # this task's own context
        self._in_flight += 1
        waited: float | None = None
        try:
            async with asyncio.timeout_at(clocks.at):
                async with self._slot(req.concurrency_group or req.model) as waited_s:
                    waited = waited_s
                    res = await self._converse(req, output, validate, retry_message, run)
        except TimeoutError:
            wire.expire("slot" if waited is None and self._limiter is not None else "request")
            res = self._ended(req, run, None)
        except SlotWaitExceeded as exc:
            res = self._ended(req, run, failure("rate_limited", provider_code="local_limiter", reason=str(exc)))
        except Exception as exc:  # a bug in this adapter: reported, never a model outage
            got = failure("internal", reason=f"{type(exc).__name__}: {exc}")
            self._report_internal(req.label, req.model, exc, got)
            res = self._ended(req, run, got)
        finally:
            self._in_flight -= 1
        res = settled(res, wire, clocks.total_s, waited if self._limiter is not None else None)
        put(Finished(replace(res, stamp={**res.stamp, "first_event_s": run.first_event_s})))

    def _ended(self, req: ChatRequest, run: _Run, got: Failure | None) -> Generated[Any]:
        """The result of a call cut off outside the model loop (deadline, slot, a bug): what it produced so far."""
        return Generated(
            value=None,
            outcome=got.outcome if got is not None else "unavailable",
            model=req.model,
            answered=Answered(requested=req.model),
            failure=got,
            messages=run.produced(),
            steps=run.steps,
        )

    async def _converse(
        self,
        req: ChatRequest,
        output: type[Any],
        validate: Callable[[Any], Sequence[Any]] | None,
        retry_message: Callable[[Sequence[Any]], str] | None,
        run: _Run,
    ) -> Generated[Any]:
        if self._model is not None:
            return await self._run_model(self._model, req, output, validate, retry_message, run)
        assert self._model_for is not None
        try:
            async with self._model_for(req) as model:
                if self._wrap is not None:
                    model = self._wrap(model)
                return await self._run_model(model, req, output, validate, retry_message, run)
        except Exception as exc:  # building the wire failed (bad URL, a refusing transport)
            got = classify_failure(exc, WireFacts(), extra=self._extra, gated=self._gated)
            assert got is not None  # an exception always classifies
            self._report_internal(req.label, req.model, exc, got)
            return self._ended(req, run, got)

    def _tool(self, tool: Tool, req: ChatRequest, run: _Run) -> AITool[None]:
        budget = tool.max_output_chars if tool.max_output_chars is not None else req.tool_output_chars
        loop = asyncio.get_running_loop()

        async def call(ctx: RunContext[None], **args: Any) -> str:
            started = loop.time()
            out = await _invoke(tool, args)
            removed = 0
            if budget is not None:
                text, removed = cut_middle(out.content, budget)
                out = replace(out, content=text) if removed else out
            run.tools[ctx.tool_call_id or tool.name] = ToolRecord(
                out, removed > 0, loop.time() - started, tool.untrusted_output
            )
            if removed:
                run.warnings.append(
                    CallWarning(
                        "clamped", f"tool:{tool.name}", f"tool output cut to {budget} chars ({removed} omitted)"
                    )
                )
            sent = wrap_untrusted(tool.name, out.content) if tool.untrusted_output else out.content
            if out.is_error:
                raise ToolFailed(sent)
            return sent

        return AITool.from_schema(
            call,
            name=tool.name,
            description=tool.description,
            json_schema=dict(tool.parameters),
            takes_ctx=True,
            args_validator=_required_check(tool),
        )

    def _settings(self, req: ChatRequest) -> tuple[dict[str, Any], tuple[CallWarning, ...]]:
        settings = call_settings(
            req.settings, req.clocks, streamed=req.streamed, cache=req.cache_system or req.cache_tools
        )
        warns: tuple[CallWarning, ...] = ()
        if req.tools:
            settings.setdefault("parallel_tool_calls", req.parallel_tools)
        if req.tool_choice == "none":
            settings.setdefault("tool_choice", "none")
        elif req.tool_choice == "required":
            warns = (
                CallWarning("unsupported", "tool_choice", "'required' is not sent: it could never let the call end"),
            )
        elif req.tool_choice != "auto":
            settings.setdefault("tool_choice", ToolOrOutput(function_tools=[req.tool_choice]))
        return settings, warns

    async def _run_model(
        self,
        model: Model,
        req: ChatRequest,
        output: type[Any],
        validate: Callable[[Any], Sequence[Any]] | None,
        retry_message: Callable[[Sequence[Any]], str] | None,
        run: _Run,
    ) -> Generated[Any]:
        settings, warns = self._settings(req)
        agent, kept = build_agent(
            model,
            output,
            settings=settings,
            output_retries=req.output_retries,
            validate=validate,
            retry_message=retry_message,
            instructions=req.system or None,
            tools=[self._tool(t, req, run) for t in req.tools],
        )
        warns = (*warns, *unsupported(model, {**(model.settings or {}), **req.settings}))
        history = to_model_messages(req.messages)
        run.history_len = len(history)
        with self._capture() as cap, capture_run_messages() as messages:
            run.messages = messages
            try:
                resilient = self._resilient(model, req.policy, req.model, req.clocks, cap)
                value, cut_short = await self._loop(agent, resilient, history, req, run)
            except Exception as exc:
                return self._failed(req, run, exc, cap, messages, warns)
        facts = self._facts(cap, messages)
        if cut_short:
            got: Failure | None = failure(
                "truncated", provider_code="max_steps", reason=f"max_steps={req.max_steps} reached"
            )
            value = None
        else:
            got = classify_failure(None, facts.wire, extra=self._extra, gated=self._gated)
            if got is not None and got.kind != "truncated":
                value = None
        res = self._result(
            req.model, value, got, _retries(messages), facts, cap, (*warns, *run.warnings), kept["problems"]
        )
        return replace(res, messages=run.produced(), steps=run.steps)

    def _failed(
        self,
        req: ChatRequest,
        run: _Run,
        exc: Exception,
        cap: CaptureLike,
        messages: Sequence[ModelMessage],
        warns: tuple[CallWarning, ...],
    ) -> Generated[Any]:
        facts = self._facts(cap, messages)
        got = classify_failure(exc, facts.wire, extra=self._extra, gated=self._gated)
        assert got is not None  # an exception always classifies
        if run.step_output:
            got = replace(got, in_stream=True)
            facts.finish = Finish("interrupted", None)
        self._report_internal(req.label, req.model, exc, got)
        res = self._result(req.model, None, got, None, facts, cap, (*warns, *run.warnings), ())
        return replace(res, messages=run.produced(), steps=run.steps)

    async def _loop(
        self, agent: Agent[None, Any], model: Model, history: list[ModelMessage], req: ChatRequest, run: _Run
    ) -> tuple[Any, bool]:
        """Drive the agent node by node; `(output, True)` when `max_steps` stopped it first."""
        async with agent.iter(
            None, message_history=history, model=model, usage_limits=UsageLimits(request_limit=None)
        ) as ai:
            async for node in ai:
                if Agent.is_model_request_node(node):
                    if run.steps >= req.max_steps:
                        run.pending = [node.request]  # the tool results the next step would have sent
                        return None, True
                    run.begin_step()
                    if req.streamed:
                        async with node.stream(ai.ctx) as events:
                            async for ev in events:
                                run.model_event(ev)
                elif Agent.is_call_tools_node(node):
                    run.end_step(node.model_response, streamed=req.streamed)
                    async with node.stream(ai.ctx) as events:
                        async for ev in events:
                            run.tool_event(ev)
            return (ai.result.output if ai.result is not None else None), False


__all__ = ["ChatModelFor", "PydanticAIChat"]
