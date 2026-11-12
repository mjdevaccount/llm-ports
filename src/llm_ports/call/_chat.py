"""The conversation path: role, request, gate, cache plan, pre-flight, trace, `Chat`, graders.

Example:
    async for ev in call.stream(ASSISTANT, messages=history, chat=chat, tools=TOOLS, hooks=HOOKS):
        if isinstance(ev, TextDelta): show(ev.text)
        elif isinstance(ev, Finished): history += ev.result.messages

`chat` is `stream` drained. Per call the role resolves to one `ChatRequest` (`chat_request_for`),
the category is entered with `observability.llm_feature`, the tier's stated capabilities are checked
(`tools`, `json_schema`, `vision`), and the `admit` hook runs. The role's `cache_prefix` is
translated by the model's `cache` capability (`context.cache_plan`), and the pre-flight
(`Role.context`, ADR-0024) counts and checks or trims the history. One trace observation wraps the
`Chat` call; each finished tool becomes a child span under it (`PortObservation.tool`). The
`Finished` event is yielded only after the observation has ended, so a caller that stops reading
at `Finished` never leaves an observation open.

Invariants:
    - `stream` yields `Finished` exactly once, last, and never raises for a model, wire, gate or
      context failure; the outcome names it. A caller's cancellation, or closing the iterator,
      propagates and stops the call.
    - The deadline (the role's timeout) starts when the call is entered and bounds the pre-flight
      count too.
    - A tool that failed, an output the budget cut, and a run that hit `max_steps` each emit an
      event (`llm_tool_failed`, `llm_tool_truncated`, `llm_max_steps`) besides the trace.
    - No fallback chain: a conversation runs on the role's own tier.

Rationale: ADR-0001, ADR-0023, ADR-0024.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import replace
from typing import Any, TypedDict, Unpack

from llm_ports import ports
from llm_ports.call._generate import (
    Hooks,
    _call_failed,
    _event,
    _log_event,
    _NotOk,
    _null_trace,
    _raise_unless_ok,
    _report,
    _settled,
    _stamped,
)
from llm_ports.call._preflight import Preflight, optimistic, preflight
from llm_ports.call.context import cache_plan
from llm_ports.call.models import Capability, missing, needs
from llm_ports.call.roles import Role
from llm_ports.observability import PortObservation, llm_feature
from llm_ports.ports import (
    CallWarning,
    ChatRequest,
    Deadline,
    Finished,
    Generated,
    GenRequest,
    Message,
    RetryMessage,
    StreamEvent,
    TokenCounter,
    Tool,
    ToolCallFinished,
    Validator,
)
from llm_ports.ports.outcomes import REASON_MAX


def chat_request_for(
    role: Role,
    *,
    messages: Sequence[Message],
    system: str = "",
    tools: Sequence[Tool] = (),
    settings: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
) -> ChatRequest:
    """Resolve the role into one conversation request. Pure.

    `settings` override the role's model settings for this call; `max_steps` overrides the role's.
    """
    tier = role.resolved_tier
    info = tier.info
    steps = max_steps or role.max_steps
    req = ChatRequest(
        model=role.resolved_model,
        messages=tuple(messages),
        system=system,
        tools=tuple(tools),
        settings={**role.model_settings(), **(settings or {})},
        self_hosted=tier.self_hosted,
        timeout_s=role.resolved_timeout_s,
        output_retries=role.output_retries,
        label=role.name,
        concurrency_group=None if info is None else (info.group or info.key),
        policy=role.policy if role.policy is not None else tier.wire_policy,
        tool_output_chars=role.tool_output_chars,
    )
    return req if steps is None else replace(req, max_steps=steps)


def _refusal(req: ChatRequest, f: ports.Failure) -> Generated[Any]:
    return Generated(
        value=None,
        outcome=f.outcome,
        model=req.model,
        answered=ports.Answered(requested=req.model),
        failure=f,
        reason=f.reason[:REASON_MAX],
    )


def _refused(role: Role, req: ChatRequest, output: type[Any], h: Hooks) -> Generated[Any] | None:
    """The `request_fault` for a capability the tier lacks, or the `gated` result; None to proceed."""
    info = role.resolved_tier.info
    if info is not None:
        images = any(p.kind == "image" for m in req.messages for p in m.parts)
        tools: frozenset[Capability] = frozenset({"tools"}) if req.tools else frozenset()
        wanted = needs(structured=output is not str, images=images, web=False) | tools
        lacks = missing(info, wanted)
        if lacks:
            reason = f"unsupported: model {info.key!r} ({info.id}) does not state {', '.join(lacks)}"
            _event(h, logging.WARNING, "llm_call_failed", role=role.name, outcome="request_fault", reason=reason)
            return _refusal(req, ports.failure("unsupported", reason=reason))
    why = h.admit(role) if h.admit is not None else None
    return None if why is None else _refusal(req, ports.failure("gated", reason=why))


def _planned(role: Role, req: ChatRequest) -> tuple[ChatRequest, tuple[CallWarning, ...]]:
    """The cache hint and the streaming switch, translated by the tier's stated capabilities."""
    info = role.resolved_tier.info
    hint, settings, warns = cache_plan(info, wanted=role.cache_prefix, settings=req.settings)
    streamed = info is None or "stream" in info.supports
    if not streamed:
        warns = (*warns, CallWarning("unsupported", "stream", f"{info.key if info else req.model} states no stream"))
    return replace(req, cache_system=hint, cache_tools=hint, settings=settings, streamed=streamed), warns


@asynccontextmanager
async def _closing(it: AsyncIterator[Any]) -> AsyncIterator[None]:
    """Close a port's event iterator on exit, when it can be closed (an async generator can)."""
    try:
        yield
    finally:
        close = getattr(it, "aclose", None)
        if close is not None:
            await close()


def _last_user(messages: Sequence[Message]) -> str:
    return next((m.text for m in reversed(messages) if m.role == "user"), "")


def _tool_meta(done: Sequence[ToolCallFinished]) -> dict[str, Any]:
    return {
        "tool_names": [t.name for t in done],
        "tool_durations_s": [round(t.duration_s, 4) for t in done],
        "tool_errors": sum(1 for t in done if t.output.is_error),
    }


def _tool_events(role: Role, res: Generated[Any], done: Sequence[ToolCallFinished], h: Hooks) -> None:
    for t in done:
        if t.output.is_error:
            _event(h, logging.WARNING, "llm_tool_failed", role=role.name, tool=t.name, detail=t.output.content[:200])
        if t.truncated:
            _event(h, logging.INFO, "llm_tool_truncated", role=role.name, tool=t.name)
    if res.failure is not None and res.failure.provider_code == "max_steps":
        _event(h, logging.WARNING, "llm_max_steps", role=role.name, steps=res.steps)


def _merged(res: Generated[Any], pre: Preflight, warns: tuple[CallWarning, ...]) -> Generated[Any]:
    return replace(res, warnings=(*warns, *pre.warnings, *res.warnings), stamp={**res.stamp, **pre.stamp()})


class TurnOptions(TypedDict, total=False):
    """The optional keywords `stream` and `chat` share; each one omitted takes the default named."""

    tools: Sequence[Tool]
    """The tools offered to the model (default none)."""
    hooks: Hooks | None
    """Trace, admit, graders, report and event hooks (default `Hooks()`)."""
    system: str
    """The system prompt (default empty)."""
    settings: Mapping[str, Any] | None
    """Model settings merged over the role's (default the role's)."""
    metadata: Mapping[str, Any] | None
    """Extra trace metadata (default none)."""
    max_steps: int | None
    """Overrides `Role.max_steps` for this call (default the role's)."""
    counter: TokenCounter | None
    """Measures the history for the pre-flight (`Role.context`); None counts nothing."""


async def stream[T](
    role: Role,
    *,
    messages: Sequence[Message],
    chat: ports.Chat,
    output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
    validate: Validator[T] | None = None,
    retry_message: RetryMessage | None = None,
    **opts: Unpack[TurnOptions],
) -> AsyncGenerator[StreamEvent]:
    """Run one conversation turn for `role` through `chat`, yielding its events and `Finished` last.

    The path is described in the module docstring. The caller appends `Finished.result.messages`
    to its own history; the port keeps none.

    Args:
        role: The role: its tier, timeout, context policy, cache prefix and graders.
        messages: The history, oldest first; the caller owns it.
        chat: The `Chat` port the call runs on (`Backends.chat_for`).
        output: `str` for prose, a pydantic model, or a `JsonOutput` subclass.
        validate: Lists the problems with a parsed value; a non-empty list re-asks.
        retry_message: Builds a re-ask's message from the problems.
        **opts: The `TurnOptions`.

    Raises:
        asyncio.CancelledError: The caller cancelled the call.
    """
    tools = opts.get("tools", ())
    system = opts.get("system", "")
    counter = opts.get("counter")
    metadata = opts.get("metadata") or {}
    h = opts.get("hooks") or Hooks()
    emit = h.on_event or _log_event
    loop = asyncio.get_running_loop()
    clocks = Deadline(total_s=role.resolved_timeout_s).start(loop.time())
    with llm_feature(role.category):
        req = chat_request_for(
            role,
            messages=messages,
            system=system,
            tools=tools,
            settings=opts.get("settings"),
            max_steps=opts.get("max_steps"),
        )
        req = replace(req, deadline=clocks)
        refused = _refused(role, req, output, h)
        if refused is not None:
            yield Finished(_stamped(role, refused))
            return
        req, warns = _planned(role, req)
        try:
            async with asyncio.timeout_at(clocks.at):
                pre = await preflight(role, counter, system=system, messages=req.messages, tools=req.tools, emit=emit)
        except TimeoutError:
            f = ports.failure("timeout", phase="total", reason=f"deadline: the call's {clocks.total_s:g} s ran out")
            yield Finished(_stamped(role, _refusal(req, f)))
            return
        if pre.refused is not None:
            yield Finished(_stamped(role, _merged(_refusal(req, pre.refused), pre, warns)))
            return
        req = replace(req, messages=pre.messages)
        prompt = _last_user(pre.messages)
        greq = GenRequest(model=req.model, prompt=prompt, system=system, settings=req.settings, label=role.name)
        res: Generated[T] | None = None
        done: list[ToolCallFinished] = []
        trace = h.trace or _null_trace
        try:
            async with trace(role, greq, prompt, {"role": role.name, **metadata}) as obs:
                inner = chat.stream(req, output, validate=validate, retry_message=retry_message)
                async with _closing(inner):
                    async for ev in inner:
                        if isinstance(ev, Finished):
                            res = ev.result
                            continue
                        if isinstance(ev, ToolCallFinished):
                            done.append(ev)
                            if isinstance(obs, PortObservation):
                                obs.tool(
                                    name=ev.name,
                                    call_id=ev.call_id,
                                    duration_s=ev.duration_s,
                                    is_error=ev.output.is_error,
                                )
                        yield ev
                if res is None:
                    res = _refusal(req, ports.failure("internal", reason="the Chat stream ended without Finished"))
                res = _merged(res, pre, warns)
                res = _settled(role, prompt, res, obs, h, {"steps": res.steps, **_tool_meta(done)})
                _raise_unless_ok(res)
        except _NotOk:
            _call_failed(h, role, res)
        if res is None:  # reachable only when a trace hook swallowed an exception
            res = _refusal(req, ports.failure("internal", reason="the trace hook swallowed the call"))
        _tool_events(role, res, done, h)
        optimistic(pre, res.failure, emit, role.name)
        res = _stamped(role, res)
        await _report(role, prompt, res, h)
    yield Finished(res)


async def chat[T](
    role: Role,
    *,
    messages: Sequence[Message],
    chat: ports.Chat,
    output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
    validate: Validator[T] | None = None,
    retry_message: RetryMessage | None = None,
    **opts: Unpack[TurnOptions],
) -> Generated[T]:
    """Run one conversation turn for `role` and return its result: `stream` drained.

    Takes exactly `stream`'s arguments.

    Raises:
        asyncio.CancelledError: The caller cancelled the call.
    """
    events = stream(
        role, messages=messages, chat=chat, output=output, validate=validate, retry_message=retry_message, **opts
    )
    async with aclosing(events):
        async for ev in events:
            if isinstance(ev, Finished):
                return ev.result
    raise RuntimeError("call.stream ended without Finished")  # unreachable: stream always yields it


__all__ = ["TurnOptions", "chat", "chat_request_for", "stream"]
