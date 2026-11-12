"""The value types of the `Chat` port: messages, tools, the request, and the stream's events.

A chat call is a multi-turn, tool-using, streamed conversation on the same call path as a one-shot
generation (same roles, gate, trace, usage, cost, failures, deadline and retry). History is the
caller's: the port is stateless, takes `messages` in and returns the new ones on
`Generated.messages`, ready to append.

Invariants:
    - No I/O; every type is frozen.
    - A `Tool` handler is a coroutine function: a synchronous one could not be cancelled when its
      timeout or the call's deadline runs out, so it is refused at construction.
    - A tool's output is data, never instruction: it travels in the provider's tool-result channel
      and is `untrusted` unless the tool says otherwise.
    - `Finished` is the single terminal event of a stream: exactly once, last.

Rationale: ADR-0023.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_ports.ports._resilience import Deadline, WirePolicy
from llm_ports.ports._types import Generated
from llm_ports.ports._usage import Finish, Usage

#: What one `Part` of a message holds.
PartKind = Literal["text", "image", "tool_call", "tool_result", "reasoning"]
#: Who a `Message` is from.
ChatRole = Literal["user", "assistant", "tool"]
#: The marker a cut tool output carries in place of what was removed; `{n}` is the count.
OMITTED_MARKER = "[... {n} chars omitted]"


@dataclass(frozen=True, slots=True)
class Part:
    """One piece of a message.

    Attributes:
        kind: What the part holds.
        text: The text of a `text`, `reasoning` or `tool_result` part.
        data: An image's bytes.
        media_type: An image's MIME type.
        call_id: The id that pairs a `tool_call` with its `tool_result`.
        name: The tool's name, on a `tool_call` or a `tool_result`.
        args: A `tool_call`'s arguments.
        untrusted: True for a `tool_result` whose content came from outside (the default for a
            tool's output); the adapter marks it as data when it sends it.
    """

    kind: PartKind
    text: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    call_id: str | None = None
    name: str | None = None
    args: Mapping[str, Any] | None = None
    untrusted: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation.

    Attributes:
        role: `user`, `assistant` (text, reasoning and tool calls) or `tool` (tool results).
        parts: The turn's parts, in order.
        cache: Ask for a prompt-cache breakpoint after this message (ADR-0024). A request, not a
            guarantee: `Usage.cache_read` on the result says whether it was served.
    """

    role: ChatRole
    parts: tuple[Part, ...]
    cache: bool = False

    @property
    def text(self) -> str:
        """The message's `text` parts, joined; reasoning and tool parts are not included."""
        return "".join(p.text or "" for p in self.parts if p.kind == "text")


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """What a tool handler returns.

    Attributes:
        content: What the model is shown.
        is_error: True when the tool failed; the model sees the content and may recover.
        meta: For the consumer's UI; never sent to the model.
    """

    content: str
    is_error: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)


#: A tool handler: the call's arguments to its output. Must be a coroutine function.
ToolHandler = Callable[[Mapping[str, Any]], Awaitable[ToolOutput]]


def _is_async(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(type(fn).__call__)


@dataclass(frozen=True, slots=True)
class Tool:
    """A client-side tool the model may call.

    Attributes:
        name: The name the model calls it by.
        description: When and why to call it, for the model.
        parameters: The arguments' JSON Schema, of type `object`.
        handler: Runs one call; a coroutine function.
        timeout_s: The most one call may take; past it the call is cancelled and the model is shown
            an error result.
        untrusted_output: What the tool returns is data from outside (the default): it is wrapped
            in a `<tool_output ... untrusted="true">` boundary marker. The marker is something a
            system prompt can name; it is not a defence on its own.
        max_output_chars: The longest output the model is shown; None leaves it to the request's
            `tool_output_chars`. A longer one is cut in the middle with a marker.

    Raises:
        TypeError: `handler` is not a coroutine function.
        ValueError: `parameters` is not an object schema, or `timeout_s` is not positive.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler
    timeout_s: float = 30.0
    untrusted_output: bool = True
    max_output_chars: int | None = None

    def __post_init__(self) -> None:
        if not _is_async(self.handler):
            raise TypeError(f"tool {self.name!r}: handler must be a coroutine function (a sync one cannot be stopped)")
        if self.parameters.get("type") != "object":
            raise ValueError(f"tool {self.name!r}: parameters must be a JSON Schema of type 'object'")
        if self.timeout_s <= 0:
            raise ValueError(f"tool {self.name!r}: timeout_s must be positive, not {self.timeout_s}")


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One chat call, already resolved from a role.

    Attributes:
        model: The model id.
        messages: The history, oldest first; the last is normally the user's turn.
        system: The system prompt; empty for none. Never trimmed.
        tools: The tools the model may call.
        tool_choice: `auto`, `none`, `required`, or one tool's name.
        max_steps: The most model requests in the call, tool rounds and re-asks included. Reaching
            it with a tool call still pending is `truncated` (`provider_code="max_steps"`).
        parallel_tools: Let the model ask for several tool calls in one step.
        settings: Sent to the model verbatim (max_tokens, temperature, ...).
        deadline: The call's clocks; None means `Deadline(total_s=timeout_s)`.
        timeout_s: The call's bound when `deadline` is None, in seconds.
        concurrency_group: The `Limiter` group; None means the model id.
        cache_system: Ask for a cache breakpoint after the system prompt (ADR-0024).
        cache_tools: Ask for a cache breakpoint after the tool definitions.
        label: The call's name in an adapter's logs.
        self_hosted: True for a self-hosted tier.
        output_retries: Re-asks for an output or a tool call the model got wrong.
        policy: How the adapter retries a wire request; None is the adapter's default.
        tool_output_chars: The request's tool-output budget, for a tool that sets none; None for
            no cut.
        streamed: Stream each model reply as it is generated. False (a model that cannot stream)
            sends each step's whole text as one `TextDelta`.

    Raises:
        ValueError: `max_steps` is below 1.
    """

    model: str
    messages: tuple[Message, ...]
    system: str = ""
    tools: tuple[Tool, ...] = ()
    tool_choice: str = "auto"
    max_steps: int = 8
    parallel_tools: bool = True
    settings: Mapping[str, Any] = field(default_factory=dict)
    deadline: Deadline | None = None
    timeout_s: float = 120.0
    concurrency_group: str | None = None
    cache_system: bool = False
    cache_tools: bool = False
    label: str = "chat"
    self_hosted: bool = True
    output_retries: int = 1
    policy: WirePolicy | None = None
    tool_output_chars: int | None = None
    streamed: bool = True

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, not {self.max_steps}")

    @property
    def clocks(self) -> Deadline:
        """`deadline`, or `Deadline(total_s=timeout_s)` when none was given."""
        return self.deadline if self.deadline is not None else Deadline(total_s=self.timeout_s)


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of the reply's text, as it was generated."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """A piece of the model's reasoning; never part of the value, never sent back to a provider."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    """The model called a tool; the handler is about to run."""

    call_id: str
    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    """A tool call ended.

    Attributes:
        call_id: The call's id.
        name: The tool's name.
        output: What the handler returned (before the untrusted marker); an error result when it
            raised, timed out, or the tool or its arguments were wrong.
        duration_s: How long the handler ran, in seconds; 0 when it never ran.
        truncated: True when the output was cut to its budget.
    """

    call_id: str
    name: str
    output: ToolOutput
    duration_s: float
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class StepFinished:
    """One model request of the call ended.

    Attributes:
        n: The step's number, from 1.
        finish: How the reply ended; None when the provider said nothing.
        usage: The step's own usage; None when unmeasured.
    """

    n: int
    finish: Finish | None
    usage: Usage | None


@dataclass(frozen=True, slots=True)
class Finished[T]:
    """The stream's last event, exactly once, in every case: success, failure, deadline."""

    result: Generated[T]


#: One event of a chat stream.
StreamEvent = TextDelta | ReasoningDelta | ToolCallStarted | ToolCallFinished | StepFinished | Finished[Any]

__all__ = [
    "OMITTED_MARKER",
    "ChatRequest",
    "ChatRole",
    "Finished",
    "Message",
    "Part",
    "PartKind",
    "ReasoningDelta",
    "StepFinished",
    "StreamEvent",
    "TextDelta",
    "Tool",
    "ToolCallFinished",
    "ToolCallStarted",
    "ToolHandler",
    "ToolOutput",
]
