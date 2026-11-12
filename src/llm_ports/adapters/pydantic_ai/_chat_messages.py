"""The mapping between the `Chat` port's `Message` and pydantic-ai's model messages.

Outbound, a tool result goes into pydantic-ai's tool-return part (the provider's tool-result
channel), wrapped in the `<tool_output ... untrusted="true">` boundary marker when it is
untrusted; reasoning is never sent back. Inbound, every message a run produced becomes a
`Message`: assistant text, reasoning and tool calls; tool results (the handler's raw output, not
the wrapped wire text); and a re-ask prompt as a user turn.

Invariants:
    - Pure, no I/O.
    - A tool result is never concatenated into user or system text.

Rationale: ADR-0023.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from xml.sax.saxutils import quoteattr

from llm_ports._extras import extra
from llm_ports.ports import Message, Part, ToolOutput

with extra("pydantic-ai"):
    from pydantic_ai import BinaryContent, CachePoint
    from pydantic_ai import messages as _messages
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelRequestPart,
        ModelResponse,
        ModelResponsePart,
        RetryPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )


#: pydantic-ai's output re-ask part (`RetryFeedbackPart`, newer releases), alongside the deprecated
#: `RetryPromptPart`; an empty tuple on a release that predates it.
FEEDBACK_PARTS: tuple[type, ...] = tuple(t for t in (getattr(_messages, "RetryFeedbackPart", None),) if t is not None)


def is_retry(p: Any) -> bool:
    """True for a part that re-asks the model: an output re-ask, or a tool call returned for a retry."""
    return isinstance(p, (RetryPromptPart, *FEEDBACK_PARTS)) or (
        isinstance(p, ToolReturnPart) and getattr(p, "outcome", None) == "retried"
    )


@dataclass(frozen=True, slots=True)
class ToolRecord:
    """One tool call as the adapter ran it.

    Attributes:
        output: The handler's output after the budget cut, before the boundary marker.
        truncated: True when the budget cut it.
        duration_s: How long the handler ran.
        untrusted: Whether the tool's output is untrusted.
    """

    output: ToolOutput
    truncated: bool
    duration_s: float
    untrusted: bool


def wrap_untrusted(name: str, content: str) -> str:
    """The boundary marker around an untrusted tool output. A marker a system prompt can name, not a defence."""
    return f'<tool_output name={quoteattr(name)} untrusted="true">\n{content}\n</tool_output>'


def _user(m: Message) -> ModelRequest:
    content: list[Any] = []
    for p in m.parts:
        if p.kind == "text" and p.text:
            content.append(p.text)
        elif p.kind == "image" and p.data is not None:
            content.append(BinaryContent(data=p.data, media_type=p.media_type or "application/octet-stream"))
    if m.cache:
        content.append(CachePoint())
    if len(content) == 1 and isinstance(content[0], str):
        return ModelRequest(parts=[UserPromptPart(content[0])])
    return ModelRequest(parts=[UserPromptPart(content)])


def _assistant(m: Message) -> ModelResponse:
    parts: list[ModelResponsePart] = []
    for p in m.parts:
        if p.kind == "text" and p.text:
            parts.append(TextPart(content=p.text))
        elif p.kind == "tool_call" and p.name is not None:
            parts.append(ToolCallPart(tool_name=p.name, args=dict(p.args or {}), tool_call_id=p.call_id or ""))
        # reasoning is never sent back
    return ModelResponse(parts=parts)


def _tool(m: Message) -> ModelRequest:
    parts: list[ModelRequestPart] = []
    for p in m.parts:
        if p.kind != "tool_result":
            continue
        name = p.name or ""
        text = p.text or ""
        parts.append(
            ToolReturnPart(
                tool_name=name,
                content=wrap_untrusted(name, text) if p.untrusted else text,
                tool_call_id=p.call_id or "",
            )
        )
    return ModelRequest(parts=parts)


def to_model_messages(messages: Sequence[Message]) -> list[ModelMessage]:
    """The history as pydantic-ai model messages, in order."""
    out: list[ModelMessage] = []
    for m in messages:
        if m.role == "user":
            out.append(_user(m))
        elif m.role == "assistant":
            out.append(_assistant(m))
        else:
            out.append(_tool(m))
    return out


def _args(part: ToolCallPart) -> Mapping[str, Any]:
    try:
        return part.args_as_dict()
    except Exception:  # malformed arguments from the model: the call itself is reported as an error result
        return {}


def from_response(r: ModelResponse) -> Message | None:
    """One model response as an assistant message; None when it holds nothing the port carries."""
    parts: list[Part] = []
    for p in r.parts:
        if isinstance(p, TextPart) and p.content:
            parts.append(Part("text", text=p.content))
        elif isinstance(p, ThinkingPart) and p.content:
            parts.append(Part("reasoning", text=p.content))
        elif isinstance(p, ToolCallPart):
            parts.append(Part("tool_call", call_id=p.tool_call_id, name=p.tool_name, args=_args(p)))
    return Message("assistant", tuple(parts)) if parts else None


def _from_request(r: ModelRequest, records: Mapping[str, ToolRecord]) -> list[Message]:
    tool: list[Part] = []
    user: list[Part] = []
    for p in r.parts:
        if isinstance(p, ToolReturnPart):
            rec = records.get(p.tool_call_id)
            text = rec.output.content if rec is not None else p.model_response_str()
            untrusted = rec.untrusted if rec is not None else False
            tool.append(Part("tool_result", text=text, call_id=p.tool_call_id, name=p.tool_name, untrusted=untrusted))
        elif isinstance(p, RetryPromptPart):
            if p.tool_name is not None:
                tool.append(Part("tool_result", text=p.model_response(), call_id=p.tool_call_id, name=p.tool_name))
            else:
                user.append(Part("text", text=p.model_response()))
        elif FEEDBACK_PARTS and isinstance(p, FEEDBACK_PARTS):
            user.append(Part("text", text=str(cast("Any", p).model_response())))
        elif isinstance(p, UserPromptPart) and isinstance(p.content, str):
            user.append(Part("text", text=p.content))
    out = [Message("tool", tuple(tool))] if tool else []
    return out + ([Message("user", tuple(user))] if user else [])


def from_model_messages(messages: Sequence[ModelMessage], records: Mapping[str, ToolRecord]) -> tuple[Message, ...]:
    """The messages a run produced, as port messages, in order."""
    out: list[Message] = []
    for m in messages:
        if isinstance(m, ModelResponse):
            got = from_response(m)
            if got is not None:
                out.append(got)
        else:
            out.extend(_from_request(m, records))
    return tuple(out)


__all__: list[str] = []
