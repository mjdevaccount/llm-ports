"""The wire capture: what every OpenAI-wire response of one call showed, before pydantic-ai read it.

pydantic-ai drops non-standard keys (`sources`), folds an absent usage block into zeros on some
paths, and keeps no response headers. The capture is an httpx2 `response` event hook on the
client `openai_compatible` and `openai_compatible_embeddings` build: each response of the call is
recorded into the capture the call opened (a `ContextVar`, so concurrent calls never share one).

A 2xx reply becomes a `WireReply` (its usage, finish reason, model, id, refusal and headers); any
response, 2xx or not, becomes the call's latest `WireFacts`.

Invariants:
    - The hook never raises; an unreadable body is skipped.
    - With no capture open, the hook does nothing.
    - A 2xx `text/event-stream` response is not read: reading it would buffer the whole stream
      before its first event reached the caller. Its usage and finish come from pydantic-ai's
      own parse of the stream.
    - Usage keeps unknown as None: an absent usage block is one unmeasured request, an absent
      count is None, a reported 0 is 0.

Rationale: ADR-0007, ADR-0019.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from llm_ports._extras import extra
from llm_ports.adapters._classify import ERROR_MESSAGE_MAX, WireFacts
from llm_ports.ports import Usage

with extra("pydantic-ai"):
    import httpx2

#: Usage keys read into a named bucket; every other integer key goes to `Usage.details`.
_NAMED = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})
_NAMED_DETAILS = frozenset({"prompt_tokens_details.cached_tokens", "completion_tokens_details.reasoning_tokens"})


class CaptureLike(Protocol):
    """What `PydanticAIGenerator` reads from a capture after a call; the type of its `capture=`.

    A capture may also carry `replies` (a sequence of `WireReply`) and `last` (the latest
    `WireFacts`); the generator reads them when present and falls back to pydantic-ai's own
    messages when not.
    """

    @property
    def finish_reason(self) -> str | None:
        """The last reply's `finish_reason`, or None."""

    @property
    def usage(self) -> Mapping[str, int | None]:
        """`input` and `output` token counts; None for a count the server did not report."""

    @property
    def sources(self) -> tuple[str, ...] | None:
        """The reply's top-level `sources` strings, or None."""


@dataclass(frozen=True)
class WireReply:
    """One 2xx reply of a call, as the wire showed it.

    Attributes:
        usage: The reply's usage, one request; unmeasured when it carried no usage block.
        finish_raw: The first choice's `finish_reason`, verbatim.
        model: The `model` the body reported.
        response_id: The body's `id`.
        refusal: True when the first choice's message carried a non-empty `refusal`.
        headers: The response headers, names lower-cased.
    """

    usage: Usage
    finish_raw: str | None = None
    model: str | None = None
    response_id: str | None = None
    refusal: bool = False
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass
class _Capture:
    """What the wire showed during one call.

    Attributes:
        finish_reason: The last reply's `finish_reason` (the 1.x field).
        usage: The last reply's `input` and `output` counts (the 1.x field).
        sources: The last reply's non-standard top-level `sources` strings, or None.
        replies: Every 2xx reply, in order.
        last: The latest response's facts, whatever its status; None before any response.
    """

    finish_reason: str | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    sources: tuple[str, ...] | None = None
    replies: list[WireReply] = field(default_factory=list)
    last: WireFacts | None = None


_active: ContextVar[_Capture | None] = ContextVar("llm_ports_capture", default=None)


@contextmanager
def capture() -> Iterator[_Capture]:
    """Open a capture for one call; `record_response` fills it from inside the httpx2 client."""
    cap = _Capture()
    token = _active.set(cap)
    try:
        yield cap
    finally:
        try:
            _active.reset(token)
        except ValueError:  # reset from another context: clear rather than leak this call's capture
            _active.set(None)


def _count(v: Any) -> int | None:
    """A token count the wire reported, or None (absent, null, a bool or not an integer)."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def parse_usage(block: Any) -> Usage:
    """Return one reply's `Usage` from its OpenAI-wire `usage` block. Pure.

    `prompt_tokens` and `completion_tokens` are the inclusive totals; `prompt_tokens_details.cached_tokens`
    (OpenAI, llama.cpp) or `cache_read_input_tokens` (a gateway relaying Anthropic) is the cached
    part; `cache_creation_input_tokens` the written part; `completion_tokens_details.reasoning_tokens`
    the reasoning part. Other integer keys go to `details` under their own names (nested ones as
    `parent.key`). No block at all is one unmeasured request.
    """
    if not isinstance(block, dict):
        return Usage(requests=1, unmeasured=1)
    details: dict[str, int] = {}
    for k, v in block.items():
        if k not in _NAMED and _count(v) is not None:
            details[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                name = f"{k}.{kk}"
                if name not in _NAMED_DETAILS and _count(vv) is not None:
                    details[name] = vv
    prompt_details = block.get("prompt_tokens_details")
    completion_details = block.get("completion_tokens_details")
    cached = _count(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None
    reasoning = _count(completion_details.get("reasoning_tokens")) if isinstance(completion_details, dict) else None
    return Usage(
        input=_count(block.get("prompt_tokens")),
        output=_count(block.get("completion_tokens")),
        cache_read=cached if cached is not None else details.pop("cache_read_input_tokens", None),
        cache_write=details.pop("cache_creation_input_tokens", None),
        reasoning=reasoning,
        requests=1,
        details=details,
    )


def _str(v: Any) -> str | None:
    if isinstance(v, str):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    return None


def _error_facts(status: int, headers: Mapping[str, str], body: Any) -> WireFacts:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, str):
        return WireFacts(status=status, headers=headers, error_message=err[:ERROR_MESSAGE_MAX])
    if not isinstance(err, dict):
        err = body if isinstance(body, dict) else {}
    message = _str(err.get("message"))
    return WireFacts(
        status=status,
        headers=headers,
        error_code=_str(err.get("code")),
        error_type=_str(err.get("type")),
        error_message=message[:ERROR_MESSAGE_MAX] if message is not None else None,
    )


def _reply(body: dict[str, Any], headers: Mapping[str, str]) -> WireReply:
    choices = body.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    refusal = isinstance(message, dict) and bool(message.get("refusal"))
    return WireReply(
        usage=parse_usage(body.get("usage")),
        finish_raw=_str(choice.get("finish_reason")),
        model=_str(body.get("model")),
        response_id=_str(body.get("id")),
        refusal=refusal,
        headers=headers,
    )


async def record_response(response: httpx2.Response) -> None:
    """Record one response into the active capture (an httpx2 `response` event hook).

    Invariants:
        - Never raises; an unreadable body is ignored.
        - With no capture open, does nothing.
    """
    cap = _active.get()
    if cap is None:
        return
    if response.status_code < 400 and _is_stream(response):
        return  # reading it here would buffer the whole stream before the caller saw one event
    try:
        await response.aread()
        headers = {k.lower(): v for k, v in response.headers.items()}
        try:
            body = json.loads(response.content or b"{}")
        except ValueError:
            body = None
        if response.status_code >= 400:
            cap.last = _error_facts(response.status_code, headers, body)
        if not isinstance(body, dict):
            return
        reply = _reply(body, headers)
        _record_1x(cap, reply, body)
        if response.status_code >= 400:
            return
        cap.replies.append(reply)
        cap.last = WireFacts(
            status=response.status_code, headers=headers, finish_raw=reply.finish_raw, refusal=reply.refusal
        )
    except Exception:  # telemetry must not break the call it describes
        return


def _is_stream(response: httpx2.Response) -> bool:
    """True for a server-sent-event body: its usage and finish arrive in pydantic-ai's own stream parse."""
    return response.headers.get("content-type", "").split(";")[0].strip().lower() == "text/event-stream"


def _record_1x(cap: _Capture, reply: WireReply, body: dict[str, Any]) -> None:
    """The 1.x fields, as 1.22 recorded them: from every JSON body, an error body included."""
    cap.finish_reason = reply.finish_raw
    cap.usage = dict(reply.usage.as_mapping())
    src = body.get("sources")
    cap.sources = tuple(u for u in src if isinstance(u, str)) if isinstance(src, list) else None


__all__ = ["CaptureLike", "WireReply", "parse_usage"]
