"""Record a live model's replies once and replay them in CI, for pydantic-ai models.

Example:
    rec = Recorder(real_model)    # wraps any pydantic-ai Model; keeps every reply
    ...                           # run the pipeline once against the live model
    json.dump(rec.replies, f)     # {key(messages): reply text}
    model = replay(json.load(f))  # a FunctionModel answering from that dict

`key(messages)` (version 1) is the sha of the first user prompt plus the attempt number (the count
of model responses already in the conversation, so an output retry's second attempt has its own
key). `key_v2` (version 2, 1.24) covers everything that changes the answer: the model, the system
prompt and instructions, the whole message list, the settings (sorted) and the output schema, plus
the attempt; a system-prompt change no longer replays the old answer. `Recorder(version=2)` writes
`{"version": 2, "entries": {...}}` (`cassette()`); `replay` reads both shapes. A changed request
misses the lookup, the call raises `NotRecorded` naming the parts that differ from the nearest
entry, and the golden set must be re-recorded. `ReplayModel.unplayed()` names the entries a run
never asked for (the `llm_replay` pytest fixture fails a partly played cassette with it).

A version-2 `Recorder` given `secrets` refuses (`SecretInCassette`) to keep a reply containing any
of them. Only the reply text is kept: no headers, no request bodies.

`Recording` keeps every raw reply parsed in `attempts` (an agent returns only the last attempt; a
grader wants every one); an unparsable reply is kept as `{"_unparsed": head}`, never dropped.
Either wrapper plugs into `PydanticAIGenerator` as its `wrap=` model wrapper; `recording(make)` and
`recorder(make)` build a generator with the wrapper on and return it with the list or dict it fills.

The module depends on pydantic-ai, so it is imported by its path and needs the `pydantic-ai` extra.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from llm_ports._extras import extra
from llm_ports.prompts import sha

with extra("pydantic-ai"):
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )
    from pydantic_ai.models import Model, ModelRequestParameters
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.models.wrapper import WrapperModel
    from pydantic_ai.settings import ModelSettings

#: Characters of an unparsable reply that `Recording` keeps.
UNPARSED_HEAD = 2000


#: The cassette shape `Recorder(version=2)` writes.
CASSETTE_VERSION = 2
#: The parts `key_v2` covers, in key order.
KEY_PARTS = ("model", "system", "messages", "settings", "schema", "attempt")


class NotRecorded(RuntimeError):
    """The replay has no reply for this request: it changed since the recording."""


class SecretInCassette(ValueError):
    """A reply about to be recorded contains a bound secret; nothing was written."""


def key(messages: list[ModelMessage]) -> str:
    """Return the recording's lookup key, `<sha of the first user prompt>:<attempt>`."""
    first = next(
        (p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)),
        "",
    )
    attempt = sum(isinstance(m, ModelResponse) for m in messages)
    return f"{sha(str(first))}:{attempt}"


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str, separators=(",", ":"))


def _part(p: Any) -> list[Any]:
    """A part's content without timestamps or generated ids."""
    if isinstance(p, ToolCallPart):
        return [p.part_kind, p.tool_name, p.args_as_dict()]
    return [getattr(p, "part_kind", type(p).__name__), getattr(p, "tool_name", None), getattr(p, "content", None)]


def _system(messages: Sequence[ModelMessage]) -> str:
    out = []
    for m in messages:
        if isinstance(m, ModelRequest):
            out += [p.content for p in m.parts if isinstance(p, SystemPromptPart)]
            if m.instructions:
                out.append(m.instructions)
    return "\n".join(out)


def _schema(params: ModelRequestParameters) -> Any:
    if params.output_object is not None:
        return params.output_object.json_schema
    return [[t.name, t.parameters_json_schema] for t in params.output_tools] or None


def key_parts(
    messages: Sequence[ModelMessage],
    *,
    model: str,
    system: str,
    settings: Mapping[str, Any] | None,
    output_schema: Any,
    attempt: int,
) -> dict[str, str]:
    """The digest of each part `key_v2` covers (`KEY_PARTS`). Pure."""
    msgs = [[m.kind, [_part(p) for p in m.parts]] for m in messages]
    return {
        "model": model,
        "system": sha(system),
        "messages": sha(_canon(msgs)),
        "settings": sha(_canon(dict(sorted((settings or {}).items())))),
        "schema": sha(_canon(output_schema)),
        "attempt": str(attempt),
    }


def key_v2(
    messages: Sequence[ModelMessage],
    *,
    model: str,
    system: str,
    settings: Mapping[str, Any] | None,
    output_schema: Any,
    attempt: int,
) -> str:
    """Return the version-2 lookup key, `<sha of every part but the attempt>:<attempt>`. Pure."""
    parts = key_parts(
        messages, model=model, system=system, settings=settings, output_schema=output_schema, attempt=attempt
    )
    return _key_of(parts)


def _v2(messages: list[ModelMessage], model: str, settings: Any, params: ModelRequestParameters) -> dict[str, str]:
    attempt = sum(isinstance(m, ModelResponse) for m in messages)
    return key_parts(
        messages,
        model=model,
        system=_system(messages),
        settings=dict(settings or {}),
        output_schema=_schema(params),
        attempt=attempt,
    )


def _key_of(parts: Mapping[str, str]) -> str:
    return f"{sha(_canon([parts[k] for k in KEY_PARTS[:-1]]))}:{parts['attempt']}"


def _text(resp: ModelResponse) -> str:
    return "".join(p.content for p in resp.parts if isinstance(p, TextPart))


class Recorder(WrapperModel):
    """A model wrapper that keeps every reply's text in `replies`.

    Version 1 keys by `key(messages)` and keeps the text; version 2 keys by `key_v2` and keeps
    `{"text", "parts"}` (the key's parts, for `NotRecorded`'s diff). Pass `replies` to share one
    dict across several wrapped models.

    Args:
        wrapped: The live model.
        replies: The dict to fill; None starts an empty one.
        version: 1 (the 1.x key, the default) or 2.
        secrets: Values that must never be written; a reply containing one raises
            `SecretInCassette` (version 2).
    """

    def __init__(
        self,
        wrapped: Any,
        replies: dict[str, Any] | None = None,
        *,
        version: int = 1,
        secrets: Iterable[str] = (),
    ) -> None:
        super().__init__(wrapped)
        if version not in (1, CASSETTE_VERSION):
            raise ValueError(f"version must be 1 or {CASSETTE_VERSION}, not {version}")
        self.replies: dict[str, Any] = replies if replies is not None else {}
        self.version = version
        self._secrets = tuple(s for s in secrets if s)

    def cassette(self) -> dict[str, Any]:
        """The recording as written to disk: version 2's `{"version", "entries"}`, version 1's dict."""
        return {"version": CASSETTE_VERSION, "entries": self.replies} if self.version == 2 else self.replies

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Forward to the wrapped model and record the reply under its key."""
        resp = await super().request(messages, model_settings, model_request_parameters)
        text = _text(resp)
        if self.version == 1:
            self.replies[key(messages)] = text
            return resp
        if any(s in text for s in self._secrets):
            raise SecretInCassette("a reply contains a bound secret value; it was not recorded")
        parts = _v2(messages, self.wrapped.model_name, model_settings, model_request_parameters)
        self.replies[_key_of(parts)] = {"text": text, "parts": parts}
        return resp


class ReplayModel(FunctionModel):
    """The model `replay` returns; `unplayed()` names the entries no request asked for."""

    played: set[str]
    entries: Mapping[str, Any]

    def unplayed(self) -> tuple[str, ...]:
        """The recorded keys never replayed, sorted."""
        return tuple(sorted(set(self.entries) - self.played))


def _nearest_diff(entries: Mapping[str, Any], want: Mapping[str, str]) -> list[str]:
    best: list[str] = list(KEY_PARTS)
    for e in entries.values():
        diff = [k for k in KEY_PARTS if e["parts"].get(k) != want[k]]
        best = min(best, diff, key=len)
    return best


def _v2_reply(entries: Mapping[str, Any], played: set[str], messages: list[ModelMessage], info: AgentInfo) -> str:
    want: dict[str, str] = {}
    for model in sorted({e["parts"]["model"] for e in entries.values()}):
        want = _v2(messages, model, info.model_settings, info.model_request_parameters)
        k = _key_of(want)
        if k in entries:
            played.add(k)
            return str(entries[k]["text"])
    differ = ", ".join(_nearest_diff(entries, want)) if want else "everything (empty cassette)"
    raise NotRecorded(f"no recorded reply for this request; it differs from the nearest entry in: {differ}")


def replay(recorded: Mapping[str, Any]) -> ReplayModel:
    """Return a model that answers from a recording: a version-1 dict or a version-2 cassette.

    The returned model raises `NotRecorded` for a request the recording never saw.
    """
    v2 = recorded.get("version") == CASSETTE_VERSION and isinstance(recorded.get("entries"), Mapping)
    entries: Mapping[str, Any] = recorded["entries"] if v2 else recorded
    played: set[str] = set()

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if v2:
            return ModelResponse(parts=[TextPart(_v2_reply(entries, played, messages, info))])
        k = key(messages)
        if k not in entries:
            raise NotRecorded(f"no recorded reply for prompt {k}: re-record the golden set")
        played.add(k)
        return ModelResponse(parts=[TextPart(entries[k])])

    model = ReplayModel(fn, model_name="recorded")
    model.played = played
    model.entries = entries
    return model


class Recording(WrapperModel):
    """A model wrapper that keeps every raw reply, parsed, in `attempts`.

    Pass `attempts` to share one list across several wrapped models (one list per run). An
    unparsable reply is kept as `{"_unparsed": head}`, its first `UNPARSED_HEAD` characters.
    """

    def __init__(self, wrapped: Any, attempts: list[Any] | None = None) -> None:
        super().__init__(wrapped)
        self.attempts: list[Any] = attempts if attempts is not None else []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Forward to the wrapped model and append the reply, parsed, to `attempts`."""
        resp = await super().request(messages, model_settings, model_request_parameters)
        text = _text(resp)
        try:
            self.attempts.append(json.loads(text))
        except ValueError:
            self.attempts.append({"_unparsed": text[:UNPARSED_HEAD]})
        return resp


#: Builds a generator around a model wrapper, as `lambda wrap: PydanticAIGenerator(..., wrap=wrap)`.
type MakeGenerator[G] = Callable[[Callable[[Model], Model]], G]


def recording[G](make: MakeGenerator[G]) -> tuple[G, list[Any]]:
    """Return a generator whose every raw reply lands, parsed, in the returned list (`Recording`)."""
    attempts: list[Any] = []
    return make(lambda m: Recording(m, attempts)), attempts


def recorder[G](make: MakeGenerator[G], *, version: int = 1) -> tuple[G, dict[str, Any]]:
    """Return a generator whose every reply lands in the returned dict under its key (`Recorder`)."""
    replies: dict[str, Any] = {}
    return make(lambda m: Recorder(m, replies, version=version)), replies


__all__ = [
    "CASSETTE_VERSION",
    "KEY_PARTS",
    "UNPARSED_HEAD",
    "MakeGenerator",
    "NotRecorded",
    "Recorder",
    "Recording",
    "ReplayModel",
    "SecretInCassette",
    "key",
    "key_parts",
    "key_v2",
    "recorder",
    "recording",
    "replay",
]
