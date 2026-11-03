"""pydantic_ai_replay -- record a live model's replies once, replay them in CI (pydantic-ai models).

    rec = Recorder(real_model)                 # wraps any pydantic-ai Model; keeps every reply
    ...run the pipeline once against the live model...
    json.dump(rec.replies, f)                  # {key(messages): reply text}
    model = replay(json.load(f))               # a FunctionModel answering from that dict

`key(messages)` is the sha of the FIRST user prompt plus the attempt number (the count of model
responses already in the conversation, so an output retry's second attempt has its own key).
Keying on the prompt TEXT is the point: change a prompt and the lookup misses, the call fails with
`NotRecorded`, and a golden gate goes red until the set is re-recorded -- a reply to a different
prompt proves nothing about this one. The pipeline under test is TODAY's code either way.

`Recording` keeps every raw reply PARSED (`attempts`: an agent returns only the last one, a
grader wants every attempt); an unparsable reply is kept as `{"_unparsed": head}`, never dropped.
Wire either into a `pydantic_ai_generator.PydanticAIGenerator` as its model wrapper.

Framework-bound (pydantic-ai), so import it explicitly, like `pydantic_ai_generator`.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from llm_ports.packs import sha

UNPARSED_HEAD = 2000


class NotRecorded(RuntimeError):
    """The replay has no reply for this prompt: the prompt changed since the recording."""


def key(messages: list[ModelMessage]) -> str:
    """`<sha of the first user prompt>:<attempt>` -- the recording's lookup key."""
    first = next(
        (p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)),
        "",
    )
    attempt = sum(isinstance(m, ModelResponse) for m in messages)
    return f"{sha(str(first))}:{attempt}"


def _text(resp: ModelResponse) -> str:
    return "".join(p.content for p in resp.parts if isinstance(p, TextPart))


class Recorder(WrapperModel):
    """The wrapped model, with every reply's text kept in `replies` under `key(messages)`."""

    def __init__(self, wrapped: Any, replies: dict[str, str] | None = None) -> None:
        super().__init__(wrapped)
        self.replies: dict[str, str] = replies if replies is not None else {}

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        resp = await super().request(messages, model_settings, model_request_parameters)
        self.replies[key(messages)] = _text(resp)
        return resp


def replay(recorded: dict[str, str]) -> FunctionModel:
    """A model answering from a `Recorder`'s dict; a prompt it never saw raises `NotRecorded`."""

    def fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        k = key(messages)
        if k not in recorded:
            raise NotRecorded(f"no recorded reply for prompt {k}: re-record the golden set")
        return ModelResponse(parts=[TextPart(recorded[k])])

    return FunctionModel(fn, model_name="recorded")


class Recording(WrapperModel):
    """The wrapped model, with every raw reply kept parsed in `attempts` (a shared list: one per
    run, many wrapped models). An unparsable reply is kept as `{"_unparsed": head}`."""

    def __init__(self, wrapped: Any, attempts: list[Any] | None = None) -> None:
        super().__init__(wrapped)
        self.attempts: list[Any] = attempts if attempts is not None else []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        resp = await super().request(messages, model_settings, model_request_parameters)
        text = _text(resp)
        try:
            self.attempts.append(json.loads(text))
        except ValueError:
            self.attempts.append({"_unparsed": text[:UNPARSED_HEAD]})
        return resp


__all__ = ["UNPARSED_HEAD", "NotRecorded", "Recorder", "Recording", "key", "replay"]
