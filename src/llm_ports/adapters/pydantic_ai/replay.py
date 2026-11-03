"""Record a live model's replies once and replay them in CI, for pydantic-ai models.

Example:
    rec = Recorder(real_model)    # wraps any pydantic-ai Model; keeps every reply
    ...                           # run the pipeline once against the live model
    json.dump(rec.replies, f)     # {key(messages): reply text}
    model = replay(json.load(f))  # a FunctionModel answering from that dict

`key(messages)` is the sha of the first user prompt plus the attempt number (the count of model
responses already in the conversation, so an output retry's second attempt has its own key).
Because the key covers the prompt text, a changed prompt misses the lookup, the call raises
`NotRecorded`, and the golden set must be re-recorded. The code under test always runs live; only
the model's replies are replayed.

`Recording` keeps every raw reply parsed in `attempts` (an agent returns only the last attempt; a
grader wants every one); an unparsable reply is kept as `{"_unparsed": head}`, never dropped.
Either wrapper plugs into `pydantic_ai_generator.PydanticAIGenerator` as its `wrap=` model wrapper.

The module depends on pydantic-ai, so it is imported explicitly, like `pydantic_ai_generator`.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from llm_ports.prompts import sha

#: Characters of an unparsable reply that `Recording` keeps.
UNPARSED_HEAD = 2000


class NotRecorded(RuntimeError):
    """The replay has no reply for this prompt: the prompt changed since the recording."""


def key(messages: list[ModelMessage]) -> str:
    """Return the recording's lookup key, `<sha of the first user prompt>:<attempt>`."""
    first = next(
        (p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)),
        "",
    )
    attempt = sum(isinstance(m, ModelResponse) for m in messages)
    return f"{sha(str(first))}:{attempt}"


def _text(resp: ModelResponse) -> str:
    return "".join(p.content for p in resp.parts if isinstance(p, TextPart))


class Recorder(WrapperModel):
    """A model wrapper that keeps every reply's text in `replies` under `key(messages)`.

    Pass `replies` to share one dict across several wrapped models.
    """

    def __init__(self, wrapped: Any, replies: dict[str, str] | None = None) -> None:
        super().__init__(wrapped)
        self.replies: dict[str, str] = replies if replies is not None else {}

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Forward to the wrapped model and record the reply's text under `key(messages)`."""
        resp = await super().request(messages, model_settings, model_request_parameters)
        self.replies[key(messages)] = _text(resp)
        return resp


def replay(recorded: dict[str, str]) -> FunctionModel:
    """Return a model that answers from a `Recorder`'s dict.

    The returned model raises `NotRecorded` for a prompt the recording never saw.
    """

    def fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        k = key(messages)
        if k not in recorded:
            raise NotRecorded(f"no recorded reply for prompt {k}: re-record the golden set")
        return ModelResponse(parts=[TextPart(recorded[k])])

    return FunctionModel(fn, model_name="recorded")


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


__all__ = ["UNPARSED_HEAD", "NotRecorded", "Recorder", "Recording", "key", "replay"]
