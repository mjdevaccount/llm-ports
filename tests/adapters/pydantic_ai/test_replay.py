"""pydantic_ai_replay: record once, replay by prompt key, refuse a prompt never recorded. Moved from
trade-data's evals/recorded_model.py (v0.9.0); the key spelling is FROZEN (recorded golden sets
on disk are keyed by it)."""

from __future__ import annotations

import json

from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from llm_ports.adapters.pydantic_ai import PydanticAIGenerator
from llm_ports.adapters.pydantic_ai.replay import NotRecorded, Recorder, Recording, key, recorder, recording, replay
from llm_ports.ports import GenRequest
from llm_ports.prompts import sha
from tests.adapters.pydantic_ai._scripted import scripted_model as _live


class _Note(BaseModel):
    headline: str


REQ = GenRequest(model="small-chat", prompt="narrate", output_retries=1)


def _not_empty(n: _Note) -> list[str]:
    return [] if n.headline else ["empty headline"]


def test_the_key_is_the_first_user_prompt_sha_and_the_attempt() -> None:
    first = [ModelRequest(parts=[UserPromptPart("narrate")])]
    assert key(first) == f"{sha('narrate')}:0"
    retried = [*first, ModelResponse(parts=[TextPart("x")]), ModelRequest(parts=[UserPromptPart("fix it")])]
    assert key(retried) == f"{sha('narrate')}:1", "a re-ask keys by the FIRST prompt, next attempt"


async def test_record_then_replay_round_trips_both_attempts() -> None:
    rec = Recorder(_live(json.dumps({"headline": ""}), json.dumps({"headline": "recorded"})))
    live = await PydanticAIGenerator(rec).generate(
        REQ, _Note, validate=_not_empty, retry_message=lambda p: "; ".join(p)
    )
    assert live.outcome == "ok" and live.retries == 1
    assert sorted(k.rsplit(":", 1)[1] for k in rec.replies) == ["0", "1"]
    again = await PydanticAIGenerator(replay(rec.replies)).generate(
        REQ, _Note, validate=_not_empty, retry_message=lambda p: "; ".join(p)
    )
    assert again.outcome == "ok" and again.value == live.value and again.retries == 1


async def test_a_changed_prompt_is_not_recorded() -> None:
    """MUST FIRE: a replay keyed on anything but the prompt text would answer a prompt it never saw."""
    rec = Recorder(_live(json.dumps({"headline": "h"})))
    await PydanticAIGenerator(rec).generate(REQ, _Note)
    same = await PydanticAIGenerator(replay(rec.replies)).generate(REQ, _Note)
    assert same.outcome == "ok" and same.value == _Note(headline="h")
    changed = await PydanticAIGenerator(replay(rec.replies)).generate(
        GenRequest(model="small-chat", prompt="a different prompt"), _Note
    )
    assert changed.outcome != "ok" and changed.value is None
    assert NotRecorded.__name__ in (changed.reason or ""), changed.reason


async def test_recording_keeps_every_attempt_parsed_including_the_unparsable() -> None:
    attempts: list[object] = []
    gen = PydanticAIGenerator(Recording(_live("not json at all", json.dumps({"headline": "fits"})), attempts))
    res = await gen.generate(REQ, _Note)
    assert res.outcome == "ok" and res.value == _Note(headline="fits")
    assert attempts == [{"_unparsed": "not json at all"}, {"headline": "fits"}]


async def test_recording_builds_the_generator_with_the_wrapper_on_and_returns_its_list() -> None:
    live = _live("not json at all", json.dumps({"headline": "fits"}))
    gen, attempts = recording(lambda wrap: PydanticAIGenerator(wrap(live)))
    res = await gen.generate(REQ, _Note)
    assert res.outcome == "ok" and attempts == [{"_unparsed": "not json at all"}, {"headline": "fits"}]


async def test_recorder_builds_the_generator_and_returns_the_dict_replay_reads() -> None:
    gen, replies = recorder(lambda wrap: PydanticAIGenerator(wrap(_live(json.dumps({"headline": "kept"})))))
    await gen.generate(REQ, _Note)
    again = await PydanticAIGenerator(replay(replies)).generate(REQ, _Note)
    assert again.outcome == "ok" and again.value == _Note(headline="kept")
