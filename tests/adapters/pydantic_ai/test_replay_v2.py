"""Replay key v2: covers model, system, messages, settings and schema; a v2 recording replays; a miss
names the differing parts; a partly played cassette is visible; a secret is never written."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, ToolCallPart, UserPromptPart

from llm_ports.adapters.pydantic_ai import PydanticAIGenerator
from llm_ports.adapters.pydantic_ai.replay import (
    CASSETTE_VERSION,
    NotRecorded,
    Recorder,
    SecretInCassette,
    key_v2,
    recorder,
    replay,
)
from llm_ports.ports import GenRequest
from tests.adapters.pydantic_ai._scripted import scripted_model as _live

MSGS = [ModelRequest(parts=[UserPromptPart("narrate")])]
BASE = {"model": "m", "system": "be brief", "settings": {"temperature": 0.1}, "output_schema": {"type": "object"}}


class _Note(BaseModel):
    headline: str


def test_key_v2_moves_with_every_part_that_changes_the_answer() -> None:
    k = key_v2(MSGS, attempt=0, **BASE)  # type: ignore[arg-type]
    assert k.endswith(":0")
    for part, other in [
        ("model", "m2"),
        ("system", "be verbose"),
        ("settings", {"temperature": 0.2}),
        ("output_schema", {"type": "string"}),
    ]:
        assert key_v2(MSGS, attempt=0, **{**BASE, part: other}) != k, part  # type: ignore[arg-type]
    assert key_v2(MSGS, attempt=1, **BASE) != k  # type: ignore[arg-type]
    later = [*MSGS, ModelResponse(parts=[ToolCallPart("f", {"a": 1})]), ModelRequest(parts=[UserPromptPart("more")])]
    assert key_v2(later, attempt=0, **BASE) != k  # type: ignore[arg-type]


def test_settings_order_does_not_change_the_key() -> None:
    a = key_v2(MSGS, attempt=0, **{**BASE, "settings": {"a": 1, "b": 2}})  # type: ignore[arg-type]
    b = key_v2(MSGS, attempt=0, **{**BASE, "settings": {"b": 2, "a": 1}})  # type: ignore[arg-type]
    assert a == b


async def test_a_v2_recording_writes_the_new_shape_and_replays() -> None:
    rec = Recorder(_live(json.dumps({"headline": "h"})), version=2)
    live = await PydanticAIGenerator(rec).generate(GenRequest(model="m", prompt="narrate"), _Note)
    cassette = json.loads(json.dumps(rec.cassette()))
    assert cassette["version"] == CASSETTE_VERSION and len(cassette["entries"]) == 1
    model = replay(cassette)
    again = await PydanticAIGenerator(model).generate(GenRequest(model="m", prompt="narrate"), _Note)
    assert again.value == live.value and model.unplayed() == ()


async def test_a_system_prompt_change_misses_and_names_the_part() -> None:
    # MUST FIRE: the v1 key (first user prompt only) replays the old answer for a new system prompt.
    rec = Recorder(_live("ok"), version=2)
    await PydanticAIGenerator(rec).generate(GenRequest(model="m", prompt="narrate", system="be brief"), str)
    changed = await PydanticAIGenerator(replay(rec.cassette())).generate(
        GenRequest(model="m", prompt="narrate", system="be verbose"), str
    )
    assert (
        changed.outcome != "ok"
        and "system" in (changed.reason or "")
        and NotRecorded.__name__ in (changed.reason or "")
    )


async def test_an_unplayed_entry_is_reported() -> None:
    gen, replies = recorder(lambda wrap: PydanticAIGenerator(wrap(_live("a"))), version=2)
    await gen.generate(GenRequest(model="m", prompt="one"), str)
    await gen.generate(GenRequest(model="m", prompt="two"), str)
    model = replay({"version": 2, "entries": replies})
    await PydanticAIGenerator(model).generate(GenRequest(model="m", prompt="one"), str)
    assert len(model.unplayed()) == 1


async def test_a_reply_containing_a_bound_secret_is_refused_and_not_written() -> None:
    rec = Recorder(_live("the key is sk-live-123"), version=2, secrets=("sk-live-123",))
    res = await PydanticAIGenerator(rec).generate(GenRequest(model="m", prompt="leak"), str)
    assert res.outcome != "ok" and rec.replies == {}
    assert SecretInCassette.__name__ in (res.reason or "") or res.failure is not None


def test_only_versions_one_and_two_exist() -> None:
    with pytest.raises(ValueError, match="version"):
        Recorder(_live("x"), version=3)


def test_a_v1_recorder_still_writes_the_v1_dict() -> None:
    assert Recorder(_live("x")).cassette() == {}


def test_a_system_part_in_the_messages_is_in_the_key() -> None:
    with_sys = [ModelRequest(parts=[SystemPromptPart("s"), UserPromptPart("u")])]
    plain = [ModelRequest(parts=[UserPromptPart("u")])]
    keys = {key_v2(m, model="m", system="", settings=None, output_schema=None, attempt=0) for m in (with_sys, plain)}
    assert len(keys) == 2


async def test_a_text_reply_in_the_history_is_in_the_key() -> None:
    a = [*MSGS, ModelResponse(parts=[TextPart("x")])]
    b = [*MSGS, ModelResponse(parts=[TextPart("y")])]
    assert key_v2(a, attempt=1, **BASE) != key_v2(b, attempt=1, **BASE)  # type: ignore[arg-type]
