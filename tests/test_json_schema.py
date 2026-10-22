"""json_schema: the strict shape both OpenAI and llama.cpp accept, and the JSON-object substring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from llm_ports.json_schema import extract_json_object, strict_json_schema


class _Inner(BaseModel):
    why: str | None = None


class _Out(BaseModel):
    category: str
    note: str | None = None
    inner: _Inner | None = None
    tags: list[str] = []


def _objects(node: Any):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def test_every_object_forbids_extras_and_requires_every_property() -> None:
    objs = list(_objects(strict_json_schema(_Out)))
    assert len(objs) >= 2, "nested $defs are walked too"
    for obj in objs:
        assert obj.get("additionalProperties") is False, obj
        assert sorted(obj.get("required", [])) == sorted(obj.get("properties", {})), obj


def test_strict_output_still_round_trips_a_null() -> None:
    schema = strict_json_schema(_Out)
    row: dict[str, Any] = {k: None for k in schema["required"]}
    row.update(category="x", tags=[])
    assert _Out.model_validate(row).note is None


def test_extract_json_object_strips_fences_and_prose() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_object('Sure! {"a": {"b": 2}} hope that helps') == '{"a": {"b": 2}}'
    assert extract_json_object("no object") == "no object"
