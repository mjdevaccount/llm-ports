"""json_schema: the strict shape both OpenAI and llama.cpp accept, and the JSON-object substring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from llm_ports.call.schema import (
    RESEARCH_ANALYSIS,
    extract_json_object,
    research,
    research_schema,
    strict_json_schema,
    without_analysis,
)
from llm_ports.ports import Generated


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


class TestResearch:
    """MUST-FIRE: the scratchpad leads the grammar, every property is required, and the value the
    caller reads carries no scratchpad."""

    def test_analysis_leads_and_every_property_is_required(self) -> None:
        got = research_schema(
            {
                "type": "object",
                "properties": {"note": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["note"],
            }
        )
        assert list(got["properties"]) == ["analysis", "note", "n"]
        assert got["required"] == ["analysis", "note", "n"]
        assert got["properties"]["analysis"] is RESEARCH_ANALYSIS and got["type"] == "object"

    def test_the_input_schema_is_not_mutated(self) -> None:
        src: dict[str, Any] = {"properties": {"a": {"type": "string"}}, "required": []}
        research_schema(src)
        assert src == {"properties": {"a": {"type": "string"}}, "required": []}

    async def test_the_pass_pops_the_scratchpad_and_forwards_kwargs(self) -> None:
        calls: list[dict[str, Any]] = []

        async def generate(role: Any, **kw: Any) -> Generated[Any]:
            calls.append(kw)
            return Generated(value={"analysis": "thinking", "note": "x"}, outcome="ok")

        res = await research(
            generate, "role", prompt="p", schema={"properties": {"note": {}}}, name="note", metadata={"m": 1}
        )
        assert res.value == {"note": "x"} and res.outcome == "ok"
        assert calls[0]["metadata"] == {"m": 1} and calls[0]["output"].__name__ == "note"
        assert list(calls[0]["output"].json_schema["properties"]) == ["analysis", "note"]

    def test_a_failed_or_prose_value_is_untouched(self) -> None:
        dead: Generated[Any] = Generated(value=None, outcome="unavailable", reason="down")
        assert without_analysis(dead) is dead
        prose: Generated[Any] = Generated(value="analysis", outcome="ok")
        assert without_analysis(prose) is prose


def test_extract_json_object_strips_fences_and_prose() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_object('Sure! {"a": {"b": 2}} hope that helps') == '{"a": {"b": 2}}'
    assert extract_json_object("no object") == "no object"
