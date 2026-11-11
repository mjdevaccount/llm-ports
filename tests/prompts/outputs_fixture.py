"""Output types a test pack's `output_schema` points at, under the import root `tests.prompts`."""

from __future__ import annotations

from pydantic import BaseModel

from llm_ports.ports import json_output


class Note(BaseModel):
    """A pydantic output type."""

    text: str


NOTE_JSON = json_output({"type": "object", "properties": {"text": {"type": "string"}}}, "note")
NOT_A_TYPE = "a string, not an output type"
