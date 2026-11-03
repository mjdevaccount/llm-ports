"""Output types on pydantic-ai: hand-written schemas, the port's `JsonOutput`, and strict schemas."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from llm_ports._extras import extra
from llm_ports.ports import JsonOutput

with extra("pydantic-ai"):
    from pydantic_ai import StructuredDict
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer


def structured_dict(schema: Mapping[str, Any], name: str) -> type[dict[str, Any]]:
    """Wrap a hand-written JSON schema as an output type that returns a plain dict.

    The schema is compiled into the grammar unchanged, so `maxLength`, `enum`, `maxItems` and
    `required` all survive. `name` is the schema's title and the name a `FunctionModel` stub
    dispatches on. No shape validation runs beyond what the grammar enforces.
    """
    return StructuredDict(dict(schema), name=name)


def framework_output(output: Any) -> Any:
    """Map the port's `JsonOutput` to pydantic-ai's `StructuredDict`; any other type passes unchanged."""
    if isinstance(output, type) and issubclass(output, JsonOutput):
        return structured_dict(output.json_schema, output.schema_name)
    return output


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a `response_format: json_schema, strict: true` schema that OpenAI and llama.cpp accept.

    OpenAI's strict mode refuses what pydantic's `model_json_schema()` emits: every object needs
    `additionalProperties: false` and every property listed in `required` (nullable via `anyOf`
    null). llama.cpp's grammar compiler accepts the strict form too, so one shape serves both. The
    transform is pydantic-ai's `OpenAIJsonSchemaTransformer`.
    """
    return OpenAIJsonSchemaTransformer(model.model_json_schema(), strict=True).walk()


__all__ = ["framework_output", "strict_json_schema", "structured_dict"]
