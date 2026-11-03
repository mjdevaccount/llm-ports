"""Strict JSON schemas, JSON-object extraction, and the research grammar.

- `strict_json_schema`: the strict-schema shape both OpenAI and llama.cpp accept.
- `extract_json_object`: the JSON-object substring of a model response that is fenced, or
  prefixed or suffixed by prose.
- `research_schema`, `research`: a free-text `analysis` scratchpad first, every property required,
  and the scratchpad popped from the value afterwards.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

from pydantic import BaseModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from llm_ports.ports import Generated, json_output
from llm_ports.roles import Role

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)

ANALYSIS_KEY = "analysis"
RESEARCH_ANALYSIS = {
    "type": "string",
    "description": "your full reasoning -- mechanism, who moves and why. Think HERE first; "
    "then fill every other field from this analysis.",
}


def research_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the research grammar for `schema`.

    Pure. A free-text `analysis` property (`RESEARCH_ANALYSIS`) is injected first, and `required`
    lists every property, including the ones pydantic marks optional because they have a default.
    """
    out = dict(schema)
    props: dict[str, Any] = {ANALYSIS_KEY: RESEARCH_ANALYSIS}
    props.update(out.get("properties") or {})
    out["properties"] = props
    out["required"] = list(props)
    return out


def without_analysis[T](res: Generated[T]) -> Generated[T]:
    """Return `res` with the `analysis` scratchpad popped from a dict value; other values pass unchanged."""
    if isinstance(res.value, dict) and ANALYSIS_KEY in res.value:
        return replace(res, value={k: v for k, v in res.value.items() if k != ANALYSIS_KEY})  # type: ignore[arg-type]
    return res


async def research(
    generate: Callable[..., Awaitable[Generated[Any]]],
    role: Role,
    *,
    prompt: str,
    schema: Mapping[str, Any],
    name: str,
    **kwargs: Any,
) -> Generated[dict[str, Any]]:
    """Run a research call that returns a grammar-enforced dict.

    The consumer's `generate` (its hooks bound) runs with `research_schema(schema)` as the grammar
    (`json_output(..., name)`), and the scratchpad is popped from the value. `kwargs` pass through
    to `generate` unchanged.
    """
    res = await generate(role, prompt=prompt, output=json_output(research_schema(schema), name), **kwargs)
    return without_analysis(res)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a `response_format: json_schema, strict: true` schema that OpenAI and llama.cpp accept.

    OpenAI's strict mode refuses what pydantic's `model_json_schema()` emits: every object needs
    `additionalProperties: false` and every property listed in `required` (nullable via `anyOf`
    null). llama.cpp's grammar compiler accepts the strict form too, so one shape serves both. The
    transform is pydantic-ai's `OpenAIJsonSchemaTransformer`.
    """
    return OpenAIJsonSchemaTransformer(model.model_json_schema(), strict=True).walk()


def extract_json_object(text: str) -> str:
    """Return the JSON-object substring of a model response.

    Tolerates markdown fences and leading or trailing prose: some models fence their JSON even when
    asked not to, and some routes cannot enforce `response_mime_type`. A leading code fence is
    stripped, then the first `{` through the last `}` is taken, so `json.loads` succeeds regardless
    of surrounding text. Falls back to the stripped text when no object is found, so the caller's
    JSON error stays meaningful.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    m = _JSON_OBJ_RE.search(s)
    return m.group(0) if m else s
