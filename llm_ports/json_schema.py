"""json_schema -- the strict-schema shape both OpenAI and llama.cpp accept, and the JSON-object
substring of an LLM response (fenced, prefixed or suffixed by prose)."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A `response_format: json_schema, strict: true` schema BOTH decoders accept.

    OpenAI's strict mode refuses what pydantic's `model_json_schema()` emits: every object needs
    `additionalProperties: false` and every property listed in `required` (nullable via
    `anyOf null`) -- a gateway that falls back from a self-hosted model to OpenAI on an outage
    otherwise cannot take a single request. llama.cpp's grammar compiler accepts the strict form
    too, so one shape serves both. pydantic-ai's OpenAI transformer is the integrated tool."""
    return OpenAIJsonSchemaTransformer(model.model_json_schema(), strict=True).walk()


def extract_json_object(text: str) -> str:
    """Return the JSON-object substring of an LLM response, tolerant of
    markdown fences and leading/trailing prose.

    gemini-2.5-pro fences ~100% of its outputs (and the OAuth CLI route
    can't enforce response_mime_type), and models occasionally add a
    sentence before/after the JSON. Strip a leading code-fence, then take
    the first ``{`` .. last ``}`` so ``json.loads`` succeeds regardless of
    surrounding noise. Falls back to the stripped text when no object is
    found (so the caller still raises a meaningful JSON error).
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    m = _JSON_OBJ_RE.search(s)
    return m.group(0) if m else s
