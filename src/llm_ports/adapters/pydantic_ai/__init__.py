"""Adapters on pydantic-ai: generator, embedder, output schemas, replay, the `openai_compatible` kind.

`PydanticAIGenerator`, `PydanticAIEmbedder`, `replay`, and `OpenAICompatibleFactory`, the
`openai_compatible` provider kind.

Needs the `pydantic-ai` extra (``pip install 'llm-ports[pydantic-ai]'``).
"""

from llm_ports._extras import extra

with extra("pydantic-ai"):
    from llm_ports.adapters._index_order import IndexOrderedTransport
from llm_ports.adapters.pydantic_ai import replay
from llm_ports.adapters.pydantic_ai._embedder import (
    EmbedModelFor,
    PydanticAIEmbedder,
    openai_compatible_embeddings,
)
from llm_ports.adapters.pydantic_ai._factory import OpenAICompatibleFactory, translate_settings
from llm_ports.adapters.pydantic_ai._generator import (
    CaptureLike,
    ModelFor,
    PydanticAIGenerator,
    classify,
    openai_compatible,
)
from llm_ports.adapters.pydantic_ai._schema import framework_output, strict_json_schema, structured_dict

__all__ = [
    "CaptureLike",
    "EmbedModelFor",
    "IndexOrderedTransport",
    "ModelFor",
    "OpenAICompatibleFactory",
    "PydanticAIEmbedder",
    "PydanticAIGenerator",
    "classify",
    "framework_output",
    "openai_compatible",
    "openai_compatible_embeddings",
    "replay",
    "strict_json_schema",
    "structured_dict",
    "translate_settings",
]
