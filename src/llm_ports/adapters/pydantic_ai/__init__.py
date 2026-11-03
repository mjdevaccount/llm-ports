"""Adapters on pydantic-ai: `PydanticAIGenerator`, `PydanticAIEmbedder`, and `replay`."""

from llm_ports.adapters.pydantic_ai import replay
from llm_ports.adapters.pydantic_ai._embedder import (
    EmbedModelFor,
    IndexOrderedTransport,
    PydanticAIEmbedder,
    openai_compatible_embeddings,
)
from llm_ports.adapters.pydantic_ai._generator import (
    REQUEST_FAULT,
    Capture,
    CaptureLike,
    ModelFor,
    PydanticAIGenerator,
    capture,
    classify,
    framework_output,
    openai_compatible,
    record_response,
    structured_dict,
)

__all__ = [
    "REQUEST_FAULT",
    "Capture",
    "CaptureLike",
    "EmbedModelFor",
    "IndexOrderedTransport",
    "ModelFor",
    "PydanticAIEmbedder",
    "PydanticAIGenerator",
    "capture",
    "classify",
    "framework_output",
    "openai_compatible",
    "openai_compatible_embeddings",
    "record_response",
    "replay",
    "structured_dict",
]
