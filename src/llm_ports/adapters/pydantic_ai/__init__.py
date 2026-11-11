"""Adapters on pydantic-ai: generator, embedder, output schemas, replay, the `openai_compatible` kind.

`PydanticAIGenerator`, `PydanticAIEmbedder`, `replay`, and `OpenAICompatibleFactory`, the
`openai_compatible` provider kind.

The failure-classification table every adapter shares (`classify_core`, `WireFacts`) and the
pydantic-ai categories over it (`classify_failure`) are exported here too.

Needs the `pydantic-ai` extra (``pip install 'llm-ports[pydantic-ai]'``).
"""

from llm_ports._extras import extra

with extra("pydantic-ai"):
    from llm_ports.adapters._index_order import IndexOrderedTransport
from llm_ports.adapters._classify import (
    ERROR_MESSAGE_MAX,
    OVERFLOW_CODES,
    OVERFLOW_MESSAGES,
    STATUS_KINDS,
    Categorize,
    Classifier,
    WireFacts,
    classify_core,
    is_context_overflow,
    kind_for_status,
    retry_after_s,
)
from llm_ports.adapters.pydantic_ai import replay
from llm_ports.adapters.pydantic_ai._capture import CaptureLike, WireReply, parse_usage
from llm_ports.adapters.pydantic_ai._classify import OPENAI_FINISH, classify, classify_failure, openai_finish
from llm_ports.adapters.pydantic_ai._embedder import (
    EmbedModelFor,
    PydanticAIEmbedder,
    openai_compatible_embeddings,
)
from llm_ports.adapters.pydantic_ai._factory import OpenAICompatibleFactory, translate_settings
from llm_ports.adapters.pydantic_ai._generator import (
    TRACEBACK_MAX,
    ModelFor,
    PydanticAIGenerator,
    openai_compatible,
)
from llm_ports.adapters.pydantic_ai._schema import framework_output, strict_json_schema, structured_dict

__all__ = [
    "ERROR_MESSAGE_MAX",
    "OPENAI_FINISH",
    "OVERFLOW_CODES",
    "OVERFLOW_MESSAGES",
    "STATUS_KINDS",
    "TRACEBACK_MAX",
    "CaptureLike",
    "Categorize",
    "Classifier",
    "EmbedModelFor",
    "IndexOrderedTransport",
    "ModelFor",
    "OpenAICompatibleFactory",
    "PydanticAIEmbedder",
    "PydanticAIGenerator",
    "WireFacts",
    "WireReply",
    "classify",
    "classify_core",
    "classify_failure",
    "framework_output",
    "is_context_overflow",
    "kind_for_status",
    "openai_compatible",
    "openai_compatible_embeddings",
    "openai_finish",
    "parse_usage",
    "replay",
    "retry_after_s",
    "strict_json_schema",
    "structured_dict",
    "translate_settings",
]
