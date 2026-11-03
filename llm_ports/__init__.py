"""llm-ports -- our seam, their impl.

Thin Protocols for LLM work (`ports`), and the one call path around them (`call.generate`): a
`Role` resolved from the consumer's tier table (`roles`), a prompt pack with a frozen content sha
(`packs`), per-call graders whose failures are recorded scores (`call_graders`), and an outcome
that is always NAMED (`ok | gated | unavailable | request_fault | truncated | invalid`) -- the
Generator never raises for a model, wire or gate failure.

Frameworks live behind the ports. `pydantic_ai_generator.PydanticAIGenerator` is the first
`Generator` adapter and `pydantic_ai_embedder.PydanticAIEmbedder` the first `Embedder` adapter
(import them explicitly: this package root imports no framework).
"""

from llm_ports.call import Hooks, generate, json_output, request_for, require_ok, tracer_trace
from llm_ports.content import Activity, ContentItem, DataClass, GateVerdict, print_gate
from llm_ports.failure import GenerationFailed
from llm_ports.ports import (
    Chunk,
    CitedAnswer,
    Claim,
    Embedded,
    Embedder,
    EmbedInput,
    Generated,
    Generation,
    Generator,
    GenRequest,
    Judge,
    JudgeVerdict,
    Outcome,
    Prompt,
    PromptSource,
    Tracer,
)
from llm_ports.roles import Role, Tier, TierTable, load_tiers

__version__ = "0.8.1"

__all__ = [
    "Activity",
    "Chunk",
    "CitedAnswer",
    "Claim",
    "ContentItem",
    "DataClass",
    "GateVerdict",
    "EmbedInput",
    "Embedded",
    "Embedder",
    "GenRequest",
    "Generated",
    "Generation",
    "GenerationFailed",
    "Generator",
    "Hooks",
    "Judge",
    "JudgeVerdict",
    "Outcome",
    "Prompt",
    "PromptSource",
    "Role",
    "Tier",
    "TierTable",
    "Tracer",
    "__version__",
    "generate",
    "json_output",
    "load_tiers",
    "print_gate",
    "request_for",
    "require_ok",
    "tracer_trace",
]
