"""Thin Protocols for LLM work, and the one call path around them.

The package root exports:

- the ports and their value types (`ports`);
- the call path `call.generate`, which resolves a `Role` from the consumer's tier table (`roles`),
  renders a prompt pack named by its content sha (`packs`), runs per-call graders whose failures
  are recorded as scores (`call_graders`), and ends in a named `Outcome` (`ok`, `gated`,
  `unavailable`, `request_fault`, `truncated` or `invalid`);
- the data-rights gate (`content.print_gate`) and `GenerationFailed`.

Frameworks live behind the ports. `pydantic_ai_generator.PydanticAIGenerator` implements
`Generator` and `pydantic_ai_embedder.PydanticAIEmbedder` implements `Embedder`; import adapters
by module path.

Invariants:
    - The package root imports no framework.
    - A `Generator` never raises for a model, wire or gate failure.

Rationale: ADR-0001.
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
    Reranked,
    Reranker,
    Seed,
    Tracer,
)
from llm_ports.roles import Role, Tier, TierTable, load_tiers

__version__ = "0.12.0"

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
    "Reranked",
    "Reranker",
    "Role",
    "Seed",
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
