"""llm-ports -- our seam, their impl.

Thin Protocols for LLM work (`ports`), and the one call path around them (`call.generate`): a
`Role` resolved from the consumer's tier table (`roles`), a prompt pack with a frozen content sha
(`packs`), per-call graders whose failures are recorded scores (`call_graders`), and an outcome
that is always NAMED (`ok | gated | unavailable | request_fault | truncated | invalid`) -- the
Generator never raises for a model, wire or gate failure.

Frameworks live behind the ports. `pydantic_ai_generator.PydanticAIGenerator` is the first
`Generator` adapter (import it explicitly: this package root imports no framework).
"""

from llm_ports.call import Hooks, generate, json_output, request_for, require_ok, tracer_trace
from llm_ports.failure import GenerationFailed
from llm_ports.ports import (
    Chunk,
    CitedAnswer,
    Claim,
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

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "CitedAnswer",
    "Claim",
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
    "request_for",
    "require_ok",
    "tracer_trace",
]
