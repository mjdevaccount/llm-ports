"""Thin Protocols for LLM work, and the one call path around them.

The package root exports:

- the ports and their value types (`llm_ports.ports`);
- the call path `generate`, which resolves a `Role` from the consumer's tier table
  (`llm_ports.call.roles`), runs per-call graders whose failures are recorded as scores
  (`llm_ports.call.graders`), and ends in a named `Outcome` (`ok`, `gated`, `unavailable`,
  `request_fault`, `truncated`, `invalid` or `refused`), with the detail in a `Failure` whose
  `kind` names what happened;
- the result contract: `Usage` summed over every request, `Cost` with its source, `Answered`
  (who answered), `Finish`, `CallWarning`, `Attempt`, and the `Pricer` and `GatewayReader` seams;
- resilience as values: `WirePolicy` / `RetryRule` (which failed wire requests are re-sent),
  `Deadline` (the call's total bound), and the `Limiter` seam (`call.resilience.LocalLimiter`);
- the data-rights gate `print_gate` (`llm_ports.provenance`) and `GenerationFailed`.

Prompt packs named by their content sha are `llm_ports.prompts`; the number law and grounded
prose are `llm_ports.grounding`; evals are `llm_ports.evals`.

Frameworks live behind the ports, in `llm_ports.adapters`, each behind its own extra:
`adapters.pydantic_ai.PydanticAIGenerator` implements `Generator`, and
`adapters.pydantic_ai.PydanticAIEmbedder` implements `Embedder`. Import an adapter by its path.

Invariants:
    - The package root imports no framework: it needs only pydantic, jinja2 and pyyaml.
    - A `Generator` never raises for a model, wire or gate failure.

Rationale: ADR-0001, ADR-0019, ADR-0020, ADR-0021.
"""

from importlib.metadata import version

from llm_ports.call import Hooks, generate, json_output, request_for, require_ok, tracer_trace
from llm_ports.call.roles import Role, Tier, TierTable, load_tiers
from llm_ports.ports import (
    Answered,
    Attempt,
    CallWarning,
    Chunk,
    CitedAnswer,
    Claim,
    Cost,
    Deadline,
    Embedded,
    Embedder,
    EmbedInput,
    Failure,
    FailureKind,
    Finish,
    GatewayReader,
    Generated,
    Generation,
    Generator,
    GenRequest,
    Judge,
    JudgeVerdict,
    Limiter,
    Outcome,
    Pricer,
    Prompt,
    PromptSource,
    Reranked,
    Reranker,
    RetryRule,
    Seed,
    Tracer,
    Usage,
    WirePolicy,
    failure,
)
from llm_ports.ports.outcomes import GenerationFailed
from llm_ports.provenance import Activity, ContentItem, DataClass, GateVerdict, print_gate

__version__ = version("llm-ports")

__all__ = [
    "Activity",
    "Answered",
    "Attempt",
    "CallWarning",
    "Chunk",
    "CitedAnswer",
    "Claim",
    "ContentItem",
    "Cost",
    "DataClass",
    "Deadline",
    "EmbedInput",
    "Embedded",
    "Embedder",
    "Failure",
    "FailureKind",
    "Finish",
    "GateVerdict",
    "GatewayReader",
    "GenRequest",
    "Generated",
    "Generation",
    "GenerationFailed",
    "Generator",
    "Hooks",
    "Judge",
    "JudgeVerdict",
    "Limiter",
    "Outcome",
    "Pricer",
    "Prompt",
    "PromptSource",
    "Reranked",
    "Reranker",
    "RetryRule",
    "Role",
    "Seed",
    "Tier",
    "TierTable",
    "Tracer",
    "Usage",
    "WirePolicy",
    "__version__",
    "failure",
    "generate",
    "json_output",
    "load_tiers",
    "print_gate",
    "request_for",
    "require_ok",
    "tracer_trace",
]
