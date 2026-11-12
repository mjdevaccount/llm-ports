"""Every OpenTelemetry attribute and metric name the library emits, spelled once.

The GenAI semantic convention is at stability `development`: it has no release, and it renames
(`cache_creation` became `cache_write`, `gen_ai.system` became `gen_ai.provider.name`, the token
usage histogram became per-kind counters). So no call site spells a name: each is a constant here,
checked against the source commit `SEMCONV_SOURCE`. A rename upstream is a one-file diff plus a
changelog line. A scheduled CI job (`.github/workflows/semconv-drift.yml`) diffs these constants
against the convention's registry at HEAD and opens an issue on drift.

Names under `llm_ports.` are the library's own (the convention has no key for them).

Invariants:
    - `CONVENTION_ATTRIBUTES` and `LIBRARY_ATTRIBUTES` are every attribute key an `OtelTracer` span
      may carry; a unit test pins the emitted key set to them.
    - No body attribute (`gen_ai.input.messages`, `gen_ai.output.messages`,
      `gen_ai.system_instructions`) is named here, so none can be emitted.

Rationale: ADR-0025.
"""

from __future__ import annotations

from typing import Final

#: The convention source the names below were checked against: repository and commit.
SEMCONV_SOURCE: Final = "open-telemetry/semantic-conventions-genai@e6658f14c9d9ea0760825f029cb82e7673f48fab"
#: The registry files in that source the drift job reads.
SEMCONV_FILES: Final = ("model/gen-ai/registry.yaml", "model/gen-ai/metrics.yaml", "model/gen-ai/token-metrics.yaml")

# -- convention attributes (model/gen-ai/registry.yaml; error.type is the core convention's) -----
PROVIDER_NAME: Final = "gen_ai.provider.name"
OPERATION_NAME: Final = "gen_ai.operation.name"
REQUEST_MODEL: Final = "gen_ai.request.model"
RESPONSE_MODEL: Final = "gen_ai.response.model"
RESPONSE_ID: Final = "gen_ai.response.id"
FINISH_REASONS: Final = "gen_ai.response.finish_reasons"
CONVERSATION_ID: Final = "gen_ai.conversation.id"
USAGE_INPUT: Final = "gen_ai.usage.input_tokens"
USAGE_OUTPUT: Final = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ: Final = "gen_ai.usage.cache_read.input_tokens"
USAGE_CACHE_WRITE: Final = "gen_ai.usage.cache_write.input_tokens"
USAGE_REASONING: Final = "gen_ai.usage.reasoning.output_tokens"
TOOL_NAME: Final = "gen_ai.tool.name"
TOOL_CALL_ID: Final = "gen_ai.tool.call.id"
TOKEN_MODALITY: Final = "gen_ai.token.modality"
ERROR_TYPE: Final = "error.type"

# -- the library's own attributes ----------------------------------------------------------------
OUTCOME: Final = "llm_ports.outcome"
TIER: Final = "llm_ports.tier"
DEPLOYMENT: Final = "llm_ports.deployment"
COST_USD: Final = "llm_ports.cost_usd"
COST_SOURCE: Final = "llm_ports.cost_source"
USAGE_UNMEASURED: Final = "llm_ports.usage_unmeasured"
ATTEMPTS: Final = "llm_ports.attempts"
PACK: Final = "llm_ports.pack"
ROLE: Final = "llm_ports.role"

# -- metrics (model/gen-ai/metrics.yaml, token-metrics.yaml) -------------------------------------
METRIC_INFERENCE_DURATION: Final = "gen_ai.client.inference.duration"
METRIC_USAGE_INPUT: Final = "gen_ai.client.inference.usage.input_tokens"
METRIC_USAGE_OUTPUT: Final = "gen_ai.client.inference.usage.output_tokens"
METRIC_USAGE_CACHE_READ: Final = "gen_ai.client.inference.usage.cache_read.input_tokens"
METRIC_USAGE_CACHE_WRITE: Final = "gen_ai.client.inference.usage.cache_write.input_tokens"
METRIC_USAGE_REASONING: Final = "gen_ai.client.inference.usage.reasoning.output_tokens"

#: Every convention attribute key the library may emit.
CONVENTION_ATTRIBUTES: Final = frozenset(
    {
        PROVIDER_NAME,
        OPERATION_NAME,
        REQUEST_MODEL,
        RESPONSE_MODEL,
        RESPONSE_ID,
        FINISH_REASONS,
        CONVERSATION_ID,
        USAGE_INPUT,
        USAGE_OUTPUT,
        USAGE_CACHE_READ,
        USAGE_CACHE_WRITE,
        USAGE_REASONING,
        TOOL_NAME,
        TOOL_CALL_ID,
        TOKEN_MODALITY,
        ERROR_TYPE,
    }
)
#: Every library-namespaced attribute key.
LIBRARY_ATTRIBUTES: Final = frozenset(
    {OUTCOME, TIER, DEPLOYMENT, COST_USD, COST_SOURCE, USAGE_UNMEASURED, ATTEMPTS, PACK, ROLE}
)
#: Every metric name the library may record.
METRICS: Final = frozenset(
    {
        METRIC_INFERENCE_DURATION,
        METRIC_USAGE_INPUT,
        METRIC_USAGE_OUTPUT,
        METRIC_USAGE_CACHE_READ,
        METRIC_USAGE_CACHE_WRITE,
        METRIC_USAGE_REASONING,
    }
)
#: Body attributes the library never sets (bodies belong in the consumer's trace store).
BODY_ATTRIBUTES: Final = frozenset({"gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions"})

__all__ = [
    "ATTEMPTS",
    "BODY_ATTRIBUTES",
    "CONVENTION_ATTRIBUTES",
    "CONVERSATION_ID",
    "COST_SOURCE",
    "COST_USD",
    "DEPLOYMENT",
    "ERROR_TYPE",
    "FINISH_REASONS",
    "LIBRARY_ATTRIBUTES",
    "METRICS",
    "METRIC_INFERENCE_DURATION",
    "METRIC_USAGE_CACHE_READ",
    "METRIC_USAGE_CACHE_WRITE",
    "METRIC_USAGE_INPUT",
    "METRIC_USAGE_OUTPUT",
    "METRIC_USAGE_REASONING",
    "OPERATION_NAME",
    "OUTCOME",
    "PACK",
    "PROVIDER_NAME",
    "REQUEST_MODEL",
    "RESPONSE_ID",
    "RESPONSE_MODEL",
    "ROLE",
    "SEMCONV_FILES",
    "SEMCONV_SOURCE",
    "TIER",
    "TOKEN_MODALITY",
    "TOOL_CALL_ID",
    "TOOL_NAME",
    "USAGE_CACHE_READ",
    "USAGE_CACHE_WRITE",
    "USAGE_INPUT",
    "USAGE_OUTPUT",
    "USAGE_REASONING",
    "USAGE_UNMEASURED",
]
