"""The call paths: `generate`, `chat` and `stream`, their hooks, roles, per-call graders, output schemas, retry and skip.

Submodules: `context` (budgets, trimming, the cache plan), `roles`, `models` (what a model can do, who serves it), `providers` (providers by name,
`build`), `graders`, `schema`, and `resilience` (the retry and skip mechanics of a long run).
"""

from llm_ports.call import context, graders, models, providers, resilience, roles, schema
from llm_ports.call._chat import chat, chat_request_for, stream
from llm_ports.call._generate import (
    DEADLINE_GRACE_S,
    OUTPUT_TRACE_MAX,
    AdmitHook,
    Hooks,
    TraceHook,
    generate,
    json_output,
    request_for,
    require_ok,
    tracer_trace,
)

__all__ = [
    "DEADLINE_GRACE_S",
    "OUTPUT_TRACE_MAX",
    "AdmitHook",
    "Hooks",
    "TraceHook",
    "chat",
    "chat_request_for",
    "context",
    "generate",
    "graders",
    "json_output",
    "models",
    "providers",
    "request_for",
    "require_ok",
    "resilience",
    "roles",
    "schema",
    "stream",
    "tracer_trace",
]
