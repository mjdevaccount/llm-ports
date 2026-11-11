"""The one call path: `generate`, its hooks, roles, per-call graders, output schemas, retry and skip.

Submodules: `roles`, `models` (what a model can do, who serves it), `providers` (providers by name,
`build`), `graders`, `schema`, and `resilience` (the retry and skip mechanics of a long run).
"""

from llm_ports.call import graders, models, providers, resilience, roles, schema
from llm_ports.call._generate import (
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
    "OUTPUT_TRACE_MAX",
    "AdmitHook",
    "Hooks",
    "TraceHook",
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
    "tracer_trace",
]
