"""The one call path: `generate`, its hooks, roles (`roles`), per-call graders (`graders`) and output schemas (`schema`)."""

from llm_ports.call import graders, roles, schema
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
    "request_for",
    "require_ok",
    "roles",
    "schema",
    "tracer_trace",
]
