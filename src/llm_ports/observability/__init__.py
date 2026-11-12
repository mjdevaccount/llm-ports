"""Observation of port calls.

`observe`, `record`, the fan-out `TeeTracer`, the null sink, event hooks, the caller's context
variables, and the pinned OpenTelemetry names (`observability.semconv`).
"""

from llm_ports.observability._context import LIVE_SOURCE, current_feature, current_source, llm_feature, llm_source
from llm_ports.observability._events import EventHook, log_events
from llm_ports.observability._noop import NoopGeneration, NoopTracer
from llm_ports.observability._observation import (
    STATUS_MAX,
    UNMEASURED_KEY,
    Observation,
    PortObservation,
    observe,
    opt_int,
    record,
)
from llm_ports.observability._tee import SINK_FAILED, TeeGeneration, TeeTracer

__all__ = [
    "LIVE_SOURCE",
    "SINK_FAILED",
    "STATUS_MAX",
    "UNMEASURED_KEY",
    "EventHook",
    "NoopGeneration",
    "NoopTracer",
    "Observation",
    "PortObservation",
    "TeeGeneration",
    "TeeTracer",
    "current_feature",
    "current_source",
    "llm_feature",
    "llm_source",
    "log_events",
    "observe",
    "opt_int",
    "record",
]
