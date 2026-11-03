"""Observation of port calls: `observe`, the null sink, event hooks, and the caller's context variables."""

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
)

__all__ = [
    "LIVE_SOURCE",
    "STATUS_MAX",
    "UNMEASURED_KEY",
    "EventHook",
    "NoopGeneration",
    "NoopTracer",
    "Observation",
    "PortObservation",
    "current_feature",
    "current_source",
    "llm_feature",
    "llm_source",
    "log_events",
    "observe",
    "opt_int",
]
