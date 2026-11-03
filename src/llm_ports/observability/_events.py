"""Event hooks: where the call path and the adapters report what they could not raise.

A component that must never raise (the call path, a tracer, a prompt source) still has things to
report: a failed call, a grader that failed, a sink that refused. It hands each one to an
`EventHook`. Without a hook it writes to the stdlib logger the component names (`log_events`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

#: Event hook: `(logging level, event, fields)`.
EventHook = Callable[[int, str, Mapping[str, Any]], None]


def log_events(logger: str) -> EventHook:
    """Return an `EventHook` that writes each event to the stdlib logger named `logger`.

    The record is `"<event> <fields>"` at the event's level.
    """
    log = logging.getLogger(logger)

    def emit(level: int, event: str, fields: Mapping[str, Any]) -> None:
        log.log(level, "%s %s", event, dict(fields))

    return emit


__all__ = ["EventHook", "log_events"]
