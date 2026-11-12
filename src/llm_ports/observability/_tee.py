"""`TeeTracer`: one `Tracer` that fans every generation out to several sinks, each isolated.

A consumer with two sinks (a trace store and OpenTelemetry) composes them once,
`TeeTracer(LangfuseTracer(client), OtelTracer(provider))`, instead of dual-emitting by hand at each
call site.

Invariants:
    - One sink failing never stops another: an exception from a sink's `start_generation`, `end` or
      `score` is caught and counted, and emits `tracer_sink_failed` (at most once per sink per
      `hold_s`). A sink that failed to open gets a `NoopGeneration` for that call.
    - Never raises.
    - `stats()` counts per sink what was started, ended and failed; at rest, `started - ended` is
      what a sink leaked.

Rationale: ADR-0025.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from llm_ports.observability._events import EventHook, log_events
from llm_ports.observability._noop import NoopGeneration
from llm_ports.observability._observation import record_tool
from llm_ports.observability._payload import deliver_or_raise
from llm_ports.ports import Generation, Tracer

#: The event a failing sink emits.
SINK_FAILED = "tracer_sink_failed"
_ERR_CAP = 200


class _Sink:
    __slots__ = ("ended", "failed", "name", "quiet_until", "started", "tracer")

    def __init__(self, name: str, tracer: Tracer) -> None:
        self.name = name
        self.tracer = tracer
        self.started = 0
        self.ended = 0
        self.failed = 0
        self.quiet_until = 0.0


class TeeTracer:
    """A `Tracer` over several sinks; the module docstring states the contract.

    Args:
        tracers: The sinks, in the order each call reaches them.
        on_event: Receives `tracer_sink_failed`; without one, the stdlib logger
            `llm_ports.observability`.
        hold_s: The fewest seconds between two `tracer_sink_failed` events from one sink.
        clock: A monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        *tracers: Tracer,
        on_event: EventHook | None = None,
        hold_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sinks = [_Sink(f"{i}:{type(t).__name__}", t) for i, t in enumerate(tracers)]
        self._emit: EventHook = on_event or log_events("llm_ports.observability")
        self._hold = hold_s
        self._clock = clock

    def stats(self) -> Mapping[str, Mapping[str, int]]:
        """Per sink (`"<index>:<class>"`): generations `started`, `ended`, and calls that `failed`."""
        return {s.name: {"started": s.started, "ended": s.ended, "failed": s.failed} for s in self._sinks}

    def _failed(self, sink: _Sink, action: str, exc: Exception) -> None:
        sink.failed += 1
        now = self._clock()
        if now < sink.quiet_until:
            return
        sink.quiet_until = now + self._hold
        self._emit(
            logging.WARNING,
            SINK_FAILED,
            {
                "sink": sink.name,
                "action": action,
                "error": f"{type(exc).__name__}: {exc}"[:_ERR_CAP],
                "failed": sink.failed,
            },
        )

    def start_generation(
        self,
        *,
        name: str,
        model: str,
        provider: str,
        input: Any,
        metadata: Mapping[str, Any],
        session_id: str | None,
        prompt_name: str | None,
    ) -> Generation:
        """Open the generation on every sink; a sink that raises gets a `NoopGeneration`."""
        opened: list[tuple[_Sink, Generation]] = []
        for sink in self._sinks:
            try:
                gen = sink.tracer.start_generation(
                    name=name,
                    model=model,
                    provider=provider,
                    input=input,
                    metadata=metadata,
                    session_id=session_id,
                    prompt_name=prompt_name,
                )
            except Exception as exc:
                self._failed(sink, "start_generation", exc)
                gen = NoopGeneration()
            else:
                sink.started += 1
            opened.append((sink, gen))
        return TeeGeneration(self, opened)


class TeeGeneration:
    """The `Generation` a `TeeTracer` opens: every `end` and `score` reaches every sink's generation."""

    def __init__(self, tee: TeeTracer, opened: list[tuple[_Sink, Generation]]) -> None:
        self._tee = tee
        self._opened = opened

    def end(self, **kw: Any) -> None:
        """End every sink's generation with the same fields; never raises."""
        for sink, gen in self._opened:
            if isinstance(gen, NoopGeneration):
                continue
            try:
                deliver_or_raise(gen, kw)
            except Exception as exc:
                self._tee._failed(sink, "end", exc)
            else:
                sink.ended += 1

    def tool(self, *, name: str, call_id: str, duration_s: float, is_error: bool) -> None:
        """Hand one tool execution to every sink's generation that records tools; never raises."""
        for _sink, gen in self._opened:
            record_tool(gen, name=name, call_id=call_id, duration_s=duration_s, is_error=is_error)

    def score(
        self, name: str, value: float, comment: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Score every sink's generation; never raises."""
        for sink, gen in self._opened:
            try:
                if metadata is None:
                    gen.score(name, value, comment)
                else:
                    gen.score(name, value, comment, metadata)
            except Exception as exc:
                self._tee._failed(sink, "score", exc)


__all__ = ["SINK_FAILED", "TeeGeneration", "TeeTracer"]
