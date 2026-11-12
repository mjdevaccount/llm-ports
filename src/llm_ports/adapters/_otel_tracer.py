"""`OtelTracer`: the `Tracer` port over OpenTelemetry (extra `[otel]`).

`TeeTracer(LangfuseTracer(client), OtelTracer())` sends every model call to the trace store AND
to the OpenTelemetry pipeline the rest of the process already exports, through one port.

`start_generation` opens a CLIENT span named `"{operation} {model}"` (the operation is the
opening metadata's `"operation"`, default `"chat"`) as a child of the ambient context; `end` sets
the result's attributes and ends it. Every attribute and metric name comes from
`observability.semconv`, pinned to one commit of the GenAI conventions.

Given a `meter_provider`, `end` also records the convention's `gen_ai.client.inference.duration`
histogram (seconds) and one counter per measured token kind. Nothing else is a metric.

Invariants:
    - No body is ever carried: the prompt, the output and system instructions never become span
      attributes, and no switch (or environment variable) turns them on. Bodies belong in the
      access-controlled trace store, through its own `Tracer`.
    - Unmeasured is absent, never zero: a usage side nobody reported has no attribute and no
      counter increment; with no side reported, `llm_ports.usage_unmeasured` is True.
    - A failure sets `error.type` to its `kind` and the span status to ERROR.
    - A tool execution of a chat call is an INTERNAL child span `execute_tool {name}` with
      `gen_ai.tool.name` and `gen_ai.tool.call.id`, timed by the tool's own duration.
    - Never raises; each generation ends its span once.

Rationale: ADR-0007, ADR-0025.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

from llm_ports._extras import extra
from llm_ports.observability import NoopGeneration, semconv
from llm_ports.ports import Answered, Cost, Failure, Finish, Generation, Usage

with extra("otel"):
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

#: The operation a generation is named by when its metadata names none.
DEFAULT_OPERATION = "chat"
#: Characters of a status description kept on a span.
_STATUS_MAX = 200
_SCOPE = "llm_ports"
# The usage fields a span and the token counters carry, in a fixed order.
_USAGE: tuple[tuple[str, str, str], ...] = (
    ("input", semconv.USAGE_INPUT, semconv.METRIC_USAGE_INPUT),
    ("output", semconv.USAGE_OUTPUT, semconv.METRIC_USAGE_OUTPUT),
    ("cache_read", semconv.USAGE_CACHE_READ, semconv.METRIC_USAGE_CACHE_READ),
    ("cache_write", semconv.USAGE_CACHE_WRITE, semconv.METRIC_USAGE_CACHE_WRITE),
    ("reasoning", semconv.USAGE_REASONING, semconv.METRIC_USAGE_REASONING),
)
# Opening-metadata keys copied to library attributes.
_OPEN_META = (("role", semconv.ROLE), ("pack", semconv.PACK))


class _Meters:
    """The duration histogram and one counter per token kind, from one meter."""

    def __init__(self, meter_provider: Any) -> None:
        meter = meter_provider.get_meter(_SCOPE)
        self.duration: Any = meter.create_histogram(
            semconv.METRIC_INFERENCE_DURATION, unit="s", description="GenAI operation duration."
        )
        self.tokens: dict[str, Any] = {
            field: meter.create_counter(metric, unit="{token}", description=f"GenAI {field} tokens.")
            for field, _, metric in _USAGE
        }


class OtelTracer:
    """A `Tracer` that writes each generation as an OpenTelemetry span; the module docstring states the contract.

    Args:
        tracer_provider: The provider spans come from; None takes the global one.
        meter_provider: When given, the duration histogram and token counters are recorded on it.
        semconv: The attribute-name set; only `"pinned"` (`observability.semconv`) exists.
        clock: A monotonic clock for the duration metric, injectable for tests.
    """

    def __init__(
        self,
        tracer_provider: Any | None = None,
        meter_provider: Any | None = None,
        *,
        semconv: Literal["pinned"] = "pinned",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if semconv != "pinned":
            raise ValueError(f"semconv must be 'pinned', not {semconv!r}")
        provider = tracer_provider if tracer_provider is not None else trace.get_tracer_provider()
        self._tracer = provider.get_tracer(_SCOPE)
        self._meters = _Meters(meter_provider) if meter_provider is not None else None
        self._clock = clock

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
        """Open a CLIENT span for one call; `input` is never carried. A failure to open yields a no-op."""
        operation = str(metadata.get("operation") or DEFAULT_OPERATION)
        attrs: dict[str, Any] = {
            semconv.OPERATION_NAME: operation,
            semconv.PROVIDER_NAME: provider,
            semconv.REQUEST_MODEL: model,
        }
        if session_id is not None:
            attrs[semconv.CONVERSATION_ID] = session_id
        attrs.update({key: str(metadata[m]) for m, key in _OPEN_META if metadata.get(m) is not None})
        try:
            span = self._tracer.start_span(f"{operation} {model}", kind=SpanKind.CLIENT, attributes=attrs)
        except Exception:
            return NoopGeneration()
        base = {k: attrs[k] for k in (semconv.OPERATION_NAME, semconv.PROVIDER_NAME, semconv.REQUEST_MODEL)}
        return OtelGeneration(span, self._meters, base, self._clock, tracer=self._tracer)


class OtelGeneration:
    """The `Generation` an `OtelTracer` opens; ends its span once and never raises."""

    def __init__(
        self,
        span: Any,
        meters: _Meters | None,
        base: Mapping[str, Any],
        clock: Callable[[], float],
        *,
        tracer: Any | None = None,
    ) -> None:
        self._span = span
        self._tracer = tracer if tracer is not None else trace.get_tracer(_SCOPE)
        self._meters = meters
        self._base = dict(base)
        self._clock = clock
        self._started = clock()
        self._ended = False

    def end(
        self,
        *,
        output: Any = None,
        usage: Mapping[str, int | None] | None = None,
        level: str | None = None,
        status_message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        usage_v2: Usage | None = None,
        cost: Cost | None = None,
        finish: Finish | None = None,
        answered: Answered | None = None,
        failure: Failure | None = None,
    ) -> None:
        """Set the result's attributes, record the metrics, end the span; `output` is never carried."""
        if self._ended:
            return
        self._ended = True
        with contextlib.suppress(Exception):  # telemetry never breaks the call
            attrs = _end_attributes(usage, usage_v2, cost, finish, answered, metadata)
            error = _error_type(level, failure)
            if error is not None:
                attrs[semconv.ERROR_TYPE] = error
            self._span.set_attributes(attrs)
            if error is not None:
                why = status_message or (failure.reason if failure is not None else None) or error
                self._span.set_status(Status(StatusCode.ERROR, why[:_STATUS_MAX]))
            self._record(attrs, error)
            self._span.end()

    def tool(self, *, name: str, call_id: str, duration_s: float, is_error: bool) -> None:
        """Write one tool execution as an INTERNAL child span `execute_tool {name}`; never raises."""
        with contextlib.suppress(Exception):  # telemetry never breaks the call
            end_ns = time.time_ns()
            start_ns = end_ns - max(0, int(duration_s * 1e9))
            attrs: dict[str, Any] = {
                semconv.OPERATION_NAME: "execute_tool",
                semconv.TOOL_NAME: name,
                semconv.TOOL_CALL_ID: call_id,
            }
            if is_error:
                attrs[semconv.ERROR_TYPE] = "tool_error"
            span = self._tracer.start_span(
                f"execute_tool {name}",
                context=trace.set_span_in_context(self._span),
                kind=SpanKind.INTERNAL,
                attributes=attrs,
                start_time=start_ns,
            )
            if is_error:
                span.set_status(Status(StatusCode.ERROR, "tool_error"))
            span.end(end_time=end_ns)

    def _record(self, attrs: Mapping[str, Any], error: str | None) -> None:
        if self._meters is None:
            return
        dims = dict(self._base)
        if semconv.RESPONSE_MODEL in attrs:
            dims[semconv.RESPONSE_MODEL] = attrs[semconv.RESPONSE_MODEL]
        if error is not None:
            dims[semconv.ERROR_TYPE] = error
        self._meters.duration.record(max(0.0, self._clock() - self._started), dims)
        # The token counters require a modality; a reply's usage does not say which, so "unknown"
        # (the convention's value for exactly that).
        token_dims = {**dims, semconv.TOKEN_MODALITY: "unknown"}
        for field, key, _ in _USAGE:
            if key in attrs:
                self._meters.tokens[field].add(attrs[key], token_dims)

    def score(
        self, name: str, value: float, comment: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Do nothing: scores belong to the trace store (an ended span takes no new data)."""


def _error_type(level: str | None, failure: Failure | None) -> str | None:
    if failure is not None:
        return failure.kind
    if level == "CANCELLED":
        return "cancelled"
    if level == "ERROR":
        return "_OTHER"  # the convention's value for an error with no known type
    return None


def _usage_sides(usage: Mapping[str, int | None] | None, usage_v2: Usage | None) -> dict[str, int]:
    if usage_v2 is not None:
        sides = {field: getattr(usage_v2, field) for field, _, _ in _USAGE}
    else:
        sides = {"input": (usage or {}).get("input"), "output": (usage or {}).get("output")}
    return {k: int(v) for k, v in sides.items() if v is not None}


def _end_attributes(
    usage: Mapping[str, int | None] | None,
    usage_v2: Usage | None,
    cost: Cost | None,
    finish: Finish | None,
    answered: Answered | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    sides = _usage_sides(usage, usage_v2)
    attrs.update({key: sides[field] for field, key, _ in _USAGE if field in sides})
    if not sides:
        attrs[semconv.USAGE_UNMEASURED] = True
    if answered is not None:
        opt = {
            semconv.RESPONSE_MODEL: answered.model,
            semconv.RESPONSE_ID: answered.response_id,
            semconv.TIER: answered.tier,
            semconv.DEPLOYMENT: answered.deployment,
        }
        attrs.update({k: v for k, v in opt.items() if v is not None})
    if finish is not None:
        attrs[semconv.FINISH_REASONS] = (str(finish.unified),)
    if cost is not None:
        attrs[semconv.COST_SOURCE] = str(cost.source)
        if cost.usd is not None:
            attrs[semconv.COST_USD] = float(cost.usd)
    meta = metadata or {}
    if meta.get("outcome") is not None:
        attrs[semconv.OUTCOME] = str(meta["outcome"])
    if isinstance(meta.get("attempts"), int):
        attrs[semconv.ATTEMPTS] = meta["attempts"]
    return attrs


__all__ = ["DEFAULT_OPERATION", "OtelGeneration", "OtelTracer"]
