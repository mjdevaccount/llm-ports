"""OpenTelemetry GenAI spans for model calls (extra `[otel]`).

`OtelTracer` is the `Tracer` port over OpenTelemetry: compose it like any other sink
(`TeeTracer(LangfuseTracer(client), OtelTracer())`) and every call the library makes is a span.
The two helpers below are for a call the library does not make.

Example:
    with gen_ai_span(tracer, operation="chat", model=m, provider="openai") as rec:
        reply = await client.call(...)
        rec.set_usage(reply.in_tok, reply.out_tok)

    record_gen_ai(tracer, operation="embeddings", model=m, provider="openai", latency_ms=840)

A span records the shape of a model call (who, which model, how long, how many tokens) in the
trace the rest of the request already lives in, named and attributed by the OpenTelemetry GenAI
semantic conventions. Attribute keys come from `opentelemetry-semantic-conventions`' constants,
so an upstream rename arrives with a version bump.

`tracer` is any OTel `Tracer`, or None when tracing is off; every function then does nothing, so a
call site needs no conditional. `attributes` carries the caller's own dimensions under its own
prefix, outside the reserved `gen_ai.*` namespace.

- `gen_ai_span` opens a child of the ambient context (`start_as_current_span`), so the model call
  lands inside the existing job or request span. Only span creation is guarded: an exception from
  the caller's block propagates, and the SDK records it on the span.
- `record_gen_ai` records a call after it is over, for a call that learns its latency and usage
  only then. The span is back-dated by the measured `latency_ms`, so its duration is the call's.
- `gen_ai.system` is no longer emitted (1.24): the conventions removed it in v1.37.0 for
  `gen_ai.provider.name`. `SYSTEM` stays exported for 1.x imports; nothing sets it.

Invariants:
    - Prompt and completion bodies are never carried, and no switch turns them on.
    - None is not zero: a provider that reported neither count gets the `unmeasured_attr` flag
      and no token attributes; a reported zero stays zero.

Rationale: ADR-0007, ADR-0025.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters._otel_tracer import DEFAULT_OPERATION, OtelGeneration, OtelTracer

with extra("otel"):
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as ga
    from opentelemetry.trace import Status, StatusCode


#: Deprecated: removed from the conventions at v1.37.0 and no longer emitted; kept for 1.x imports.
SYSTEM = ga.GEN_AI_SYSTEM
PROVIDER_NAME = ga.GEN_AI_PROVIDER_NAME
OPERATION = ga.GEN_AI_OPERATION_NAME
REQUEST_MODEL = ga.GEN_AI_REQUEST_MODEL
INPUT_TOKENS = ga.GEN_AI_USAGE_INPUT_TOKENS
OUTPUT_TOKENS = ga.GEN_AI_USAGE_OUTPUT_TOKENS
#: The default unmeasured flag. Not a convention key (the conventions have none), so it is not under `gen_ai.*`.
UNMEASURED = "llm.tokens_unmeasured"
#: Characters of an error message kept in a span status.
ERROR_MAX = 200


class GenAiRecorder:
    """Attaches a call's results to its span once the call returns.

    A no-op over `span=None`. Never raises.
    """

    __slots__ = ("_span", "_unmeasured")

    def __init__(self, span: Any, *, unmeasured_attr: str = UNMEASURED) -> None:
        self._span = span
        self._unmeasured = unmeasured_attr

    def _set(self, key: str, value: Any) -> None:
        if self._span is None or value is None:
            return
        with contextlib.suppress(Exception):  # telemetry must never break the call
            self._span.set_attribute(key, value)

    def set_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Set the token counts as reported; when neither is reported, set the unmeasured flag, never zeros."""
        self._set(INPUT_TOKENS, input_tokens)
        self._set(OUTPUT_TOKENS, output_tokens)
        if input_tokens is None and output_tokens is None:
            self._set(self._unmeasured, True)


def span_name(operation: str, model: str) -> str:
    """Return the conventions' span name, `{operation} {model}`; the call's identity lives in attributes."""
    return f"{operation} {model}"


def attributes_for(
    *, operation: str, model: str, provider: str, attributes: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the convention attributes plus the caller's own, with None values dropped. Pure."""
    out: dict[str, Any] = {PROVIDER_NAME: provider, OPERATION: operation, REQUEST_MODEL: model}
    out.update({k: v for k, v in (attributes or {}).items() if v is not None})
    return out


@contextlib.contextmanager
def gen_ai_span(
    tracer: Any | None,
    *,
    operation: str,
    model: str,
    provider: str,
    attributes: Mapping[str, Any] | None = None,
    unmeasured_attr: str = UNMEASURED,
) -> Iterator[GenAiRecorder]:
    """Open a child span around one live model call; the module docstring states the contract.

    Yields:
        A `GenAiRecorder` over the span, or a no-op recorder when `tracer` is None or span
        creation failed.
    """
    null = GenAiRecorder(None)
    if tracer is None:
        yield null
        return
    # Guard creation only: a try around the `yield` would re-enter this generator on the caller's
    # exception and lose it behind "generator didn't stop after throw()".
    try:
        cm = tracer.start_as_current_span(
            span_name(operation, model),
            attributes=attributes_for(operation=operation, model=model, provider=provider, attributes=attributes),
        )
    except Exception:
        yield null
        return
    with cm as span:
        yield GenAiRecorder(span, unmeasured_attr=unmeasured_attr)


def record_gen_ai(
    tracer: Any | None,
    *,
    operation: str,
    model: str,
    provider: str,
    latency_ms: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error: str | None = None,
    attributes: Mapping[str, Any] | None = None,
    unmeasured_attr: str = UNMEASURED,
) -> None:
    """Record a span for a call that already finished, back-dated by `latency_ms`; never raises.

    A non-empty `error` sets the span status to `ERROR`, with the message cut to `ERROR_MAX`
    characters; without one, the status is left unset.
    """
    if tracer is None:
        return
    try:
        end_ns = time.time_ns()
        start_ns = end_ns - max(0, int(latency_ms)) * 1_000_000
        span = tracer.start_span(
            span_name(operation, model),
            start_time=start_ns,
            attributes=attributes_for(operation=operation, model=model, provider=provider, attributes=attributes),
        )
        GenAiRecorder(span, unmeasured_attr=unmeasured_attr).set_usage(input_tokens, output_tokens)
        if error:
            span.set_status(Status(StatusCode.ERROR, error[:ERROR_MAX]))
        span.end(end_time=end_ns)
    except Exception:
        return


__all__ = [
    "DEFAULT_OPERATION",
    "ERROR_MAX",
    "INPUT_TOKENS",
    "OPERATION",
    "OUTPUT_TOKENS",
    "PROVIDER_NAME",
    "REQUEST_MODEL",
    "SYSTEM",
    "UNMEASURED",
    "GenAiRecorder",
    "OtelGeneration",
    "OtelTracer",
    "attributes_for",
    "gen_ai_span",
    "record_gen_ai",
    "span_name",
]
