"""otel_gen_ai -- OpenTelemetry GenAI spans for model calls (extra `[otel]`).

    with gen_ai_span(tracer, operation="chat", model=m, provider="openai") as rec:
        reply = await client.call(...)
        rec.set_usage(reply.in_tok, reply.out_tok)

    record_gen_ai(tracer, operation="embeddings", model=m, provider="openai", latency_ms=840)

The SHAPE of a model call -- who, which model, how long, how many tokens -- as a span in the
trace the rest of the request already lives in, named and attributed by the OpenTelemetry GenAI
semantic conventions (`opentelemetry-semantic-conventions`' constants, so a rename upstream is a
version bump, not a hunt). Prompt and completion BODIES are never carried: that is a prompt
store's job, and a trace store fills up on them. There is no switch to turn bodies on.

`tracer` is any OTel `Tracer`, or None when tracing is off -- every function then does nothing,
so a call site needs no conditional. `attributes` carries the consumer's own dimensions under its
own prefix (never inside the reserved `gen_ai.*` namespace).

- `gen_ai_span` opens a CHILD of the ambient context (`start_as_current_span`), so the model call
  lands inside the job or request span that already exists. Only span CREATION is guarded: an
  exception from the caller's block propagates and the SDK records it on the span.
- `record_gen_ai` is the after-the-fact shape, for a call that learns its latency and usage only
  once it is over. The span is BACK-DATED by the measured `latency_ms`: opened and closed in the
  same instant it would publish a 0 ms model call into a latency waterfall.
- NONE IS NOT ZERO: a provider that reported neither count gets the `unmeasured_attr` flag and no
  token attributes; a reported zero stays a zero.
- `gen_ai.system` (deprecated in the conventions) is emitted beside `gen_ai.provider.name`, which
  replaced it, so a query written against either keeps working.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping
from typing import Any

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as ga
from opentelemetry.trace import Status, StatusCode

SYSTEM = ga.GEN_AI_SYSTEM
PROVIDER_NAME = ga.GEN_AI_PROVIDER_NAME
OPERATION = ga.GEN_AI_OPERATION_NAME
REQUEST_MODEL = ga.GEN_AI_REQUEST_MODEL
INPUT_TOKENS = ga.GEN_AI_USAGE_INPUT_TOKENS
OUTPUT_TOKENS = ga.GEN_AI_USAGE_OUTPUT_TOKENS
#: The default unmeasured flag. Not a convention key (the conventions have none), so it is not under `gen_ai.*`.
UNMEASURED = "llm.tokens_unmeasured"
ERROR_MAX = 200


class GenAiRecorder:
    """What a call site attaches once the call returns. A no-op over `span=None`; never raises."""

    __slots__ = ("_span", "_unmeasured")

    def __init__(self, span: Any, *, unmeasured_attr: str = UNMEASURED) -> None:
        self._span = span
        self._unmeasured = unmeasured_attr

    def _set(self, key: str, value: Any) -> None:
        if self._span is None or value is None:
            return
        with contextlib.suppress(Exception):     # telemetry must never break the call
            self._span.set_attribute(key, value)

    def set_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Token counts as reported; neither reported = the unmeasured flag, never zeros."""
        self._set(INPUT_TOKENS, input_tokens)
        self._set(OUTPUT_TOKENS, output_tokens)
        if input_tokens is None and output_tokens is None:
            self._set(self._unmeasured, True)


def span_name(operation: str, model: str) -> str:
    """The conventions' `{operation} {model}`: low cardinality, the call's identity is attributes."""
    return f"{operation} {model}"


def attributes_for(*, operation: str, model: str, provider: str,
                   attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Pure. The convention attributes plus the consumer's own (None values dropped)."""
    out: dict[str, Any] = {SYSTEM: provider, PROVIDER_NAME: provider, OPERATION: operation,
                           REQUEST_MODEL: model}
    out.update({k: v for k, v in (attributes or {}).items() if v is not None})
    return out


@contextlib.contextmanager
def gen_ai_span(tracer: Any | None, *, operation: str, model: str, provider: str,
                attributes: Mapping[str, Any] | None = None,
                unmeasured_attr: str = UNMEASURED) -> Iterator[GenAiRecorder]:
    """A child span around one live model call (module docstring)."""
    null = GenAiRecorder(None)
    if tracer is None:
        yield null
        return
    # Guard creation ONLY: a try around the `yield` would re-enter this generator on the caller's
    # exception and lose it behind "generator didn't stop after throw()".
    try:
        cm = tracer.start_as_current_span(
            span_name(operation, model),
            attributes=attributes_for(operation=operation, model=model, provider=provider,
                                      attributes=attributes))
    except Exception:
        yield null
        return
    with cm as span:
        yield GenAiRecorder(span, unmeasured_attr=unmeasured_attr)


def record_gen_ai(tracer: Any | None, *, operation: str, model: str, provider: str, latency_ms: int,
                  input_tokens: int | None = None, output_tokens: int | None = None,
                  error: str | None = None, attributes: Mapping[str, Any] | None = None,
                  unmeasured_attr: str = UNMEASURED) -> None:
    """A span for a call that already FINISHED, back-dated by `latency_ms`; `error` set = status
    ERROR (a swallowed failure would otherwise close the span UNSET and read as a healthy call).
    Never raises."""
    if tracer is None:
        return
    try:
        end_ns = time.time_ns()
        start_ns = end_ns - max(0, int(latency_ms)) * 1_000_000
        span = tracer.start_span(
            span_name(operation, model), start_time=start_ns,
            attributes=attributes_for(operation=operation, model=model, provider=provider,
                                      attributes=attributes))
        GenAiRecorder(span, unmeasured_attr=unmeasured_attr).set_usage(input_tokens, output_tokens)
        if error:
            span.set_status(Status(StatusCode.ERROR, error[:ERROR_MAX]))
        span.end(end_time=end_ns)
    except Exception:
        return


__all__ = ["ERROR_MAX", "INPUT_TOKENS", "OPERATION", "OUTPUT_TOKENS", "PROVIDER_NAME", "REQUEST_MODEL",
           "SYSTEM", "UNMEASURED", "GenAiRecorder", "attributes_for", "gen_ai_span", "record_gen_ai",
           "span_name"]
