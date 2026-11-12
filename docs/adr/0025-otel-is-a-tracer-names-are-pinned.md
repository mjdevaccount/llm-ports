---
status: accepted
date: 2026-11-12
---

# OTel is a Tracer; names are pinned

Decided in the release after 1.23.0.

## Context and problem statement

The library observed a model call two ways. A `Tracer` (the trace store) received the whole
generation through `observe`. OpenTelemetry received a span only where a caller remembered to
open one with `gen_ai_span` or `record_gen_ai`, so a consumer with both sinks emitted twice by hand
at every call site, and the two sinks saw different fields.

The OpenTelemetry GenAI conventions are `development`, have no release, and renamed their cache
and metric names this year (`cache_creation` to `cache_write`; `gen_ai.client.token.usage` split
into per-kind counters); `gen_ai.system` was removed at v1.37.0 for `gen_ai.provider.name`.
pydantic-ai, the trace store and the convention's HEAD currently disagree on names.

`observe` caught `Exception` only, so a cancelled call (`asyncio.CancelledError` is a
`BaseException`) left its generation open forever.

## Decision drivers

- One lifecycle and one payload for every path: `generate`, a consumer's own `observe`, and a
  call the library did not make.
- A convention rename is a one-file change, and something outside the unit tests notices it.
- Bodies (prompts, outputs) stay in the access-controlled trace store.
- Unmeasured is absent, never zero (ADR-0007), on every sink.

## Decision

1. **OpenTelemetry is a `Tracer`.** `adapters.otel.OtelTracer` implements the port:
   `start_generation` opens a CLIENT span named `"{operation} {model}"` in the ambient context;
   `end` sets the result's attributes and `error.type = failure.kind`. Given a meter provider it
   records `gen_ai.client.inference.duration` and one counter per measured token kind, and no
   other metric. Two sinks compose as `TeeTracer(trace_store, OtelTracer())`; the tee isolates
   each sink (an exception is counted in `stats()` and emits a throttled `tracer_sink_failed`).
2. **Names are pinned and isolated.** Every attribute and metric name lives in
   `observability.semconv`, with the commit of `semantic-conventions-genai` it was checked
   against (`SEMCONV_SOURCE`). A unit test asserts every emitted key is one of those constants.
   A weekly CI job (`semconv-drift.yml`) diffs the pinned names against the convention's HEAD,
   opens an issue on drift, and goes red when it cannot fetch (exit 2), never green.
3. **Bodies stay off OTel.** `OtelTracer` never sets `gen_ai.input.messages`,
   `gen_ai.output.messages` or `gen_ai.system_instructions`, and reads no environment variable
   (the content-capture switch included).
4. **`gen_ai.system` is no longer emitted.** The old helpers keep their signatures and
   `SYSTEM` stays exported for 1.x imports.
5. **pydantic-ai's own instrumentation stays off** inside the adapter: two emitters would double
   every span in a third name dialect.
6. **One lifecycle, one payload.** `observe` ends its generation exactly once on every exit; a
   `BaseException` (cancel, interrupt) ends it at level `CANCELLED` and re-raises. `generate`,
   `observe` and `observability.record` (one call for a finished outside call) build the end
   through one function, and `Generation.end` gains keyword-only `usage_v2`, `cost`, `finish`,
   `answered` and `failure`, passed only to an `end` that accepts them (a 1.x implementation keeps
   working).

## Considered options

- **Keep the helpers and document dual emission.** Rejected: the two sinks drift, and every
  consumer re-implements the fan-out.
- **Use the `opentelemetry-semantic-conventions` constants.** Rejected for the GenAI names: the
  package lags the GenAI repo and its incubating module renames on its own schedule. The pin is
  the GenAI repo's own commit.
- **Turn on pydantic-ai's instrumentation.** Rejected: double spans, a different convention
  version, and bodies on by default.
- **Bodies on OTel behind a switch.** Rejected: bodies belong in the access-controlled trace
  store; a switch is one deploy away from leaking them.

## Consequences

- A consumer adds OTel with one line at its composition root, and every call the library makes
  is a span with the same fields the trace store sees.
- A convention rename is a diff to `semconv.py`, a changelog line, and a closed drift issue.
- `observe` now ends a cancelled call's generation (at `CANCELLED`, Langfuse `WARNING`), so a
  dashboard counting open generations no longer counts cancelled ones.
- Tool-execution child spans (`execute_tool {name}`) land with the chat port (spec 5).
