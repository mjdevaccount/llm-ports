---
status: accepted
date: 2026-11-03
---

# One observation lifecycle; unmeasured usage is unmeasured, never zero

Decided in 0.11.0 (`observation`, `langfuse_tracer`, `otel_gen_ai`). Recorded in 0.12.0.

## Context and problem statement

Each model call should produce one observation: what was asked, what came back, how long it took,
how many tokens it used, how it was scored. Two sinks read it (Langfuse and OpenTelemetry GenAI
spans), and consumers also wrap model calls the package's call path does not make. Separate
lifecycles drift: one sink ends a span on error, another leaves it open; one books missing usage
as zero.

`int(v or 0)` turns "no usage block arrived" into "this call used zero tokens". A cost dashboard
then reports a free model where it should report an unknown one.

## Decision drivers

- An observation is opened once and ended exactly once, on success and on error.
- Missing data reads as missing.
- Telemetry never breaks the call it describes.

## Considered options

1. Each adapter and consumer manages its own observation.
2. One context manager owns the lifecycle; adapters translate it into their sink.

## Decision outcome

Chosen option 2. `observation.observe(tracer, ...)` opens one generation, holds the last `update`,
and ends it once: on a clean exit with what was held; on an exception at `ERROR` with the error
line and measured latency, after which the exception propagates. `call.tracer_trace` is built on
it, and a consumer wrapping its own call uses the same function.

- Usage is the raw `{input, output}` pair the provider reported, each coerced by `opt_int`, which
  returns None for nothing readable. A reported zero stays zero.
- A call that reported neither count gets no usage block and `tokens_unmeasured: true`.
- `Tracer` and `Generation` never raise; a sink that is down yields a no-op generation and logs
  once, not once per call.
- The OpenTelemetry span is back-dated by the measured latency; opened and closed in the same
  instant it would publish a 0 ms model call. An error sets span status `ERROR`, so a swallowed
  failure never reads as a healthy call. Prompt and completion bodies are never put on a span:
  that is a prompt store's job, and a trace store fills up on them.

### Consequences

- Good: both sinks agree on every call; an unknown cost is visibly unknown.
- Bad: a sink that needs bodies on spans cannot get them from this adapter.
