---
status: accepted
date: 2026-11-12
---

# One retry owner and a total deadline

Decided in the release after 1.23.0.

## Context and problem statement

A call could be retried in three places at once: the provider SDK's own retries, a retrying HTTP
transport, and the application's loop around the call. Two retries in each multiply to eight
requests, and nobody can say how long a call may take. The only bound was the HTTP client's
per-phase timeout, which a slow trickle of bytes never trips. The embedder had its own retry,
the generator had none, and the reranker had none, so the same 429 meant three different things.

ADR-0020 gave every failure a `kind`. This ADR decides who acts on it, and for how long.

## Decision drivers

- Retries never multiply: a test can count the wire requests a call makes.
- A spent quota is never retried; a busy queue is.
- A caller can say how long a call may take, end to end, and the call ends then.
- A caller's cancellation is never turned into a result.

## Decision

1. **One retry owner: the adapter's own loop, keyed on `Failure.kind`** (`adapters._retry`,
   shared by the generator, the embedder and the reranker). The provider SDK stays at
   `max_retries=0`. A `WirePolicy` maps each retryable kind to a `RetryRule` (tries, exponential
   wait with jitter, whether a server hint is honoured and up to what). Only `rate_limited`,
   `overloaded`, `server`, `timeout` and `transport` can carry a rule, so `quota`, `auth` and a
   request-side kind are never retried by construction. From the SDKs: `retry-after-ms` beats
   `Retry-After`, and `x-should-retry: false` ends the retry whatever the kind.
2. **Retries multiply nowhere.** A wire retry re-sends one model request. pydantic-ai's output
   re-ask is a new model request, with its own wire retries. Fallback (ADR-0022) starts only
   after a tier's wire policy gave up. The pydantic-ai generator wraps the call's model in a
   `WrapperModel`, so every model request the agent makes goes through the one loop.
3. **`Deadline.total_s` is a wall-clock bound on the whole call.** The facade starts it when
   `generate` is entered and passes it down on `GenRequest.deadline`; the adapter ends the call
   AT it (`asyncio.timeout_at`), across the slot wait, every request, every backoff and every
   fallback tier. A backoff that would end past it ends the call at once. An expiry is a
   `timeout` result with `phase="total"`. A generator that ignores the deadline is cut off
   `DEADLINE_GRACE_S` after it by the facade.
4. **`Deadline.attempt_s` bounds one request** and is a retried `timeout`; `first_event_s` bounds
   the wait for a streamed reply's first event.
5. **A cancellation propagates.** `asyncio.CancelledError` is never retried, never caught, and
   never a result; the limiter slot is released on the way out.
6. **Concurrency is a slot taken inside the deadline and outside the retry loop** (`Limiter`,
   `call.resilience.LocalLimiter`). The wait spends the caller's deadline and is reported as
   `limiter_wait_s`; a retry keeps its slot. Waiting past `max_wait_s` is `rate_limited` with
   `provider_code="local_limiter"`, not retried.
7. **One HTTP client per generator per event loop**, reused across calls and closed by
   `aclose()`; the per-request bound is set per call.
8. **The generator retries by default** (`WirePolicy.default()`). The embedder and reranker keep
   their 1.x default of one request; `RetryPolicy` (1.x) is `RetryPolicy.wire()`.

## Considered options

- **pydantic-ai's `AsyncHTTPX2TenacityTransport`.** It retries below the classification, so it
  cannot tell a spent budget from a busy slot, and its attempts would be invisible to
  `Generated.attempts`. Not installed; its header precedence is taken.
- **The SDK's own retries.** Same blindness, and they stack under any loop above them.
- **pydantic-ai's `request_timeout` as the per-request bound.** The agent graph applies it around
  the wrapper's whole retry loop, not one request, so the adapter enforces `attempt_s` itself.
- **A distributed limiter in the library.** A gateway already is one (ADR-0018 keeps scheduling
  with the consumer); the in-process limiter only stops one process queueing behind itself.

## Consequences

- A call's `attempts` lists every retried request (kind, wait), each tagged with its tier;
  `stamp["wire_requests"]` and `stamp["limiter_wait_s"]` are on every result.
- `llm_wire_retry`, `llm_deadline_expired` and `llm_fallback` are events a consumer can count.
- A caller of the generator who wants 1.x's single request passes `policy=WirePolicy.none()`.
- A cancelled or timed-out request may still be billed; `Role.stream_internally` streams it so
  closing the connection is a signal the server can act on.
