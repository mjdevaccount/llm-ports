---
status: accepted
date: 2026-11-11
---

# A failure has a kind

Decided in the release after 1.22.0.

## Context and problem statement

ADR-0001 names six outcomes, and they are deliberately coarse: a fallback policy is a short
`match`. But two policies cannot be written on them. A retry policy cannot tell a full queue
(`rate_limited`, back in seconds) from a spent quota (`quota`, back tomorrow): both were
`unavailable`. A health tracker could not tell a model outage from a caller's bug: an exception
the classifier did not recognise (a `KeyError` in a validator) was `unavailable`, so the backend
was marked down for the caller's mistake. And a reply cut by the token ceiling was `ok` whenever
the cut text happened to parse.

Every consumer re-derived the detail from provider strings: a 429 with "budget" in the body, a
"Please retry in 8h26m" hint, a `finish_reason` of `length`.

## Decision drivers

- Outcomes stay coarse: existing `match` statements with a default arm keep compiling.
- The detail is classified once, by one table every adapter shares, from the exception chain and
  what the wire showed, with no response body in the reason.
- A bug is not an outage. A cut reply is not a whole one.

## Decision

1. `Outcome` gains exactly one member, `refused`: the model or its provider declined.
2. `Failure(kind, outcome, retryable, retry_after_s, status, provider_code, phase, in_stream,
   reason)`, with `outcome` and `retryable` derived from `kind` (`OUTCOME_FOR`, `RETRYABLE`).
   Sixteen kinds: `gated`; under `unavailable` `rate_limited`, `quota`, `overloaded`, `server`,
   `timeout`, `transport`, `auth`, `not_found`; under `request_fault` `bad_request`,
   `context_overflow`, `unsupported`, `internal`; `truncated`; `invalid_output`; `refused`.
3. One classification order (`adapters._classify.classify_core`): extra classifiers first (a
   gateway's knowledge), then a gated type in the chain, the framework's categories (pydantic-ai's
   `ModelAPIError` classes, walked over the whole chain), a context-overflow code or message, the
   HTTP status, a transport error, and otherwise `internal`. With no exception, a refusal or
   `content_filter` finish is `refused` and a `length` finish is `truncated`.
4. **Behaviour fix: a `length` finish is `truncated` on the success path too.** The cut value is
   kept on the result; `Role.accept_truncated` / `GenRequest.accept_truncated` restore `ok` with
   a `CallWarning`.
5. **Behaviour fix: an unrecognised exception is `internal`, under `request_fault`**, never
   retried and never counted against a backend. It is reported as `llm_call_internal_error` with
   its traceback.
6. `adapters.litellm_proxy.classify_gateway` names a 429 that no wait clears (a budget, a per-day
   limit, a provider retry hint over two minutes) `quota`; a parallel-limit 429 stays
   `rate_limited` with the server's `Retry-After`.
7. The pydantic-ai floor is 2.62, the first release with every category class the table reads.

## Considered options

- **Split `unavailable` into finer outcomes.** Every consumer's `match` would need new arms, and
  most policies only need the coarse split. The detail belongs in `kind`.
- **Keep classifying a bug as `unavailable`.** That is the failure mode this ADR removes: a
  validator bug took a healthy backend out of rotation.
- **Classify by provider strings in each adapter.** The reranker, the embedder and the generator
  would drift. One table, with an `extra` hook for gateway knowledge, keeps them equal.
- **Retry, deadlines and fallback here.** They read `kind` and `retryable` but are a separate
  decision (the next row on the critical path).

## Consequences

- A retry policy reads `failure.retryable` and `failure.retry_after_s`; the embedder's own retry
  now does, so `auth`, `not_found`, `quota` and `internal` are no longer retried.
- A consumer's `match` with a default arm keeps compiling; an exhaustive `match` must add
  `refused` (a type checker names the missing arm).
- A caller that relied on a cut reply being `ok` sets `accept_truncated`.
