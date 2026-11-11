---
status: accepted
date: 2026-11-11
---

# A result names what was measured, what it cost and who answered

Decided in the release after 1.22.0.

## Context and problem statement

Through 1.22.0 a `Generated` reported `usage` as `{"input", "output"}` from the last reply on a
failed run and the run's total on a successful one, `model` as the id the request named, and
`finish_reason` as a provider string. A consumer that needed more rebuilt it from the wire: its
own capture for cached and reasoning tokens, its own price lookup, its own reading of which
deployment a gateway alias resolved to. Each rebuild coerced an unknown to zero somewhere
(`tokens or 0`, `cost or 0.0`), so an unpriced call looked free and a USD cap never fired on it.

The question: what does a result say about the call it describes, so that no consumer re-derives
it, and so that an unknown stays visible as unknown?

## Decision drivers

- Unknown is None at every level. A side the provider did not report is never 0.
- A call is every request it made: output re-asks, tool steps, wire retries that returned a body.
- A gateway in front knows the deployment and its negotiated price; the body only knows the alias.
- 1.x is additive: no public name is removed, and the 1.x fields keep their 1.22.0 values.

## Decision

1. `Usage` (ports) has inclusive buckets: `input` includes cached tokens, `output` includes
   reasoning tokens; `cache_read`, `cache_write` and `reasoning` are the parts. `requests` counts
   wire requests and `unmeasured` those that reported no usage. `None + n = n`, and `complete` is
   false while any request was unmeasured. An adapter for a provider that reports exclusive
   buckets normalises them; the OpenAI wire is already inclusive.
2. `Generated.usage_v2` is the sum over every request of the call. The capture keeps every 2xx
   reply (`WireReply`), not the last; the framework's own `ModelResponse` usage is the fallback
   when the call did not ride `openai_compatible`.
3. `Cost(usd, source, priced_model, reason)` with precedence: a gateway's cost header, then a
   `Pricer`, then unknown. `usd` is None when unknown, with a reason (`no_usage`,
   `unknown_model`, `pricer_error`). `Cost` itself is None only when no pricer and no gateway
   cost exist. A self-hosted model costs what the caller says (`overrides`), never free by
   default.
4. `Answered(requested, model, deployment, response_id, ...)` says who answered: `model` is what
   the response body reported, `deployment` what the gateway's header reported.
5. `Finish(unified, raw)`: a closed vocabulary with `other` for an unknown string; `raw` is never
   rewritten, and absent is None, never `stop`.
6. A setting the model class declares it ignores (`Model.unsupported_settings`, read before the
   call) is a `CallWarning(kind="unsupported")` on the result.
7. Two seams, both constructor arguments: `Pricer` (`adapters.genai_prices.GenaiPricesPricer` on
   genai-prices' dated table) and `GatewayReader` (`adapters.litellm_proxy`, no vendor import).
8. The 1.x fields stay: `usage` is derived from `usage_v2` on a returned run and keeps the last
   reply's on a failed one (as 1.22.0 did); `finish_reason` is `finish.raw`; `model` stays the
   requested id. A golden fixture written against 1.22.0 pins all of them.

## Considered options

- **Report `Generated.model` as the model that answered.** Rejected for 1.x: it is a third
  behaviour change to a field consumers key ledgers on. `answered.model` carries it; `model` is
  deprecated in its docstring.
- **A price table inside the library.** Prices change weekly; genai-prices ships a dated,
  Decimal table and its own refresh.
- **Capture setting warnings with `warnings.catch_warnings`.** The filter is process-global, so
  concurrent calls would see each other's warnings. pydantic-ai declares the ignored settings on
  the model class; reading that before the call is exact and needs no global state.
- **Sum usage in the 1.x `usage` field on failure too.** More correct, but a change to a 1.x value
  the golden fixture forbids; `usage_v2` is the correct sum.

## Consequences

- A consumer deletes its own capture, price lookup and zero-coercions and reads the result.
- An unpriced model is visible: `llm_cost_unknown` once per model per process, `cost.usd is None`.
- A call with no usage block is `unmeasured`, reported per call as `llm_usage_unmeasured`.
- On the framework path (a fixed `model=`), pydantic-ai's sub-buckets are plain counts: a provider
  that reports no cached tokens reads 0 there; the wire path keeps None.
- A unit test greps `src/` for `or 0` in code and fails unless the line carries
  `# zero-ok: <why>`.
