---
status: accepted
date: 2026-11-12
---

# Context is measured, not estimated: a pre-flight count, a trim that keeps turns whole, a cache plan by capability

## Context and problem statement

A long conversation, or a prompt with a large document, can exceed the model's window. The wire
then refuses it (`context_overflow`), after the call was queued and traced, and a caller that
retried the same request got the same refusal. A characters-divided-by-four estimate is wrong
by tens of percent for code, JSON and non-English text, so a guard built on it refuses requests
that fit and passes requests that do not: a guard that looks armed and is not.

Prompt caching has the same shape: a hosted model caches a prefix up to a breakpoint, a
llama.cpp server caches by slot with `cache_prompt`, and a model with neither silently ignores a
hint, so a caller cannot tell whether the hint did anything.

## Decision

1. **A `TokenCounter` port returns an exact count or None.** `Counted(tokens, method, failure)`:
   `provider` (the provider's own count API), `tokenizer` (the model's tokenizer), or
   `unsupported`. It never returns an estimate. `adapters.http.HttpTokenCounter` speaks LiteLLM's
   `/utils/token_counter` shape; the `openai_compatible` factory builds one for every model whose
   table entry states `count_tokens`.
2. **`Role.context` chooses the pre-flight.** `off` (the default) counts nothing. `check` refuses
   an overflow before the wire: `request_fault` / `context_overflow`, `provider_code="preflight"`,
   zero generation requests. `trim` cuts the history and counts again to confirm; when the first
   cut was not enough it cuts every turn the policy allows, and refuses what still does not fit.
   The budget is the table's `context_window` less the role's reply ceiling (`max_tokens`, else
   the model's `max_output`) and a 5 % margin. Both `call.generate` and `call.chat` run it.
3. **Unknown sends.** No counter, a model that cannot be counted, a failed count request or no
   window in the table is an `unknown` verdict: the request is sent, the result carries a
   `CallWarning`, and `llm_call_warning` fires once per model. A missing optimisation is never an
   outage.
4. **Trimming keeps turns whole.** Tool outputs over `TrimPolicy.tool_output_chars` are cut in
   the middle first; then whole turns (a user message and everything to the next one) are dropped
   oldest-first between `keep_first` and `keep_last`. A tool call is never separated from its
   result.
5. **Every count is reported.** `stamp["count_requests"]`, `stamp["trimmed"]`, and the events
   `llm_count_failed`, `llm_trimmed`, `llm_context_overflow` (`where` = `preflight` or `wire`) and
   `llm_count_optimistic` (the pre-flight said it fits; the wire said it did not).
6. **The cache hint follows the stated capability.** `Role.cache_prefix` on a model without
   `cache` is dropped with a `CallWarning`; on a self-hosted model with `cache` it becomes
   `extra_body.cache_prompt`; on a hosted model it becomes a pydantic-ai `CachePoint` after the
   system prompt and tools. `context.volatile_system` lints a pack whose system text would defeat
   the breakpoint.

## Considered options

- **tiktoken locally.** Rejected as the default: exact only for OpenAI's own encodings, and wrong
  for the self-hosted models this package serves; a consumer can still wrap it in a
  `TokenCounter` and state `tokenizer`.
- **Refuse on unknown.** Rejected: the guard would turn every uncounted model into an outage.

## Consequences

- A count request adds one round trip per call when the role opts in; `off` remains the default.
- Whether a given gateway forwards `/utils/token_counter` for a given model is the table's
  statement (`count_tokens`), checked by `testing.conformance.check_token_counter`.
