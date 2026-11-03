---
status: accepted
date: 2026-11-03
---

# A model call ends in a named outcome and never raises

Decided in 0.1.0 (`Generator`, `generate`), extended in 0.8.0 (`Embedder`) and 0.11.0
(`Reranker`). Recorded in 0.12.0.

## Context and problem statement

A call to a language model fails in many ways that are not bugs: the server is down, the gateway
refuses, the request is too large, the reply hits the token cap, the reply does not parse. A
caller that receives these as exceptions has to know every SDK's exception tree (pydantic-ai wraps
openai wraps httpx wraps a transport), and a forgotten `except` turns a model outage into a crashed
request. A caller that receives `None` cannot tell "the model is down" from "the model said
nothing".

## Decision drivers

- A caller's fallback policy should be a `match` on a small, closed set of names.
- The reason a call failed must be recordable without leaking a provider's response body.
- A failure must stay distinguishable from an empty answer.

## Considered options

1. Raise library exceptions, one class per failure kind.
2. Return `None` on failure.
3. Return a result whose `outcome` is one of a closed set of names.

## Decision outcome

Chosen option 3. `Generator.generate`, `Embedder.embed` and `Reranker.rerank` return a result with
`outcome` in `ok | gated | unavailable | request_fault | truncated | invalid` and a one-line
`reason`. The adapter classifies the exception chain, root cause first:

- a gate refusal anywhere in the chain wins (`gated`);
- a `length` finish is a budget failure (`truncated`) even when it surfaces as a parse error;
- then status codes (`request_fault` for a refused request such as 400/413/422, `unavailable`
  otherwise);
- then the parse failure itself (`invalid`).

A partial result is `invalid`, never a short list: a batch with one vector missing would shift
every later vector onto the wrong row. `failure.failure_reason` keeps the exception class, status
code and the first line of the message, cut before any `body:` payload, within 200 characters.

`Judge` is the exception. It raises when the judging model cannot answer, and the caller records
the sample as `could_not_run`. An unjudged sample must never read as a pass, and a `Judge` result
with a `failed` outcome would invite exactly that default.

### Consequences

- Good: fallback policy is a `match` on `res.outcome`; a dead model is a recorded row, not a stack
  trace.
- Good: one taxonomy across three ports; the embedder reuses the generator's classifier.
- Bad: a caller whose contract is that failure propagates must opt in (`GenerationFailed`,
  raised by the facade for that caller).
- Neutral: adapters retry nothing by default (`max_retries=0`), because a gateway in front
  usually retries and two stacked policies multiply.

## More information

- A provider error's full text once reached a user-facing feed through a persisted reason field:
  a 300-character `ModelHTTPError: status_code: 500, model_name: ..., body: {...}`, one row per
  scheduler tick. A proxy error can also embed a keyed URL. Hence the body-safe cut.
- With a judge asked for its verdict as a tool call (pydantic-evals' default), a self-hosted model
  answered in a mix of its XML tool-call syntax and JSON: 400 of 854 judged rows carried a reason
  ending `", "pass": false, "score": 0.0` or a `<parameter=pass>` tag. The judge adapter now asks
  for the native structured-output grammar, and a reply it cannot read raises rather than being
  guessed.
