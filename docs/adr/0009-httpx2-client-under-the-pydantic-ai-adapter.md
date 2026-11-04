---
status: accepted
date: 2026-11-03
---

# The pydantic-ai adapter hands its provider an httpx2 client

Decided in 1.0.1.

## Context and problem statement

`openai_compatible` and `openai_compatible_embeddings` build the HTTP client their OpenAI provider
rides, so they can install a response hook (the wire capture) and the index-order re-sort. Through
1.0.0 that was an `httpx.AsyncClient`. pydantic-ai moved its OpenAI-compatible providers to httpx2
(v2.32.0, pydantic/pydantic-ai#7351; the OpenAI SDK 3 is built on it), and pydantic-ai 2.65 warns
on every call:

```text
PydanticAIDeprecationWarning: `httpx.AsyncClient` support for OpenAI-compatible providers is
deprecated and will be removed in v3; use `httpx2.AsyncClient` instead.
```

The warning is a `UserWarning` subclass, so `-W error::DeprecationWarning` never caught it, and
under pydantic-ai 3 the 1.0.0 builders stop working. httpx2 is a fork of httpx 0.28.1 with the
same API under another import name, but its classes are distinct: an httpx transport cannot be
mounted on an httpx2 client. Both builders take a caller's `transport=`, and callers pass legacy
httpx transports (a `MockTransport` in tests, a counting or admitting transport in production).

## Decision drivers

- No deprecated path in the calls this package makes, before pydantic-ai 3 removes it.
- A caller's legacy httpx transport keeps working: 1.x does not break `transport=`.
- A deprecation at our call site fails the suite instead of reaching consumers' logs.

## Considered options

1. Keep the httpx client and filter the warning.
2. Switch to httpx2 and accept only httpx2 transports.
3. Switch to httpx2, accept a transport from either library, and bridge a legacy one.

## Decision outcome

Chosen option 3, in `adapters.pydantic_ai._wire`:

- Both builders build an `httpx2.AsyncClient`. `transport=` accepts an httpx2 transport as is; a
  legacy one rides `LegacyTransport`, which copies the request across (method, URL, headers,
  body, extensions, so the client's timeout still applies) and the reply back, with the body read
  whole and its encoding headers dropped. The adapter's calls do not stream.
- A legacy `httpx.TransportError` re-raises as the httpx2 error of the same name, chained, so the
  OpenAI SDK's connection and timeout handling sees its own types. Any other exception crosses
  unchanged, so a caller's gate refusal still classifies as `gated`.
- The index-order re-sort is library-free (`ordered_reply`); the public
  `IndexOrderedTransport` stays an httpx transport for callers that build their own httpx client,
  and the httpx2 builder uses a private twin.
- `classify` maps an `httpx2.HTTPStatusError` as it maps an httpx one.
- pytest turns a `DeprecationWarning` or `UserWarning` attributed to an `llm_ports` frame into an
  error (`filterwarnings` in `pyproject.toml`).

Option 1 only hides the warning, and breaks at pydantic-ai 3. Option 2 breaks every caller's
legacy transport in a patch release. `httpx2.alias_httpx()` is for applications only, never
libraries, and no library bridges the two transport types, so the bridge is ours.

### Consequences

- Good: no pydantic-ai deprecation on either builder, and a caller's transport keeps working.
- Good: a new deprecation at our call site fails CI.
- Neutral: the default transport is httpx2's, which verifies TLS against the OS trust store
  (`truststore`) rather than `certifi`; `SSL_CERT_FILE` and `SSL_CERT_DIR` are still honoured.
- Bad: a legacy transport's reply is buffered in full; streaming through the bridge is not
  supported.
