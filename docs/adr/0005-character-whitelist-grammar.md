---
status: accepted
date: 2026-11-03
---

# The digit-free grammar is a character whitelist

Decided in 0.3.0 (`slots.PROSE_CHARS`). Recorded in 0.12.0.

## Context and problem statement

The digit-free grammar ([ADR-0002](0002-grounded-prose-by-construction.md)) reaches the model as a
JSON-schema `pattern` that the model server compiles into a sampling grammar. The obvious pattern,
`^[^0-9]*$`, was probed on llama.cpp's json-schema-to-grammar converter and failed: the pattern is
compiled outside JSON string escaping, so the negated class admitted raw control bytes, and the
model emitted replies that `json.loads` could not parse.

## Decision drivers

- Nothing the grammar admits may need JSON escaping.
- The pattern must compile on the converter's regex subset.
- A length cap must hold under the grammar, not only in validation afterwards.

## Considered options

1. A negated class: `[^0-9]`.
2. A negated class that also excludes control characters and quotes.
3. A whitelist of the characters prose needs: letters, space and `,.;:'()/&?!-`.

## Decision outcome

Chosen option 3, `PROSE_CHARS = [A-Za-z ,.;:'()/&?!-]`. Constraints of the converter shaped the
rest:

- `maxLength` is not enforced alongside `pattern`, so a capped run carries its cap inside the
  pattern as `{0,N}`. A slot counts as one repetition, so the grammar's cap is approximate; the
  pydantic field keeps the exact `max_length`.
- The regex subset has no `(?:`, so groups are plain `(...)`.
- The complement used by `to_prose` is derived from `PROSE_CHARS`, so the two cannot drift.

### Consequences

- Good: a grammar-admitted reply always parses as JSON.
- Bad: prose loses characters such as `"`, `%`, `$` and non-ASCII letters; figures and symbols come
  from rendered slots.
- Bad: a stricter or richer alphabet is a change to every consumer's grammar, so `PROSE_CHARS` is
  part of the public contract.
