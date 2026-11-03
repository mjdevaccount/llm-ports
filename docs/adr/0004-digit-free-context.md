---
status: accepted
date: 2026-11-03
---

# Context a digit-free writer reads is digit-free too

Decided in 0.6.0 (`slots.to_prose`). Recorded in 0.12.0.

## Context and problem statement

The digit-free grammar ([ADR-0002](0002-grounded-prose-by-construction.md)) stops the model
writing a digit, but the prompt can still show it figures in words and free text: a user's note
such as `p=0.81 (at +0.13%)`, a tool's reason string, a JSON key. A model at low temperature
reaches for a figure it can see. With the digit token masked, the top admissible token is the tail
of a number word, and with no repetition penalty the model emits that tail until the length cap.

On a production writer, 47 of 105 replies were the fragment ` ight` repeated 80 times, and the
sampled judge failed 11 of 11. In a replay of one input, three samples per variant: the original
context looped 3 of 3; with the context sanitised, 0 of 3.

Telling the model to ignore figures in context does not work either: when a cached research note
was allowed to back a fact but not a figure, 18 of 32 claims still carried the note's figure.

## Decision drivers

- The figures the model must not copy should be absent from what it reads, not forbidden in what
  it writes.
- A value the writer needs stays reachable by its slot id.

## Considered options

1. Add a repetition penalty to the sampler.
2. Instruct the model not to use numbers from the context.
3. Remove every digit run from free-text context before the model reads it.

## Decision outcome

Chosen option 3. `slots.to_prose` removes every digit run whole (`numbers.DIGITS` reads `14:00`,
`3.1%` and `5,000` as one run each, so no half a number remains), turns every character outside
`PROSE_CHARS` into a space, and collapses whitespace. `numbers.redact` replaces each number with a
caller-chosen hole for context that must keep its shape. Rendered values are never passed through
either: code fills those.

### Consequences

- Good: the loop has nothing to start from; the grammar and the context agree.
- Bad: context loses its figures; a writer that needs one must have it as a rendered slot.
- Neutral: the sampler setting is left to the model server; the fix holds at any setting.
