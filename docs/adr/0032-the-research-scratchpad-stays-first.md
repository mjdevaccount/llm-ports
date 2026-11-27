---
status: accepted
date: 2026-11-27
---

# The research scratchpad stays first

Decided in 1.33.0. Records a measurement behind `call.schema.research_schema`; changes no code.

## Context and problem statement

`research_schema` injects a free-text `analysis` property first and requires every property, so a
model writes its reasoning before the fields it fills from it. Grammar-constrained decoding writes
properties in schema order, so a field placed before the reasoning is committed before any
reasoning exists. Whether the position matters for a given task is an empirical question, and one
consumer measured it on a structured extraction from images.

## Decision drivers

- The grammar's shape is a measured choice, not a default copied from elsewhere.
- A result that bounds a claim is recorded as a bound, not as a reversal.
- The reasoning text is the only record of why a reply came out as it did.

## Considered options

1. Scratchpad first (the current grammar).
2. Scratchpad last.
3. No scratchpad.

## Decision outcome

Option 1, unchanged.

### Evidence

Three arms over the same image set and the same hard checks: error rates 27.6% (first), 23.3%
(last) and 23.1% (absent), p = 0.43. No arm differed. Nearly every error was perceptual: the model
misread what was drawn, and no amount of reasoning text recovers a value the model did not see
(`docs/guides/slots.md`, "When the input is an image").

This is a bound, not a contradiction of the case for reasoning first: where errors are reasoning
errors, the position can matter; where they are perception errors, it cannot. The scratchpad stays
first for auditability. Reading it is how two rendering defects in that image set were diagnosed.

### Consequences

- No change to `research_schema` or `research`.
- A consumer whose errors are perceptual buys nothing by moving or removing the scratchpad, and
  loses the record of why a reply was wrong.
