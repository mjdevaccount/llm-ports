---
status: accepted
date: 2026-11-03
---

# Grounded prose by construction

Decided in 0.3.0 (`slots`). Recorded in 0.12.0.

## Context and problem statement

A writer role turns computed facts into prose: an order summary, a coaching note, a report. The
facts include figures the model must not change. Instructions do not hold: on a production
summarizer, a 9B-parameter self-hosted model given "use only the numbers you were given" did its
own arithmetic and invented a figure in 40 of 64 replies (about 60%) with the instruction in
place. A check after the fact (drop any sentence with an unlicensed number) throws away the reply
the reader needed.

## Decision drivers

- The invented-figure rate should be zero by construction, not reduced by prompting.
- The check that remains must be able to say which figure failed.
- The approach must work on self-hosted models through a JSON-schema grammar.

## Considered options

1. Instruct the model and check the output (drop or retry failing sentences).
2. Let the model write numbers, then compare each against the facts and repair.
3. Take the digit token away: the model cites each value by an id, and code prints the value.

## Decision outcome

Chosen option 3, in `slots`:

1. Code renders every value once, as text, under an id with no digits (`{total}`, `{count}`).
2. The output grammar admits no digit in a prose run (`prose_pattern`, `slot_pattern`), so the
   model can write `{total}` and cannot write "1,300".
3. Code fills each slot (`fill`) and runs the belt over the filled text (`belt`): every number must
   be a rendered value. The count is 0 by construction; a non-zero count is an alarm, never a
   tolerated rate.

`problems` reports what is wrong before filling (an unknown slot, a stray brace, a value cited
twice) in words a single retry can act on. With one or two ids, the grammar itself allows each
slot at most once: a model wrote the same slot for two different meanings, and its retry returned
the same line, so the rule had to live in the grammar.

### Consequences

- Good: zero invented figures by construction; the belt stays as a tripwire.
- Good: one implementation shared by every grounded writer instead of one weaker copy each.
- Bad: the model can still spell a number or emit the tail of a number word; the number law reads
  spelled numbers ([ADR-0003](0003-the-number-law-compares-by-value.md)) and the context it reads
  is digit-free ([ADR-0004](0004-digit-free-context.md)).
- Bad: the grammar is a regex the model server must compile; see
  [ADR-0005](0005-character-whitelist-grammar.md) for the constraints that shaped it.

## More information

Under the digit-free grammar the same summarizer produced 0 invented figures.
