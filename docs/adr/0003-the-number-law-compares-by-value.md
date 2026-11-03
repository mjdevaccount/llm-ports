---
status: accepted
date: 2026-11-03
---

# One number law, comparing figures by value

Decided in 0.3.0 (`canon`) and 0.4.0 (`loose`); its English statement joined it in 0.12.0
(`law_text`). Recorded in 0.12.0.

## Context and problem statement

Several readers ask the same question, "did the model write a number no input rendered?": the
grounding belt, the slots belt, a per-call grader, a span-citation check, and a model judge's
rubric. Each grew its own reading of a number, and they disagreed:

- An exact-string compare failed honest replies that wrote "2.20" against a rendered 2.2: 5 of 40
  failing replies on one production writer.
- Under a digit-free grammar, models spelled the figure ("nine one point seven million") or, with
  the digit token blocked, emitted the rest of the word they were reaching for ("ighteen-year
  high"). A digits-only grader passed what the writers' belts refused.
- A model judge, given a prose restatement of the law, applied a different law from the code.

## Decision drivers

- A rule and a reader may disagree about judgement, never about what a number is.
- A failure must name the figure, so a retry or a person can act on it.

## Considered options

1. Each reader keeps its own regex and comparison.
2. One function returns a count; readers threshold it.
3. One module owns the reading of a number, returns the figures by name, and states the law in
   English for a model judge.

## Decision outcome

Chosen option 3, in `numbers`:

- `canon` compares digit runs by value: separators and trailing decimal zeros are dropped.
- `loose(text, allowed, words_ok)` returns every unlicensed figure in order, digit runs then
  spelled numbers and number-word tails. A `%` in the inputs licenses "percent".
- `grounding.belt`, `slots.belt`, `numbers.uncited` and `call_graders.numbers_from_inputs` all read
  through `loose`.
- `law_text(mode, *, web)` is the law's one English statement; `judge_sampling.rubric` composes it.
  Its output is pinned byte-for-byte by a golden test, because the rubric is prompt text and a
  change to it moves every verdict.

### Consequences

- Good: every reader agrees by construction; changing the law is one change.
- Good: failures are named, so a single retry message says exactly which figure to fix.
- Bad: the number-word list matches word tails such as "ills"; it is tuned for model output
  under a digit-free grammar and can flag ordinary English in unconstrained text.

## More information

A value the output grammar itself asks for (a score, a confidence number, an enum label) is the
model's answer, not a figure taken from the input. Before the law said so first, a judge failed
848 of 854 sampled replies, reading the grammar's own fields as numbers the input "did not carry";
with the clause appended last, a replay still failed `conviction: 4`, so it leads the law.
