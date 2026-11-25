---
status: accepted
date: 2026-11-25
---

# A reply may round one input value

Decided in 1.27.0. Extends ADR-0003; the law it states for slot-filled prose is unchanged.

## Context and problem statement

ADR-0003 licenses a figure in model output when an input rendered the same value. That is the
right law where code prints every figure the model may use: a slot belt, a cited span. It is the
wrong law for `call.graders.numbers_from_inputs`, which reads free prose written over raw inputs.
A prompt that carries `realized_r 0.5918` gets "realized 0.59R" back; one that carries
`net_gex -19813893517` gets "-19.8B of gamma". Both are the input, shown the way a person would
show it, and both fail.

Read against two weeks of replies on three roles that write a paragraph over a table of floats,
the grader failed between two thirds and all of them, and most failures were a rounding. The
score had stopped separating a reply that invented a figure ("wait for a 15-minute retest", no 15
anywhere in the input) from one that did not, so nothing could act on it.

## Decision drivers

- A figure that is one input value, shown coarser, is not a number the model made up.
- A figure computed from two inputs still is: a model's arithmetic is the thing the law exists
  to keep out, and it is wrong often enough to matter.
- An invented small integer must stay caught. Every small integer is the rounding of some
  fraction, so a rule that licenses "15" because an input reads 14.878 licenses most inventions.
- A belt over values code printed keeps comparing them as printed.
- The law stays in one module.

## Considered options

1. A relative tolerance: license a run within some percent of an input value.
2. Tell every prompt to copy figures digit for digit, and keep the law exact.
3. License the renderings of ONE value: rounding to the decimals written, a scale, a percentage.

## Decision outcome

Chosen option 3, as `numbers.rerenders(run, after, allowed)` and `numbers.loose(..., rerender=True)`.

A digit run is licensed when some allowed value renders as it:

- **Rounding.** The value rounded, half up or half even, to the decimals the run is written with,
  when it has at least one: "0.59" for 0.5918, "14.88" for 14.878. The run's trailing zeros are
  its precision, so "28.00" is not 28.89.
- **A percentage.** A run followed by `%`, "percent" or "pct" may show a value that has a
  fractional part, times 100 ("75%" for 0.7486), or the value itself rounded ("13.5%" for 13.49).
- **A scale.** A run followed directly by `K`, `k`, `M`, `B`, `T` or `bn`, or by "thousand",
  "million", "billion" or "trillion", may show the value divided by it: "19.8B" for 19813893517.

Five limits keep the invented figure caught:

- **A whole number with no marker is never a rounding.** "15" is licensed only by an input 15.
- **A marked whole number rounds a magnitude of 1 or more.** "0%" is not 0.004.
- **A whole value is never moved to a percentage.** "100%" is not a `1` in the inputs, and
  "200%" is not a `2`: a step number or a day count would license either. The cost is that
  "100%" for a probability printed as `1.0` fails, and is re-asked where the role gates.
- **A scale letter is a capital, or `k`.** A lower-case `m`, `b` or `t` directly after a digit
  is a timeframe or a unit ("a 15m retest", "the 5m chart"), and `mm` is millimetres. A letter
  after a space is a unit too.
- **The marker sits on the figure's own line**, after at most one plain space.

`loose` also licenses the scale or percent word the rendering needed ("billion" in
"19.8 billion") and no other number word. It asks that of a run that equals an input too: beside
a date's `20`, "-20 billion" is still 19813893517 shown coarser, and its "billion" is licensed.

`rerender` is off by default. `numbers_from_inputs` turns it on; `grounding.belt`, `slots.belt`
and `uncited` do not, because the figures they check were printed by code and a model has no
reason to change one.

`law_text`, the same law stated in English for a model judge, is unchanged in this release: its
text is pinned by a golden rubric, and a judge reads the reply against the inputs with its own
judgement of what a rounding is.

### Consequences

- Good: the grader's failures are inventions and arithmetic, so a consumer can act on one
  (ADR-0028).
- Good: no prompt has to forbid rounding, which a reader of the prose would not thank it for.
- Bad: a wrong figure that happens to round another input is licensed. "0.6" passes when the
  inputs hold 0.5918 for a different quantity. The law never knew which quantity a figure
  belonged to; this widens that existing gap by the width of a rounding.
- Bad: a whole-number rounding ("48 minutes" for 48.1) still fails. The cost is a re-ask where the
  role gates on the grader; the alternative licenses the invented count.
- Neutral: a marked whole number is looser than an unmarked one. "5%" is licensed by an input in
  0.045 to 0.055 or in 4.5 to 5.5.
- Neutral: a capital scale letter is still read as a scale. "a 15M retest" is licensed by an
  input 15234567. Nothing on the page tells the two readings apart.

### How much a dense prompt licenses

The law asks whether SOME input renders as the figure, so the more values a prompt carries, the
more made-up figures land on one. Measured on synthetic prompts of 150 values, 40,000 made-up
figures per shape, the share that passes:

| a made-up figure | 50 probabilities, 50 values in 1 to 100, 50 strikes | 150 values in 0 to 800 |
|---|---|---|
| a whole percentage, "37%" | 63% | 16% |
| one decimal below 1, "0.4" | 98% | 1% |
| two decimals below 1, "0.41" | 40% | 0.3% |
| two decimals below 800, "512.37" | 0.2% | 0.2% |
| a bare whole number, "512" | 0.1% | 0% |

A gate on this grader keeps out an invented level, count or price. It does not keep out an
invented probability on a prompt that already carries fifty of them: such a role needs a slot
grammar (ADR-0004), where the model cites the value and never writes it.

### Option 1, rejected

A tolerance has no principled width, and any width wide enough for "19.8B" licenses a different
number that is merely close. It also licenses a bare "15" beside 14.878.

### Option 2, rejected

It was tried by instruction on one role and held for the figures the instruction named. A
9-billion-parameter model rounds anyway, and prose reading `-19813893517` is worse prose.
