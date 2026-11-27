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
`net_flow -19813893517` gets "-19.8B of flow". Both are the input, shown the way a person would
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
- **The marker sits on the figure's own line**, after at most one space or hyphen
  ("75 percent", "a 20-billion gap"), and ends before anything but a letter or a digit: the
  underscore of markdown italics (`_19.8B_`) ends one as a full stop does.

`loose` also licenses the scale or percent word the rendering needed ("billion" in
"19.8 billion") and no other number word. It asks that of a run that equals an input too: beside
a date's `20`, "-20 billion" is still 19813893517 shown coarser, and its "billion" is licensed.

**An input shown at a scale is that value at that scale.** Beside an input "-1.2B", "-1.2B" and
"-1.2 billion" both pass; beside "19.8B", so do "19.8 billion" and "about 20B", by the rounding
rule above (a whole "1B" is not "0.6B").

**A scale is a second claim, and an equal run does not make it.** When no value re-renders the
figure at its scale and no input shows it at that scale, a run the inputs license as it stands
(an equal run, or a plain rounding) does not carry the scale. Beside a date's `20`, "20B" is an
invented magnitude and is named as `20B`; "20 billion" is named as `billion`, even when another
figure in the inputs carries the word; and a word the number-word reader cannot read on its own
is named with the run (`20million`, `20 billion` for "20 billion_"). A percentage the value
already was is not a scale and keeps its word, whatever the value's size ("18 percent" beside
`18`, "18.4 percent" beside `18.37`, "0.24 percent" beside `0.2409`).

**A digit run with two points is not a number.** `1.2.3` is a version and `2026.11.20` a date:
each is licensed only as printed, what follows it marks nothing, and its trailing zero is not a
decimal to drop, so an input `1.2.0` licenses neither "1.2" nor "120%". `canon` reads such a run
as written for every reader, the belts included.

**The law is stated once, as a test.** `tests/grounding/test_number_law.py` holds `law`: the
clauses above, one a line, every input tried, over figures whose parts are known, so it parses
nothing it judges. `loose(rerender=True)` must name exactly what `law` names, on a row for each
sentence of this record and on generated replies (figures made from the prompt: one run kept
under another marker, one value rounded, shown at a scale, moved and nudged a step, or summed
with another). The same test counts how many generated figures each clause decided and fails
when one falls under a floor, so the comparison cannot go quiet where the law is. Four more
properties hold there: a
reply that quotes its prompt is licensed; more inputs never name more; a bare whole number is
licensed only by an input that printed it; and with `rerender` off the reading is exact. A change
to this law changes `law` first.

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
- Bad: a scale the inputs spell another way is refused. "1M" for 999,999 fails (the run is not
  the value at that scale to the decimals written), and so does "19.8MM".
- Bad: a scale the inputs carry in a LABEL is refused. Beside `{"net_flow_bn": -19.8}` or
  "volume (millions): 20", the replies "-19.8bn" and "20 million" are named (`19.8bn`,
  `million`): the law reads a scale after the figure, never in a key or a heading, and so does
  not know these from "20B" beside a date's 20. Exact comparison passed them. A prompt that
  prints the unit on the value ("-19.8bn") is read; measure a role with labelled units in shadow.
- Bad: any one input is a magnitude, a year or a price level included. Beside a date `2026-11-20`,
  "about 2K units" is 2026 shown in thousands and passes, as "2 thousand" does. It is the
  rounding rule working on a value the law cannot know is a date.
- Bad: the scale spellings are a closed table. `$20m`, `20 bn`, `20MM`, `20 mln` and
  `20 billions` are read as no scale at all, so beside a date's `20` each passes as the run it
  is. Exact comparison passed them too.
- Bad, and older than this decision: the law reads digit runs, so it refuses honest figures a
  reader would accept, and each is a re-ask where the role gates. A clock without its leading
  zero ("9:30" beside "09:30") names `9`. A numbered list ("1. ... 2. ...") names its markers. A
  value quoted from a compact JSON array (`[540,545,550]` is ONE run, by the thousands-separator
  rule) names `545`. A figure quoted from the system prompt is not in the prompt the grader
  reads. Measure a role in the consumer's shadow before it gates on this grader.
- Bad, and older than this decision: a spelled number inside underscore italics is not read.
  Beside a date's `20`, "20 _billion_" and "_twenty billion_" pass: the number-word reader ends a
  word at a word boundary, and an underscore is a word character. The marker after a digit run
  is read through italics (`_19.8B_`); the spelled word is not yet.
- Bad, and the other way: only the reply's prose is read. A number in a numeric field of a
  structured reply (`{"level": 999.5}`) is never checked; a role whose figures live in typed
  fields needs a grader over those fields, and this one is no gate for them.

### How much a dense prompt licenses

The law asks whether SOME input renders as the figure, so the more values a prompt carries, the
more made-up figures land on one. Measured on synthetic prompts of 150 values, 40,000 made-up
figures per shape, the share that passes:

| a made-up figure | 50 probabilities, 50 values in 1 to 100, 50 price levels | 150 values in 0 to 800 |
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
