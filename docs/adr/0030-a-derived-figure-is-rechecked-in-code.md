---
status: accepted
date: 2026-11-26
---

# A derived figure is re-checked in code, and code rules only where it is authoritative

Decided in 1.29.0. Completes the neutral consequence of ADR-0029 ("re-checking a `derived`
claim's `op` in code is left to the consumer for now"); changes nothing it decided.

## Context and problem statement

ADR-0029 labels a claim `derived` when it follows from cited facts by a stated operation, and
carries that operation as text (`Claim.op`). A judge measured on a claim-level gold set refused
most derived figures (a distance as a percentage, a square-root move) or recomputed them wrong. A
language model is a weak calculator, and a judge asked to verify arithmetic is asked to do the one
thing it does worst. The arithmetic is mechanical: once the judge names the facts and the
operation, code can recompute it exactly.

Two questions follow. What grammar may code evaluate, given the op is model-written text? And
where may a code result overrule the judge, given an op certifies a figure and a claim is more
than its figure?

## Decision drivers

- Arithmetic is decided by code, never by a judge, wherever the judge has named the operation.
- Model-written text is never executed: the evaluator accepts a closed grammar and nothing else.
- A code ruling is never wrong in the direction it rules. Where its evidence covers only part of
  a claim, it does not rule on the rest.
- A figure shown coarser than the computed value is the same figure (ADR-0027); a figure close to
  but outside its tolerance is neither confirmed nor refuted.

## Considered options

1. Leave the recheck to each consumer (the status quo of ADR-0029).
2. Evaluate op text with a sandboxed expression library.
3. A closed `ast` grammar in `grounding.derive`, with `evals.settle` ruling on labelled claims.

## Decision outcome

Option 3.

- `grounding.derive.recheck(op, shown, tol=, facts=, band=)` reads `expr = numeric = result`.
  It evaluates the numeric part in a closed grammar (numbers, names, `+ - * /`, unary `-` and `+`,
  `abs`, `sqrt`, a chain of `< <= > >=`), holds it to the written result at that result's printed
  decimals, and, given the facts, evaluates the expression over them so a wrong substitution
  breaks. A bare comparison states its own result; text after a colon is a comment.
- The shown figure holds when it is the value, a rendering the number law licenses
  (`numbers.rerenders`: rounded, scaled, a percentage), or within the claim's tolerance. Outside
  the tolerance but within `band` times it (default 2) is `abstain`: a wrong tolerance and a wrong
  figure look alike there. A percent figure is judged on the reading nearer the value in ratio
  ("0.30%" beside 0.097 is 0.30, never 0.003).
- Anything else is `unchecked`. Code never guesses.
- `evals.settle.settle(claim, ...)` rules on a labelled claim: `breaks` makes it `contradicted`
  whatever the label said; `holds` confirms a grounded label and never overturns an ungrounded one
  (a true figure can sit beside a flipped direction word or an invented cause). The figures the
  number law refuses ride along as `refused`, a hint for the judge's prompt, never a ruling.

### Evidence

Measured against three gold families before release (counts are claims):

| family | ops rechecked | recheck agrees with gold | settle rulings | wrong rulings |
|---|---|---|---|---|
| synthetic, by construction (321 cases) | 1,876 | 1,876 | 1,876 | 0 |
| synthetic, historical inputs (3,399 cases) | 18,391 | 18,391 | 18,391 | 0 |
| hand-labelled replies (41 cases, 286 claims) | 13 | 7 hold, 6 prose ops unchecked | 4 | 0 |

Every planted figure shift (332 claims across the synthetic families) breaks; every figure shown
one decimal coarser (153) holds. A first draft let the number law rule too: a claim with no op
whose figures no input licenses was ruled ungrounded. That was right on 802 of 802 synthetic
claims and wrong on 5 of 5 hand-labelled ones (a forecast the prompt asked for, a date rendered as
"Nov 6", a decimal comma, a window length). Free text renders figures in ways the law does not
read, so a refusal became a hint.

Published method: program-aided reasoning hands arithmetic to an interpreter rather than the model
(PAL, Gao et al. 2022, arXiv:2211.10435; Program of Thoughts, Chen et al. 2022, arXiv:2211.12588).
Fact-checking benchmarks report small checker models weakest on arithmetic claims (MiniCheck,
Tang et al. 2024, arXiv:2404.10774).

Checked before writing: `simpleeval` evaluates a wider language than an op chain needs (power,
modulo, floor division, equality, membership, conditionals, string operations), and each would
have to be switched off one by one; a grammar that accepts more than the derivation language
grades text it should refuse. `asteval` is a near-complete Python interpreter. Python's own `ast`
with a node whitelist is the smallest evaluator that accepts exactly the grammar, with no new
dependency. `numbers.rerenders` (ADR-0027) is reused for the rounding licence.

### Consequences

- Good: a judge's arithmetic mistakes are overruled by code, and a planted wrong figure is caught
  whether or not the judge sees it.
- Good: a judge's prompt can carry the refused figures as named hints rather than a whole input.
- Bad: an op written as prose (a date step, `|x|` notation, a regime mapping) is `unchecked`; a
  consumer that wants it checked writes the op in the grammar.
- Neutral: date arithmetic is not in the grammar. A date op is `unchecked` until a date step is
  added as its own node kind.
