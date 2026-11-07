---
status: accepted
date: 2026-11-07
---

# Forecast scoring lives in a standalone core; fitting libraries live in adapters

Decided in 1.11.0.

## Context and problem statement

A consumer records thousands of claims a voice made about the future (a range holding, a
direction, a level touched) and resolves them against what happened. It measured that one voice
states 0.68 for its ranges and is right 38% of the time, and that nothing scores, calibrates or
combines the other voices at all. The library had a citation type (`ports.Claim`) and a
learning-loop core (`learn`), but no forecast type, no proper scoring rule, no calibration map
and no combiner. Every consumer would write its own Brier score, its own isotonic wrapper and its
own way of reading "likely".

## Decision drivers

- The mechanics are generic: any voice that asserts a future event is scored the same way.
- Fitting needs the scientific stack (numpy, scipy, scikit-learn, pandas); a caller that only
  calls models must not install it.
- A fitted map must be applied where no fitting library is installed, and stored as data.
- The library's rule: a failure is a named outcome, never an exception across a port.

## Considered options

1. Leave it to each consumer.
2. A `forecast` package in the bottom layer, standard library only like `learn`: the `Forecast`
   value, binary proper scores (Brier and log loss are their definitions, a line each), the
   reliability curve, `Calibrator` and `Combiner` ports, `PiecewiseLinear` (a fitted monotone
   map as JSON-ready knots), `HedgeMap` (beta-binomial shrinkage by the method of moments) and
   `LogitPool` (the equal-weight, optionally extremized, baseline). Fitters and the non-binary
   scores in four adapters, one extra each: `scoringrules` (interval score, CRPS), `sklearn`
   (isotonic), `venn-abers` (small-n calibration with a validity guarantee), `river` (an online
   logistic logit pool).
3. Put everything in one adapter over scikit-learn and scoringrules.

## Decision outcome

Option 2. Option 3 makes applying a stored map and scoring a binary forecast depend on the
scientific stack. `forecast` imports nothing else from `llm_ports` (a `forecast-standalone`
contract, like `learn-standalone`), so it can be lifted into any codebase; a consumer wires it to
`learn` (a `Loop`, the gate, `bh_fdr`) itself. The four extras are grouped as `forecast` and left
out of `all`, which stays the set a model-calling caller needs. The stdlib Brier score is checked
against scoringrules' in the adapter tests.

### Consequences

- Good: a calibration map fitted offline is applied anywhere from its JSON.
- Good: each voice's hedge words get a learned probability without human labels.
- Bad: `HedgeMap`'s method-of-moments prior strength is crude at two or three sources; it falls
  back to a fixed strength below two sources and to near-full pooling when the sources agree
  within binomial noise. A hierarchical fit would be better and needs a heavier dependency.
- Bad: `VennAbersFit` holds the vendor model and is not serialisable; refit it from the history.
