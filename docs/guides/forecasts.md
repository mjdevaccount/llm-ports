# Forecasts: grade, calibrate, combine

Anything that says what the market, a metric or the world will do is a **voice**: a model role,
a human author a model transcribed, a rule-based assembler. `llm_ports.forecast` turns what each
voice said into a track record, fixes its confidence, and pools voices, with no human labels:
the world is the label.

The loop has four steps:

| step | what | where |
|---|---|---|
| record | `Forecast`: author, subject, kind, stated `probability` and/or `hedge`, verbatim `text` | core |
| score | `score(obs, fct)`: Brier score, log score, skill over the base rate; `reliability()` and `ece()` | core |
| calibrate | `Calibrator`: `fit(X, y)`, then `predict_proba(X)`; a refused fit records `Insufficient` | core port; fitters in `adapters.sklearn` (isotonic) and `adapters.venn_abers` (small n) |
| combine | `Combiner`: `predict_proba_one(x)`, `learn_one(x, y)`; `LogitPool` is the fixed baseline | core; `adapters.river.RiverLogitPool` learns online |

Verbal confidence is learned per source: `HedgeMap` gives "likely" from each author the
probability it has earned, shrunk toward the word's pooled rate when that author has few resolved
uses of it. Ranges and distributions score through `adapters.scoringrules` (interval score, CRPS).

## Standard names at every seam

Each seam speaks a protocol an existing library already defines, so an outside model drops in
without a wrapper and a reader who knows the library reads the code unchanged:

| seam | protocol | so this drops in |
|---|---|---|
| `Calibrator`, `HedgeMap` | scikit-learn estimator: `fit(X, y)` returns the estimator, `predict_proba(X)` gives `(P(no), P(yes))` rows, `get_params` / `set_params`, fitted state ends in `_` | a scikit-learn, venn-abers or MAPIE calibrator; `sklearn.base.clone` |
| `Combiner` | River classifier: `learn_one(x, y)`, `predict_proba_one(x)` with `x` as `voice -> probability` | any River binary classifier |
| `score` | scoringrules: `brier_score(obs, fct)`, `log_score(obs, fct)`, observation first, lower is better | scoringrules' vectorised scores in `adapters.scoringrules` |
| `Forecast`, `Resolved` | Metaculus / Good Judgment question fields, schema.org `Claim` / `ClaimReview`, W3C PROV | a platform export or a fact-check feed maps field for field (the table is in `llm_ports.forecast._types`) |

## The whole loop

```{.python title="examples/forecasts.py"}
--8<-- "examples/forecasts.py"
```

## Choosing a calibrator

- **Isotonic** (`IsotonicCalibrator`, extra `[sklearn]`): a monotone map, from about 30 resolved
  forecasts up (`MIN_PAIRS`); the fitted map is a `PiecewiseLinear` in `map_`, stored as JSON and
  applied without scikit-learn.
- **Venn-ABERS** (`VennAbersCalibrator`, extra `[venn-abers]`): valid at any sample size; it
  returns a bracket `(p0, p1)` whose width says how much a short history can say. Use it for a
  voice with tens of resolved forecasts, isotonic once it has hundreds.

Neither raises: for too few pairs or a history where every outcome was the same, `fit` records
`Insufficient` in `insufficient_` and `predict_proba` returns None.

## Combining earns weight

A learned combiner reaches a decision the same way any learning loop does: logged in shadow
against the equal-weight `LogitPool` and the best single voice, and promoted only through a gate
(`llm_ports.learn.gate`). `RiverLogitPool` returns `{}` until it has heard `warmup` outcomes, so
a caller falls back to its baseline instead of reading an untrained model.

Rationale: [ADR-0013](../adr/0013-forecast-scoring-in-a-standalone-core.md).
