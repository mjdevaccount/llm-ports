# Forecasts: grade, calibrate, combine

Anything that says what the market, a metric or the world will do is a **voice**: a model role,
a human author a model transcribed, a rule-based assembler. `llm_ports.forecast` turns what each
voice said into a track record, fixes its confidence, and pools voices, with no human labels:
the world is the label.

The loop has four steps:

| step | what | where |
|---|---|---|
| record | `Forecast`: producer, subject, kind, stated `p` and/or `hedge`, evidence span | core |
| score | `score()`: Brier, log loss, skill over the base rate; `reliability()` and `ece()` | core |
| calibrate | `Calibrator.fit` returns a map (`PiecewiseLinear`) or `Insufficient` | core port; fitters in `adapters.sklearn` (isotonic) and `adapters.venn_abers` (small n) |
| combine | `Combiner`: `LogitPool` is the fixed baseline | core; `adapters.river.RiverLogitPool` learns online |

Verbal confidence is learned per source: `HedgeMap` gives "likely" from each author the
probability it has earned, shrunk toward the word's pooled rate when that author has few resolved
uses of it. Ranges and distributions score through `adapters.scoringrules` (interval score, CRPS).

## The whole loop

```python
--8<-- "examples/forecasts.py"
```

## Choosing a calibrator

- **Isotonic** (`IsotonicCalibrator`, extra `[sklearn]`): a monotone map, from about 30 resolved
  forecasts up (`MIN_PAIRS`); the result is a `PiecewiseLinear`, stored as JSON and applied
  without scikit-learn.
- **Venn-ABERS** (`VennAbersCalibrator`, extra `[venn-abers]`): valid at any sample size; it
  returns a bracket `(p0, p1)` whose width says how much a short history can say. Use it for a
  voice with tens of resolved forecasts, isotonic once it has hundreds.

Both return `Insufficient` (never raise) for too few pairs or a history where every outcome was
the same.

## Combining earns weight

A learned combiner reaches a decision the same way any learning loop does: logged in shadow
against the equal-weight `LogitPool` and the best single voice, and promoted only through a gate
(`llm_ports.learn.gate`). `RiverLogitPool` returns None until it has heard `warmup` outcomes, so
a caller falls back to its baseline instead of reading an untrained model.

Rationale: [ADR-0013](../adr/0013-forecast-scoring-in-a-standalone-core.md).
