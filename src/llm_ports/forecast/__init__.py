"""Forecasts: grade what a voice said would happen, calibrate it, combine voices.

A voice is anything that asserts a future event: a model role, a human author transcribed by a
model, an assembler. Each assertion is a `Forecast`; once the world settles it, a `Resolved`.
This package holds the parts that do not depend on who the voices are:

- `score`: proper scores (Brier, log loss, skill), the reliability curve and its error.
- `calibrate`: the `Calibrator` port and `PiecewiseLinear`, a fitted map applied without a
  fitting library (fitters: `adapters.sklearn`, `adapters.venn_abers`).
- `hedge`: `HedgeMap`, the probability each source's verbal confidence has earned.
- `combine`: the `Combiner` port, `LogitPool` (the fixed baseline) and the log-odds transform
  (a learned combiner: `adapters.river`).

Invariants:
    - Standard library only: imports nothing else from `llm_ports` and no third-party package
      (enforced by an import-linter contract), like `llm_ports.learn`.

Rationale: ADR-0013.
"""

from llm_ports.forecast import calibrate, combine, hedge, score
from llm_ports.forecast._types import Forecast, Resolved, finite_real, valid_p
from llm_ports.forecast.calibrate import Calibrated, Calibrator, Insufficient, PiecewiseLinear
from llm_ports.forecast.combine import Combiner, LogitPool
from llm_ports.forecast.hedge import HedgeMap, HedgeRow
from llm_ports.forecast.score import Bin, Score

__all__ = [
    "Bin",
    "Calibrated",
    "Calibrator",
    "Combiner",
    "Forecast",
    "HedgeMap",
    "HedgeRow",
    "Insufficient",
    "LogitPool",
    "PiecewiseLinear",
    "Resolved",
    "Score",
    "calibrate",
    "combine",
    "finite_real",
    "hedge",
    "score",
    "valid_p",
]
