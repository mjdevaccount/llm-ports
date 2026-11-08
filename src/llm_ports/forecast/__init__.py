"""Forecasts: grade what a voice said would happen, calibrate it, combine voices.

A voice is anything that asserts a future event: a model role, a human author transcribed by a
model, an assembler. Each assertion is a `Forecast`; once the world settles it, a `Resolved`.
The seams speak standard protocols, so outside models drop in unwrapped: calibrators and
`HedgeMap` are scikit-learn estimators (`fit(X, y)`, `predict_proba(X)`, `get_params`), combiners
are River classifiers (`learn_one(x, y)`, `predict_proba_one(x)`), scores take scoringrules'
names and order (`brier_score(obs, fct)`), and the values take forecasting-platform, schema.org
and W3C PROV field names (`_types` maps them).
This package holds the parts that do not depend on who the voices are:

- `score`: proper scores (Brier, log score, skill), the reliability curve and its error.
- `skill`: `brier_skill`, skill over a per-event REFERENCE forecast (a base rate, a prior),
  with a cluster-bootstrap interval.
- `calibrate`: the `Calibrator` port and `PiecewiseLinear`, a fitted map applied without a
  fitting library (fitters: `adapters.sklearn`, `adapters.venn_abers`).
- `platt`: `PlattCalibrator`, a two-parameter logistic map fitted and applied with the
  standard library; it refuses a slope that inverts the voice's ranking.
- `hedge`: `HedgeMap`, the probability each source's verbal confidence has earned.
- `combine`: the `Combiner` port, `LogitPool` (the fixed baseline), `PriorPool` (shrink toward
  and fall back to a prior voice, extremize), `BestVoice` (the best-single-voice baseline) and
  the log-odds transform (a learned combiner: `adapters.river`).
- `walk`: `Event` and `walk_forward`, a combiner replayed in time order that learns an outcome
  only once it was known; `compare` scores several on the same events (tuning:
  `adapters.optuna`).

Invariants:
    - Standard library only: imports nothing else from `llm_ports` and no third-party package
      (enforced by an import-linter contract), like `llm_ports.learn`.

Rationale: ADR-0013.
"""

from llm_ports.forecast import calibrate, combine, hedge, platt, score, skill, walk
from llm_ports.forecast._estimator import Params
from llm_ports.forecast._types import Forecast, Resolution, Resolved, finite_real, valid_p
from llm_ports.forecast.calibrate import Calibrator, Insufficient, PiecewiseLinear
from llm_ports.forecast.combine import BestVoice, Combiner, LogitPool, PriorPool
from llm_ports.forecast.hedge import HedgeMap, HedgeRow
from llm_ports.forecast.platt import Platt, PlattCalibrator
from llm_ports.forecast.score import Bin, Decomposition, Score
from llm_ports.forecast.skill import Skill, brier_skill
from llm_ports.forecast.walk import Event, walk_forward

__all__ = [
    "BestVoice",
    "Bin",
    "Calibrator",
    "Combiner",
    "Decomposition",
    "Event",
    "Forecast",
    "HedgeMap",
    "HedgeRow",
    "Insufficient",
    "LogitPool",
    "Params",
    "PiecewiseLinear",
    "Platt",
    "PlattCalibrator",
    "PriorPool",
    "Resolution",
    "Resolved",
    "Score",
    "Skill",
    "brier_skill",
    "calibrate",
    "combine",
    "finite_real",
    "hedge",
    "platt",
    "score",
    "skill",
    "valid_p",
    "walk",
    "walk_forward",
]
