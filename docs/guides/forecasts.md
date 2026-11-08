# Forecasts: grade, calibrate, combine

Anything that says what the market, a metric or the world will do is a **voice**: a model role,
a human author a model transcribed, a rule-based assembler. `llm_ports.forecast` turns what each
voice said into a track record, fixes its confidence, and pools voices, with no human labels:
the world is the label.

The loop has four steps:

| step | what | where |
|---|---|---|
| record | `Forecast`: author, subject, kind, stated `probability` and/or `hedge`, verbatim `text` | core |
| score | `score(obs, fct)`: Brier score, log score, skill over the base rate; `reliability()`, `ece()`, `mce()` and `decompose()` (Murphy: reliability, resolution, uncertainty) | core |
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

## Replaying in time order, against the baselines

`walk_forward(combiner, events)` replays `Event`s as the combiner would have run: each forecast
is made at its `forecast_time`, and an outcome is learned only once its `resolve_time` has
passed. `compare` runs several combiners over the same events and scores them on the same
resolved, answered subset, so a learned pool is always read beside the equal-weight
`LogitPool` and `BestVoice`. `PriorPool` starts every combiner from a prior voice (a base rate for
the subject) instead of 0.5:

```python
import random
from datetime import datetime, timedelta

from llm_ports.forecast import BestVoice, Event, LogitPool, PriorPool
from llm_ports.forecast.walk import compare, holdout_start

rng = random.Random(2)
t0 = datetime(2026, 1, 1)
events = []
for h in range(600):
    y = rng.random() < 0.55
    x = {"sharp": 0.75 if y == (rng.random() < 0.8) else 0.25, "noise": rng.random(), "base": 0.55}
    t = t0 + timedelta(hours=h)
    events.append(Event(x=x, forecast_time=t, resolve_time=t + timedelta(hours=3), resolution=y))

scores = compare(
    {
        "equal": PriorPool(inner=LogitPool(), prior_key="base"),
        "best": PriorPool(inner=BestVoice(min_n=20), prior_key="base"),
        "prior": PriorPool(inner=BestVoice(min_n=10**9), prior_key="base"),
    },
    events,
    since=holdout_start(events, 0.2),
)
assert scores["best"].brier_score < scores["prior"].brier_score
```

`adapters.optuna.tune(build, events, space, study_name=..., storage=...)` searches a combiner's
parameters (a River pool's learning rate, `alpha`, `shrink`) on the train split only and reports
the best parameters' holdout `Score`; the study persists to any Optuna storage, where
`optuna-dashboard` reads it.

Behind a transaction-mode pooler (PgBouncer), Optuna's RDB storage cannot be kept in one schema:
it creates its tables on the connection's search path and Alembic checks them in the default
schema. `adapters.optuna.journal_storage(engine, "schema.table")` is the storage for that case:
Optuna's `JournalStorage` over `SqlJournal`, one append-only `(n bigint primary key, log text)`
table that the caller's migration creates and every statement names in full.

Rationale: [ADR-0013](../adr/0013-forecast-scoring-in-a-standalone-core.md),
[ADR-0015](../adr/0015-a-combiner-is-judged-walk-forward.md).
