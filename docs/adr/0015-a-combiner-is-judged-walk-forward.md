---
status: accepted
date: 2026-11-08
---

# A combiner is replayed walk-forward, shrunk toward a prior, and tuned only on the past

Decided in 1.13.0.

## Context and problem statement

1.11.0 gave `forecast` a `Combiner` port, the fixed `LogitPool` and River's online logit pool.
A consumer that wants to log a learned combiner in shadow still had to write four things itself:
the replay that shows what the combiner would have said at each moment, the baselines it is
judged against (equal weight and the best single voice), a way to start from a prior forecast
(a base rate for the subject) instead of 0.5, and a search over the pool's learning rate,
extremizing factor and shrinkage. Each of these leaks the future if it is done carelessly: a
replay that learns an outcome at forecast time rather than resolve time, or a search scored on
the same events it reports.

## Decision drivers

- An outcome becomes known at its resolve time; a replay that learns it earlier is a backtest
  with look-ahead.
- A tuned number reported on the events it was tuned on is not out of sample.
- The combiner should answer something useful before it has learned anything.
- The core stays standard library only; the search library is an extra.
- The two baselines must be scored on exactly the same events as the combiner.

## Considered options

1. Leave replay, baselines and tuning to each consumer.
2. Core (`forecast.walk`, `forecast.combine`): `Event` (Metaculus field names: `forecast_time`,
   `resolve_time`, `resolution`), `walk_forward` (predict in forecast-time order; learn each
   outcome only at its resolve time), `holdout_start` and `compare` (every combiner scored on
   the same resolved, answered events); `PriorPool` (shrink toward a prior voice in log-odds,
   extremize, fall back to the prior while the inner combiner abstains); `BestVoice` (the
   best-single-voice baseline, online). Adapter (`adapters.optuna`, extra `[optuna]`): `tune`,
   a seeded TPE study over the train split whose objective is the walk-forward Brier score, the
   best parameters then replayed over every event and scored on the holdout only.
3. A grid search in the core.

## Decision outcome

Option 2. Option 3 is the "grid in code" the search library replaces, and it has no storage a
study viewer can read. Optuna was chosen over a hand-rolled search because its studies persist
to any SQLAlchemy database (`storage`) and `optuna-dashboard` reads them with no code.

`PriorPool` puts the prior in two places on purpose: as an ordinary voice the inner combiner can
weigh, and as the log-odds the result is shrunk toward. The tuner then chooses how much of the
prior to keep (`shrink`) separately from how sharp to be (`alpha`); multiplying the voices'
deviation from the prior by both would have made the two parameters one.

An abstention is scored as 0.5 in the objective, so a configuration cannot improve its score by
answering less; out of sample the holdout `Score` counts only answered events and says how many.

### Consequences

- Good: the same `walk_forward` call that judges a combiner produces its shadow forecasts for the
  open events, so the logged forecast and the scored one cannot drift apart.
- Good: a study persists in the consumer's database, and a viewer reads it without code.
- Bad: each trial replays the whole train history, so cost grows with trials times events; at
  tens of thousands of events and tens of trials a River pool still runs in seconds, a heavier
  combiner may not.
- Bad: `BestVoice` chooses by running mean Brier with a fixed `min_n`, which favours a voice that
  speaks only on easy events; it is a baseline to beat, not a combiner to ship.
