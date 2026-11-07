---
status: accepted
date: 2026-11-07
---

# A judge earns trust from human labels, per judge model

Decided in 1.10.0.

## Context and problem statement

A sampled production judge decides things: a fail share pages, a verdict gates a prompt. A
small local judge measured on 20 hand-graded live replies across 11 roles failed 11 of the 14
replies a human graded good (Cohen's kappa about 0.14) while catching every seeded-bad copy: a
judge that fails nearly everything, whose alert meant nothing. The library already measured both
directions (`examples.agreement`, `calibration.verdict`), but nothing turned them into a decision,
so each consumer wrote its own trust rule over its own file, and the human labels growing in the
label store were read by no judge. A judge label also carried no record of which model gave it,
so a new judge model would have been scored on its predecessor's labels.

## Decision drivers

- One trust rule, shared by every judge; the consumer supplies data and thresholds only.
- A statistic on 20 items is noisy: the rule must carry its interval, and a caller must be able
  to require the lower end, not only the point.
- Swapping the judge model must never inherit trust.

## Considered options

1. Leave trust to each consumer.
2. `calibration.trust(pairs, replay=...)` returning a `Trust` with the numbers, fed by
   `examples.pairs(..., by=model)`; the trace port's `score` carries `metadata` so a judge label
   records its model in the store's existing `by` field.
3. Write judge labels through the `ExampleStore` instead of the trace handle (it already stores
   `by`), which needs the observation id the trace port does not expose.

## Decision outcome

Option 2. `trust` checks, first failure deciding: at least `min_labelled` pairs (default 20), the
seeded-bad replay `CALIBRATED` (required by default), and kappa at least `min_kappa` (default
0.6, Landis-Koch "substantial") read as the point or, with `bound="low"`, the 95% interval's lower
end. The interval is the Fleiss-Cohen-Everitt large-sample one (`stats.kappa_ci`), checked against
statsmodels to 1e-5; statsmodels itself is not a dependency because `learn` is standard library
only. `score(..., metadata=...)` is an optional argument, so a handle written before it still
satisfies the port as long as no caller passes metadata to it.

### Consequences

- Good: a consumer's trust module shrinks to loading its data; live human labels reach the judge.
- Good: a judge with no labels is untrusted by construction.
- Bad: judge labels written before 1.10.0 have no `by` and count for no model under `pairs(by=...)`.
