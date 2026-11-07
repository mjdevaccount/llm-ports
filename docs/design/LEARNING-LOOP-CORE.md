# The learning-loop core: one spine, one gate, one ledger

Status: DESIGN, 2026-11-07. Supersedes the scope of `EVAL-LABELS-CONSOLIDATION.md`, which stays
as the detail of ONE loop kind (LLM labels, §LLM labels below). Owner of the decision: this file.
Facts read from `origin/main` of each repo on 2026-11-07 (trade-data `6e941731`, trade-ui,
ci-kit, work-timeline) with `git show` / `git grep`; every load-bearing claim carries its
`file:line`. A claim not checked that way says UNVERIFIED.

## What was asked, in priority order

1. **Centralize the core in one good home**: portable, zero EDGE imports, storage behind a port,
   usable in an unrelated codebase.
2. **Make a new loop cheap**: this doc states the cost in files and lines; the first new consumer
   (a multi-symbol daily trend loop) measures it.
3. **Then fan out**: existing loops adopt it, judges and copilot read the shared store.

Hard limits: `trade-data/docs/CARD-SYSTEM-GATE.md` and `docs/PATTERN-GATE.md` are SEALED (never
edited; the core may only WRAP what they decide); pattern `strategy_weight` stays 0. trade-data's
existing learn engine is reused, never forked ("it must not end with two engines"). No fine-tuning
or training run; no Anthropic API (strong-model labelling = Gemini via the box gateway or llm-box
models).

## The shape

Every loop in this platform is the same five moves:

```text
  evidence in        labelled outcomes         fit / calibrate        gate             act + audit
  (series, events,  ──▶ (outcome resolved,  ──▶ (a model, a weight, ──▶ (evidence ──▶ ──▶ (promote at weight,
   replies, picks)       human / model /         a threshold, a           verdict)          show, page) + one
                         judge label)            calibration)                               ledger row per run
```

trade-data already names the stages `predict / resolve / score / consume / surface`
(`service/platform/loops.py:9-16`) and registers ~45 loops against them. The core does not
replace that vocabulary; it supplies the parts every loop re-wrote: the gate, the run record, the
labelled-example contract, the statistics.

### Where the shape occurs (and what each is missing)

| loop | evidence | outcome / label | fit | gate today | missing |
|---|---|---|---|---|---|
| picker signal weights | picker runs | `picker_outcomes` (honest fill, EOD) | `signals/weight_learning.py` IC-EWA | `signals/eval_metrics.py` `_blockers` (:526) | standard verdict row |
| prob calibration | p_up per pick | outcome | `signals/prob_calibration.py` Platt | `usecases/fit_prob_calibration.py` `earned()` | verdict lives in code, not ledger |
| hazard exit / hold duration | paths | MFE timing | `fit_hazard_exit.py`, `learn_hold_duration.py` | concordance > 0.53 / eff_n ≥ 30 | same |
| meta-label | features | take/skip outcome | `signals/meta_label.py` | identity under 40 samples | same |
| config / exit grids | replay | graded path | grid | `signals/promotion_gate.py` `evaluate_gate` (not wired live) | wiring |
| Pattern Lexicon | detectors | base rates | — | `PATTERN-GATE.md` (SEALED) | wrap only |
| Daily Card | card | session outcome | — | `CARD-SYSTEM-GATE.md` (SEALED) | wrap only |
| Setup Radar | setups | forward return | — | `setup_radar.py` `wilson_lb` (:284) | gate, ledger |
| event study | calendar events | forward return | `usecases/event_study.py` | `fdr_families` (:215) | ledger, gate |
| churn / news | articles | human-labelled fields (203) | — | `usecases/churn_eval.py` pass/fail | already the precedent |
| LLM judge trust | judged replies | human verdicts (~20) | — | `platform/llm_judge_trust.py` `assess` | labels (§LLM labels) |
| LLM roles (34) | replies | none | — | sampled 9B judge (kappa ~0.19) | labels, coverage |
| **trend / "what's popping"** (new) | `quant.ohlcv_daily_mv` + calendar | forward return | rank / logistic | — | everything: the proof consumer |

Gates in use today, three shapes and no shared type: `promotion_gate.GateVerdict(promote,
clauses, blocker)` (`signals/promotion_gate.py:69`), `domain/promotion.PromotionDecision`
(`domain/promotion.py:71`, a None clause fails closed), `llm_judge_trust.Trust(model, trusted,
reason, kappa…)` (`platform/llm_judge_trust.py:53`), plus `earned()` booleans in each fit usecase.
The Research Ledger (`signals/research_ledger.py`) then re-derives an earn state per run kind from
each kind's ad-hoc `metrics`, because no run states its verdict the same way.

## Decision 1 — the core's home: `llm_ports.learn`

A new subpackage of llm-ports, enforced standalone:

- **Why llm-ports and not a new repo.** llm-ports is already the estate's published, portable
  library: PyPI release by tag, strict mypy + `pyright --verifytypes` 100%, import-linter layers,
  an executed docs site, a `test-lowest` leg, and a core that installs with no vendor. A new repo
  would rebuild all of that for ~600 lines and add a second pin to every consumer: "should
  probably not have two". The pure statistics it needs (`cohen_kappa`, `gwet_ac1`) already live
  here (`evals/calibration.py:54,75`).
- **Why a standalone subpackage.** An import-linter `forbidden` contract: `llm_ports.learn` imports
  nothing from the rest of `llm_ports` and no vendor; it depends on the standard library only.
  Splitting it into its own distribution later is a file move, not a refactor. The name "llm" is
  the only cost; the trigger to split is a consumer that wants `learn` and objects to the name.
- **Its Protocols live in `learn/`, not `ports/`.** Deliberate exception to "core depends on a
  port in `ports/`": `ports/` carries LLM types, and the standalone contract forbids reaching it.
  ADR-0010 records this.
- **Layering.** `learn` sits in the bottom layer beside `ports`; `evals` (LLM labels, judge trust)
  builds on it. `evals.calibration.cohen_kappa` / `gwet_ac1` move to `learn.stats` and stay
  re-exported from `evals.calibration` (no consumer change).

## Decision 2 — the run ledger: `quant.learn_runs`, behind a port

| option | for | against | verdict |
|---|---|---|---|
| **extend `quant.learn_runs`** | exists (`migrations/076_learn_runs.sql`), ~20 kinds already write it (`learn_runs_repo.py:19-33`), `run_kind` is free text with no CHECK (:11), grants for app/copilot/grafana, `research_registry` FKs onto it, `latest_with_verdict` already reads `metrics->>'verdict'` (`learn_runs_repo.py:106`), the Research Ledger page and `/compiler/runs` render it | Postgres-only; a schema-specific adapter | **chosen** |
| MLflow tracking + registry | real UI, model registry | a NEW service (Postgres backend, backup, auth), a second registry beside `loops.py`, mig076 already rejected it for a single operator ("Deliberately NOT MLflow/W&B", :6); EDGE "models" are small param rows, not artifacts | rejected for EDGE; possible later optional adapter for portable users |
| Langfuse | already deployed; datasets, scores, annotation queues | LLM-shaped (traces, observations); an options fit is not a trace | rejected as the ledger; KEPT as the LLM label store (§LLM labels) |

Portability comes from the port, not the table: `learn.RunLedger` (`append`, `latest`) with three
implementations: `MemoryRunLedger` and `JsonlRunLedger` (stdlib, in the core, so the library works
anywhere with zero setup), and trade-data's `LearnRunsLedger` over `LearnRunsRepo` (adapter,
trade-data). An MLflow adapter is an optional extra when an outside user asks for it.

## Decision 3 — ONE gate: evidence in, verdict + audit row out

```text
Clause(name, value, threshold, op, passed: bool | None)      passed None = not evaluable
Verdict(gate, subject, decision, clauses, blocker, n)        frozen, JSON-able
decision ∈ pass | fail | insufficient | could_not_run
decide(gate, subject, clauses, *, floor=("n",)) -> Verdict   strict AND; first failing clause = blocker
```

- **Fails closed.** A clause that is None is not a pass: a None on a floor clause (sample size)
  gives `insufficient`, a None elsewhere gives `fail` with that clause as blocker. Same rule as
  `domain/promotion.py` and `promotion_gate._ok` (:78).
- **Vocabulary = the one already in the ledger.** `pass | fail | could_not_run` is what
  `churn_eval` writes (`usecases/churn_eval.py:150-161`) and what `quant.prompt_eval_run` CHECKs
  (mig344); `insufficient` is the Research Ledger's existing state. A recorded verdict sets
  `metrics.verdict = decision` and `metrics.gate = Verdict` in full, so every existing
  `latest_with_verdict` reader keeps working and the Research Ledger reads any gated kind
  generically (`pass`→earned, `fail`→not_earned, `insufficient`, `could_not_run`→error; a run
  with no verdict stays `informational`).
- **Audit = the ledger row.** `learn.record(ledger, kind, verdict, params, metrics)` appends one
  row. A gate is a pure function `evidence -> Verdict`; there is no base class to inherit.
- **Existing gates adopt by returning clauses**, thresholds unchanged: `promotion_gate`,
  `domain/promotion`, each fit's `earned()`, `eval_metrics._blockers`, `llm_judge_trust.assess`.
- **Sealed gates are wrapped, never changed.** The card and pattern gates keep their code and
  documents; an adapter in trade-data reads their DECIDED outputs and records them as `Verdict`s
  (`gate="card-system"`, `"pattern"`), so they appear on the same ledger and page. The pattern
  code's `bh_fdr` (`signals/pattern_stats.py:57`) is NOT replaced by the lifted one.

## Decision 4 — the training-data contract and its read surface

```text
Example(topic, subject, value, source, by, at, reason="", ref={})
  topic   what is being learned: "llm:narrative.research", "trend:1d", "picker:SPY"
  subject the thing labelled (an observation id, "SPY@2026-11-06", a pick id)
  value   the label: "good" | "bad" for verdicts, a number for an outcome
  source  human | model | judge | outcome      (outcome = resolved mechanically from the market)
  by      who: an operator id, a model id, a grader name
  ref     provenance: trace id, table + key, git sha
```

- **One contract, many physical stores, no copying.** Options outcomes stay in their tables
  (`quant.picker_outcomes` …); LLM verdict labels stay in Langfuse scores; the operator's trade
  truth stays in `quant.autopsy_calibration`. Each store is an `ExampleStore` adapter that READS
  its table as `Example`s; `learn.union(stores)` reads several as one. A copy would drift.
- **`ExampleStore`** (Protocol, never raises, named outcomes): `examples(topic, *, since, source)`;
  optional `put(example)` for stores that accept writes (Langfuse scores, a JSONL file).
- **Pure readers in the core:** `coverage(examples, topics, target)` (every topic appears, zero
  rows included: the gap is the point); `agreement(examples, a="judge", b="human")` (kappa + AC1
  on shared subjects); `few_shot(examples, k, balanced=True)` (deterministic, both classes).
- **The read surface for judges and copilot** ("all of this could be used by the judge"):
  `few_shot` hands an LLM judge graded examples of its own role; copilot's existing
  `get_similar_setups` (pgvector over decision reflections) stays the similarity search and gains
  an `ExampleStore` adapter, not a second index. Similarity search is NOT in the core (it needs a
  vector store); the core defines `SimilarExamples` as an optional Protocol only.

## Decision 5 — the engine: trade-data's learning plane, generic parts lifted

"We have a training engine already": the engine IS trade-data's learning plane, and stays there:
`FIT_REGISTRY` dispatch (`usecases/fit_registry.py`, `POST /admin/run/fit/{model}`), admin-run
declarations with `learn_kind` (`admin/admin_run_declarations_*.py`), Windmill schedules, the loop
registry with its `loop-stalled` watchdog, the ledger. Nothing of that moves; it gains the core's
types. What lifts is pure math that is duplicated today:

| function | copies on trade-data main | lifts to |
|---|---|---|
| Wilson interval | `domain/stats.py:17`, `signals/level_calibration.py:93`, `signals/setup_radar.py:284` (`wilson_lb`, z=1.28) | `learn.stats.wilson` |
| Benjamini-Hochberg | `signals/eval_metrics.py:272` (+ `pattern_stats.py:57`, sealed: stays) | `learn.stats.bh_fdr` |
| effective n | `eval_metrics.py:148,178`, `sample_weighting.py:121`, `calibrate_level_reactions.py:45` | `learn.stats.effective_n` (the variants that are one formula) |
| kappa / AC1 | llm-ports `evals/calibration.py` | `learn.stats` (re-exported) |
| reliability / ECE / Brier, Platt with slope refusal, purged CV | `signals/picker_calibration.py`, `prob_calibration.py`, `domain/stats.py:38`, `purged_cv` | lifted in the slice whose consumer needs them (the trend loop), not before |

A lift is byte-for-byte on numbers: trade-data's own tests run against the lifted function before
its copy is deleted. Fit usecases (they read tables) are NOT lifted.

## Decision 6 — ONE registry, ONE view of when training runs

- **The registry is `service/platform/loops.py`** and its generated `docs/LEARNING-LOOPS.md` +
  `GET /ops/loops` (`admin/ops_router.py:36`). Not forked. A loop built on the core declares its
  `LoopSpec` the same way; a helper derives its `score` slot (`quant.learn_runs` where
  `run_kind=<kind>`, precedent `loop_registrants_llm.py:53`).
- **"When does training run" is a JOIN of four records that already exist**, not a new store:
  admin-run declarations name the `learn_kind` each run stamps (12 already do,
  `admin_run_declarations_replay.py:30-165`); the Windmill tree holds the cron (next run);
  `ops.work` (written through `work-timeline`) holds each run's start, end and status;
  `quant.learn_runs` holds its result and verdict. One read, `GET /research/schedule`, joins them
  per kind: last run, its status, its verdict, next run, and a MISSED flag (a scheduled run with no
  `ops.work` row in its window, the rule `Slot.work_name` already encodes, `loops.py:107`).
- **The view is the existing Research Ledger page** (trade-ui `#/research`, `src/routes.tsx:462`),
  which already lists every kind with its earn state. It gains last / next / missed columns. No new
  page.

## Decision 7 — the ratchet: mapped, not adopted

The ratchet (ci-kit `ci_kit/ratchet.py`: `RatchetSet`, `RatchetCount`) IS a gate: findings in →
regression or not → the baseline file in git is the audit. It does not adopt the ledger: it runs
in CI, which cannot reach the box's Postgres, and its audit trail (the baseline's git history, with
a stated reason per re-freeze) is already append-only. If a ratchet ever reports into the view,
it does so by emitting a `Verdict` JSON as a CI artifact; nothing is built for it now. ci-kit stays
its home.

## The core's public surface (`llm_ports.learn`)

| module | holds |
|---|---|
| `learn` (types) | `Example`, `Clause`, `Verdict`, `Decision`, `Run`, `Outcome` types for the store calls |
| `learn.gate` | `clause(...)` helpers (`at_least`, `at_most`, `between`), `decide`, `record` |
| `learn.stats` | `wilson`, `bh_fdr`, `effective_n`, `cohen_kappa`, `gwet_ac1` |
| `learn.examples` | `coverage`, `agreement`, `few_shot`, `union` |
| `learn.ledger` | `RunLedger`, `ExampleStore`, `SimilarExamples` Protocols; `MemoryRunLedger`, `JsonlRunLedger`, `JsonlExampleStore` |
| `learn.loop` | `Loop` Protocol (`examples() -> fit() -> gate()`), `run(loop, ledger) -> Run`: exactly one ledger row per run, every exit included (a stage that fails records `could_not_run` with its reason; never raises) |

Every function is pure or behind a Protocol; no clock without injection; no I/O outside the JSONL
adapters. Docs: guide `docs/guides/learning-loops.md` with an executed, self-asserting example
(a loop over synthetic data from evidence to recorded verdict), API page, ADR-0010.

## The cost of a new loop (the adoption deliverable)

After slice 3 (below), a new loop in trade-data is:

| file | new / edit | about |
|---|---|---|
| `service/learn/<loop>.py` | new | the `Loop`: read evidence (a repo call), fit, return clauses: ≤ 120 lines |
| `tests/learn/test_<loop>.py` | new | pure tests of fit + clauses |
| `windmill/u/admin/<loop>.schedule.yaml` + `.flow` | new | the cron and the thin POST, copied |
| `service/usecases/fit_registry.py` | +1 line | registers it (dispatched by the existing `POST /admin/run/fit/{model}`) |
| `service/admin/admin_run_declarations_learn.py` | +1 line | `declare(..., learn_kind=<kind>)` |
| `service/usecases/loop_registrants.py` | +1 entry | the `LoopSpec` (score slot derived) |

**Zero migrations, zero routes, zero UI code**: the ledger is `learn_runs`, the route is the fit
route, the Research Ledger page lists any kind and its schedule. Budget: ≤ 7 files, ≤ 250 lines
including tests. The trend loop's PR reports its `git diff --stat` against this budget; a miss is
a core defect to fix, not a number to restate.

## Portability (acceptance criterion, proved in CI)

- import-linter contract `learn-standalone`: `llm_ports.learn` imports no other `llm_ports`
  module and no third-party package.
- `tests/contract/test_core_imports.py` gains a leg: a bare install imports `llm_ports.learn` and
  runs the guide's example with the JSONL ledger.
- The deployment-vocabulary pre-commit hook already rejects EDGE names in `src/llm_ports/`.
- Docs say how to plug in a ledger: implement two methods.

## Telemetry (what fails silently, where it lands, what watches it)

| silent failure | lands in | watched from outside by |
|---|---|---|
| a loop stops running | no `learn_runs` row / no `ops.work` row in window | `loop-stalled` watchdog (exists) via the derived score slot; MISSED flag on the schedule view |
| a loop runs but cannot decide (data gone, store down) | `metrics.verdict = could_not_run` with reason | Research Ledger shows `error`; `GET /research/schedule` counts consecutive `could_not_run` per kind, exported as `edge_learn_could_not_run_streak{kind}`; alert at ≥ 3 (trade-infra) |
| a gate silently passes on missing evidence | impossible by construction: None fails closed | unit tests of `decide` both directions (a pass and every fail mode) |
| a ledger write fails | `stamp_run` swallows it today (`learn_ledger.py:45-54`) | `run()` returns the ledger outcome; the trade-data runner counts `edge_learn_ledger_write_failed_total{kind}`; alert > 0 |
| a lifted statistic changes a number | n/a | trade-data's existing tests run against the lifted function before the copy is deleted |

## Slices (PRs), in order

1. **llm-ports: `llm_ports.learn` core** (types, gate, stats, examples, ledger, loop runner, JSONL
   adapters, testing fakes, standalone contract, guide, ADR-0010, API page) → release 1.5.0.
2. **llm-ports: LLM labels on the core**: re-scope WIP `04678fd` (below) → 1.6.0.
3. **trade-data: adoption plumbing**: pin; `LearnRunsLedger` adapter; `run()` behind
   `POST /admin/run/fit/{model}`; Research Ledger reads `metrics.gate` generically; derived score
   slot; delete the Wilson / BH / effective-n copies (not `pattern_stats`); the two failure gauges.
4. **trade-data + trade-ui + trade-infra: schedule view**: `GET /research/schedule`, Research
   Ledger columns, the `could_not_run` streak alert.
5. **trade-data: existing gates return `Verdict`s**: promotion_gate, domain/promotion, fit
   `earned()`s, llm_judge_trust; sealed-gate wrappers. Thresholds unchanged.
6. **trade-data: LLM label loop** (original Step 4): queue, coverage, LABELS card (§LLM labels).
7. **trade-data: the trend loop**, first new consumer, weight 0, display-only on the Research
   Ledger; measures the adoption budget. Reads `quant.ohlcv_daily_mv` on `bar_date` (never `ts`)
   joined to the calendar; honest-fill forward returns per `docs/STRATEGY-SYSTEM.md`. Roadmap line
   in `plans/estate-roadmap.md` when this doc lands.
8. **Fan-out**: judges read `few_shot`; copilot's similar-setups adapter; remaining loops.

Slices 1-2 are core, 3-5 make it easy, 6-8 go wide: the operator's order.

## LLM labels (one loop kind)

`EVAL-LABELS-CONSOLIDATION.md` holds the inventory and the label decisions; they stand with these
changes:

- `ports.Label` / `ports.LabelStore` (WIP `04678fd`) become `learn.Example` (topic
  `llm:<role>`, source human / model / judge) and `learn.ExampleStore`; `evals.labels` keeps only
  what is LLM-specific: `draw` (stratified by the judge's verdict), `label_of(JudgeVerdict)`.
  `coverage` and `agreement` move to `learn.examples`.
- `adapters/langfuse/labels.py` becomes `LangfuseExampleStore` (scores `label:<role>` on the
  production generation; annotation queue per role); `adapters/langfuse/experiments.py` stays.
- The judge-trust gate becomes a `Loop`: examples = human labels from the store, clauses =
  `labelled >= 20`, `kappa >= 0.6`, `seeded_bad_all_caught`, recorded as `run_kind =
  'llm_judge_trust'`; `edge_llm_judge_trusted` reads its latest verdict instead of a baked JSONL.
- Langfuse stays the LLM label store (UI exists, the label sits on the reply it labels, already
  backed up). A replay read from a Langfuse export was cut at 4,000 characters (sibling notes
  2026-11-07; source of the cut UNVERIFIED): the queue job records the length it enqueued and the
  label loop refuses a label whose stored reply length differs.

## Corrections to the inputs

- `training-engine-inventory.md` §6 says `llm_judge_trust` is not on main. It is: trade-data#2823
  merged as `6e941731`, and `service/platform/llm_judge_trust.py` (109 lines) is on main.
- `eval-labels-design-review.md` (the prior reviewer's findings) is an empty file on disk; its
  findings are not recoverable. This doc gets a fresh review.

## Not in scope

Training or fine-tuning runs; a new judge model choice (measured by a sibling session); MLflow;
editing either sealed gate; moving any fit usecase out of trade-data; a new UI page.
