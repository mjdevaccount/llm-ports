# The learning-loop core: one spine, one gate, one ledger

Status: DESIGN, 2026-11-07, revision 2 (answers the review). Supersedes the scope of `EVAL-LABELS-CONSOLIDATION.md`, which stays
as the detail of ONE loop kind (LLM labels, §LLM labels below). Owner of the decision: this file.
Facts read from `origin/main` of each repo on 2026-11-07 (trade-data `6e941731`, trade-ui,
ci-kit, work-timeline) with `git show` / `git grep`; every load-bearing claim carries its
`file:line`. A claim not checked that way says UNVERIFIED.

## What was asked, in priority order

1. **Centralize the core in one good home**: portable, zero EDGE imports, storage behind a port,
   usable in an unrelated codebase, including a portable Postgres ledger (Decision 2).
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
| level-source champion / challenger | level events | reaction hits | — | `domain/promotion.py` `evaluate` (promote / hold / reject) | ledger row |
| level calibration | level reactions | reaction / no reaction | `calibrate_level_reactions.py` | `signals/level_calibration.py` coarse gate cell | standard verdict |
| weight ledger | learned weights | — | — | `quant.weight_ledger` (mig 074), versioned, operator opt-in to apply | the APPLY stage of the picker loop: read, not replaced |
| pattern ladder | detectors | base rates | — | `PATTERN-GATE.md` ladder (SEALED) | wrap only |
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
(`domain/promotion.py:71`: `promote | hold | reject`; a clause that could not be evaluated
gives HOLD, a clause that LOOKED and disqualified gives REJECT, and REJECT wins over HOLD),
`llm_judge_trust.Trust(model, trusted, reason, kappa…)` (`platform/llm_judge_trust.py:53`), plus
`earned()` booleans in each fit usecase. The Research Ledger (`signals/research_ledger.py`) then
re-derives an earn state per run kind from each kind's ad-hoc `metrics`, because no run states
its verdict the same way. llm-ports itself has two more: `evals.calibration.verdict` (a judge's
calibration verdict) and `provenance.GateVerdict` (a grounding gate's). Both stay: they are
LLM-shaped results, not loop gates; `evals.calibration` builds on `learn.stats` (Decision 1).

The ledger is already in use under the name this core must NOT take: `metrics.gate` holds
`{promote, blocker, clauses: dict[str, bool]}`, written by `compiler/regrade.py:126` and
`strategy/lever_grid.py:284`, read by `research_ledger._promote_gate` (:125-129, AUTHORITATIVE),
`regrade.verdict_of` (:139), `jobs/handlers/replay.py:315,366`, `strategy/hunt.py:150`, copilot's
chat tune agent and trade-ui `TuneChatPanel.tsx:62-65`. And `metrics.verdict` is not one
vocabulary: `supported | refuted | inconclusive` (regrade), `probe_pass | probe_fail`
(`rag_eval`), free text (`ou_meanrev_diag`). Decision 3 therefore writes under a new key.

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

**A portable Postgres ledger** (requirement 6) is a later llm-ports adapter, `adapters/postgres/`
behind a `postgres` extra (`psycopg`), owning one table of the same shape as `quant.learn_runs`
(`run_kind, subject, params jsonb, metrics jsonb, created_at`). It is NOT in slice 1: the core
proves the port with two stdlib ledgers first, and the first outside user who wants Postgres
gets the adapter. trade-data keeps its own adapter over `LearnRunsRepo` either way: its table has
grants, FKs and readers a generic adapter must not own.

## Decision 3 — ONE gate: evidence in, verdict + audit row out

```text
Clause(name, passed: bool | None, value, threshold, op, role)   passed None = not evaluable
role    ∈ bar | floor | veto                                     default bar
Verdict(gate, subject, decision, clauses, blocker, reason)      frozen, JSON-able
decision ∈ pass | fail | insufficient | could_not_run
decide(gate, clauses, *, subject="", unevaluated="fail") -> Verdict
```

`decide`, in order (first rule that applies wins; the blocker is the first clause, in the order
given, that the rule names):

1. no clauses → `could_not_run`.
2. a `veto` clause evaluated False → `fail` (we looked and it disqualifies, whatever the sample).
3. a `floor` clause not True (False or None) → `insufficient` (a sample floor not met: anything
   else measured on that sample is noise, so a failing `bar` is not yet a `fail`).
4. `unevaluated="fail"` (default): any clause not True → `fail`. `unevaluated="insufficient"`:
   a clause evaluated False → `fail`, else a clause that is None → `insufficient`.
5. otherwise `pass`.

- **Fails closed.** A None clause is never a pass under either policy. The default (`fail`) is
  `promotion_gate._ok` (:78) and the strict AND of each fit's `earned()`; `insufficient` is the
  domain/promotion law ("a conjunct that could not be evaluated has not been satisfied", HOLD).
- **promote / hold / reject map exactly**: `domain/promotion` adopts with `min_sample`,
  `reaction_lift` when the lift is positive but not robust, and `deflated_bar` as `floor`
  (failing = unproven = HOLD), `displacement` and `reaction_lift` with no lift at all as `veto`
  (REJECT), and `unevaluated="insufficient"`. Then `pass`=promote, `insufficient`=hold,
  `fail`=reject, and REJECT-beats-HOLD is rule 2 before rule 3. The clause's role is set by the
  gate when it evaluates the clause, so one clause can be a `veto` in one branch and a `floor` in
  another, as `reaction_lift` is there.
- **Every clause is kept, not only the blocker.** `Verdict.clauses` holds every clause in order
  and `Verdict.blocking` lists every clause that is not True, so a surface renders all reasons
  (what `PromotionDecision.reasons` lists) and the clause map that "cannot hide a skipped
  conjunct" (`domain/promotion.py:79-88`).
- **Its own key in the ledger: `metrics.learn_verdict`.** A recorded run writes
  `metrics.learn_verdict = Verdict.as_dict()` (its `decision` is the verdict) and never writes
  `metrics.gate` or `metrics.verdict`, whose readers are listed above. A kind that already
  writes those keeps writing them; nothing reading them changes. The Research Ledger gains ONE
  generic branch: a row with `learn_verdict` maps `pass`→earned, `fail`→not_earned,
  `insufficient`→insufficient, `could_not_run`→error, checked after the existing
  `_promote_gate` branch so a kind with both keeps today's state. A row with neither stays
  `informational`. `latest_with_verdict` is unchanged; `LearnRunsLedger.latest(decisions=...)`
  filters on `metrics->'learn_verdict'->>'decision'`.
- **Audit = the ledger row.** `loop.run` (and `loop.record` for a gate run outside a `Loop`)
  appends one row. A gate is a pure function `evidence -> clauses`; there is no base class.
- **Existing gates adopt by returning clauses**, thresholds unchanged: `promotion_gate`,
  `domain/promotion`, each fit's `earned()`, `eval_metrics._blockers`, `llm_judge_trust.assess`,
  the level-calibration coarse gate cell.
- **Sealed gates are wrapped, never changed.** The card and pattern gates (and the pattern
  ladder) keep their code and documents; an adapter in trade-data reads their DECIDED outputs and
  records them as `Verdict`s (`gate="card-system"`, `"pattern"`), so they appear on the same
  ledger and page. The pattern code's `bh_fdr` (`signals/pattern_stats.py:57`) is NOT replaced.

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
  an `ExampleStore` adapter, not a second index. Copilot's event tools (`get_event_performance`,
  `get_calendar`, `get_structural_events`, `get_market_moments`) are evidence READERS for the
  event-study and trend loops: a loop's `evidence()` calls the same usecases they call, never a
  copy. Similarity search is NOT in the core (it needs a vector store); a `SimilarExamples`
  Protocol is added when its first adapter is, not before.

## Decision 5 — the engine: trade-data's learning plane, only agreement + FDR math lifted

"We have a training engine already": the engine IS trade-data's learning plane, and stays there:
`FIT_REGISTRY` dispatch (`usecases/fit_registry.py`, `POST /admin/run/fit/{model}`), admin-run
declarations with `learn_kind` (`admin/admin_run_declarations_*.py`), Windmill schedules, the loop
registry with its `loop-stalled` watchdog, the weight ledger, the run ledger. Nothing of that
moves; it gains the core's types. What lifts is only math that has ONE definition:

| function | today | in the core |
|---|---|---|
| kappa / AC1 | llm-ports `evals/calibration.py:54,75` | `learn.stats.cohen_kappa`, `gwet_ac1`; re-exported from `evals.calibration` |
| false-discovery control | `signals/eval_metrics.py:272` (+ `pattern_stats.py:57`, sealed: stays) | `learn.stats.bh_fdr`, **Benjamini-Yekutieli by default** (valid under any dependence between tests, the estate's correlated signals), `dependent=False` for plain Benjamini-Hochberg (independent tests) |

NOT lifted, on evidence: **Wilson** is already one scipy shim in trade-data (`domain/stats.py`;
`level_calibration` and `setup_radar` delegate to it, consolidation B3), so a lift would add a
second copy, not remove one. **Effective n** is three DIFFERENT estimators (consolidation F4:
autocorrelation-adjusted, Kish weights, cluster count), not one formula with three spellings.
Reliability / ECE / Brier, Platt, purged CV lift in the slice whose consumer needs them (the
trend loop), with the same test. A lift is byte-for-byte on numbers: trade-data's own tests run
against the lifted function before its copy is deleted. Fit usecases (they read tables) are NOT
lifted.

## Decision 6 — ONE registry, ONE view of when training runs

- **The registry is `service/platform/loops.py`** and its generated `docs/LEARNING-LOOPS.md` +
  `GET /ops/loops` (`admin/ops_router.py:36`). Not forked. A loop built on the core declares its
  `LoopSpec` the same way; a helper derives its `score` slot (`quant.learn_runs` where
  `run_kind=<kind>`, precedent `loop_registrants_llm.py:53`). `docs/LEARNING-LOOPS.md` (every
  loop), `docs/design/LEARNING-LOOP-OVERVIEW.md` (the picker loop in depth) and
  `docs/AUTOMATION-TIMELINE.md` (what runs when) each gain one paragraph pointing here; none is
  replaced.
- **A loop's run kind rides its declaration, not its route.** `AdminRunSpec.kind` is the first
  path segment only (`platform/admin_run.py:52`), and `spec_for` resolves
  `/admin/run/fit/{model}` to `fit` (:74-82), so a route-level `learn_kind` cannot name a
  per-model loop. Each `FIT_REGISTRY` entry therefore declares its own `learn_kind` (a
  `FitSpec(name, fit, read, learn_kind)` replacing the bare `(fit, read)` tuple), the fit
  dispatcher stamps that kind, and its `ops.work` row is named `fit.<model>`, not `fit`
  (today an HTTP run is named by its root segment, so every fit loop would hide behind one
  name: PLAUSIBLE per the review, slice 3 verifies and fixes it before any MISSED flag reads it).
  "Every learn kind" is then ONE read, `learn_kinds()`: admin-run declarations with a
  `learn_kind` plus `FIT_REGISTRY` specs.
- **"When does training run" is a JOIN of records that already exist**, not a new store:
  `learn_kinds()` names each kind and its trigger; the LIVE schedules come from Windmill through
  `adapters/windmill_ops.list_schedules` (`windmill_ops.py:75`, already read by
  `usecases/automation_status.py:80`), never the `windmill/` tree (it is not in the image); box
  timers are named as triggers but carry no next-run in this view; `ops.work` holds each run's
  start, end and status; `quant.learn_runs` holds its result and verdict. One read,
  `GET /research/schedule`, joins them per kind: last run, status, verdict, next run, and a
  MISSED flag (a scheduled run with no `ops.work` row in its window, the rule `Slot.work_name`
  already encodes, `loops.py:107`). A kind with no schedule says `manual` (every `FIT_REGISTRY`
  model today, `loop_registrants.py:85,94,102`), never blank.
- **The view is keyed by registered loop, not by run kind**: a loop in `loops.py` with no
  `learn_runs` stamp (hazard exit today) shows as a row with `no ledger row`, so it is visible
  instead of absent.
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
| `learn` (types) | `Example`, `Clause`, `Verdict`, `Decision`, `LedgerRow`, store results (`ExamplesRead`, `RowRead`, `Written`) |
| `learn.gate` | clause helpers (`at_least`, `at_most`, `equals`, `check`), `decide`, `could_not_run` (pure) |
| `learn.stats` | `bh_fdr` (BY default), `cohen_kappa`, `gwet_ac1` |
| `learn.examples` | `coverage`, `agreement`, `few_shot` |
| `learn.stores` | `MemoryRunLedger`, `JsonlRunLedger`, `MemoryExampleStore`, `JsonlExampleStore`, `union` |
| ports | `RunLedger`, `ExampleStore`, `WritableExampleStore`, `Loop`, `Fit` Protocols / values |
| `learn.loop` | `run(loop, ledger) -> Run`; `record(ledger, kind, verdict)` for a gate run outside a loop |

**The loop has four stages: `evidence → fit → clauses → apply`.** `run` drives the first three,
decides the verdict, then calls `loop.apply(verdict, fit)`, so what a loop does with its verdict
(store the fitted map on `pass`, the identity on anything else, as `fit_prob_calibration` does
inside `compute` today via `earned()`) is a named stage, not buried in the fit. `apply` returns
what it did (`"stored"`, `"held"`, ...), recorded in the same row. A stage that raises becomes
`could_not_run` with the stage named (an `apply` that raises after a `pass` is recorded as
`apply_failed` in the row, keeping the verdict). Exactly one ledger row per run, every exit
included; `run` never raises. A loop that wants no action implements `apply` as a no-op. The
legacy per-model routes keep their own path until slice 5 moves each onto `run`; until then they
stamp through `stamp_run` as today.

**Async**: the Protocols are `async` (the stores a loop reads are async; trade-data is asyncio),
`fit` and `clauses` stay synchronous (pure). A synchronous backend runs its calls in a thread.

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
| `service/usecases/fit_registry.py` | +1 line | `register_loop(<Loop factory>)`: a slice-3 helper builds the `FitSpec` (`fit` = `run(loop, ledger)`, `read` = the ledger's latest `pass`, `learn_kind` = `loop.kind`), dispatched by the existing `POST /admin/run/fit/{model}` |
| `service/usecases/loop_registrants.py` | +1 entry | the `LoopSpec` (score slot derived) |

**Zero migrations, zero routes, zero UI code**: the ledger is `learn_runs`, the route is the fit
route, the Research Ledger page lists any kind and its schedule. A cross-sectional loop (many
symbols, one verdict) ignores the dispatcher's `symbol` argument, as the scope-global slippage
fit already does (`fit_registry.py` `_fit_slippage`), and puts its universe in `params`. The
budget holds only for a loop that stores nothing beyond its ledger row (its fitted `model` lives
in `metrics`); a loop that needs a per-symbol table pays one migration and one repository, stated
in its PR. Budget: ≤ 7 files, ≤ 250 lines
including tests. The trend loop's PR reports its `git diff --stat` against this budget; a miss is
a core defect to fix, not a number to restate.

## Portability (acceptance criterion, proved in CI)

- import-linter contract `learn-standalone`: `llm_ports.learn` imports no other `llm_ports`
  module and no third-party package.
- `import llm_ports.learn` still runs `llm_ports/__init__` (pydantic, jinja2, pyyaml), so the
  contract alone does not prove portability. `tests/contract/test_learn_standalone.py` copies
  `src/llm_ports/learn/` into a temporary directory as a top-level package, rewrites its
  `llm_ports.learn` imports, and runs it in a subprocess whose `sys.modules` must then hold no
  third-party module: the "lift it into another codebase" claim, executed.
- `tests/contract/test_core_imports.py` (the `core-only` CI job, a bare install) walks `llm_ports.learn`
  with the rest of the core and asserts it loads no vendor module; `examples/learning_loop.py` runs on every pytest leg.
- The deployment-vocabulary pre-commit hook already rejects EDGE names in `src/llm_ports/`.
- Docs say how to plug in a ledger: implement two methods.

## Telemetry (what fails silently, where it lands, what watches it)

| silent failure | lands in | watched from outside by |
|---|---|---|
| a loop stops running | no `learn_runs` row / no `ops.work` row in window | `loop-stalled` watchdog (exists) via the derived score slot; MISSED flag on the schedule view |
| a loop runs but cannot decide (data gone, store down) | `metrics.learn_verdict.decision = could_not_run` with reason | Research Ledger shows `error`; a `/metrics` collector (`service/observability/learn_gauge.py`, precedent `day_brief_gauge.py`) computes `edge_learn_could_not_run_streak{kind}` from the ledger at every Prometheus scrape, independent of any page being open; alert at ≥ 3 (trade-infra) |
| a loop never earns: stuck at `insufficient` forever (the sample never grows) | `learn_verdict.decision = insufficient`, every run | the same collector exports `edge_learn_days_insufficient{kind}` (days since the kind's first consecutive `insufficient`); alert at ≥ 30 days, so a floor that cannot be reached is named rather than waited on |
| an `apply` fails after a pass | `metrics.learn_apply = apply_failed` | the collector's `edge_learn_apply_failed_total{kind}`; alert > 0 |
| a gate silently passes on missing evidence | impossible by construction: None fails closed | unit tests of `decide` both directions (a pass and every fail mode) |
| a ledger write fails | `stamp_run` swallows it today (`learn_ledger.py:45-54`) | `run()` returns the ledger outcome; the trade-data runner counts `edge_learn_ledger_write_failed_total{kind}`; alert > 0 |
| a lifted statistic changes a number | n/a | trade-data's existing tests run against the lifted function before the copy is deleted |

## Slices (PRs), in order

1. **llm-ports: `llm_ports.learn` core** (types, gate, stats, examples, stores, loop runner with
   its `apply` stage, standalone contract, guide, ADR-0010, API page) → release 1.5.0.
2. **llm-ports: LLM labels on the core**: re-scope WIP `04678fd` (below) → 1.6.0.
3. **trade-data: adoption plumbing**: pin; `LearnRunsLedger` adapter; `FitSpec` with
   `learn_kind` + `register_loop`; `ops.work` named `fit.<model>`; `learn_kinds()`; Research
   Ledger reads `metrics.learn_verdict` generically; derived score slot; `eval_metrics` BH copy
   replaced by `learn.stats.bh_fdr(dependent=False)` (numbers unchanged; not `pattern_stats`); the
   `learn_gauge` collector.
4. **trade-data + trade-ui + trade-infra: schedule view**: `GET /research/schedule` (Windmill
   via `list_schedules`, keyed by registered loop), Research Ledger columns, the three alerts.
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
  findings are not recoverable. This doc got a fresh review (`learning-loop-core-design-review.md`
  in the session notes, DO NOT SHIP, 11 findings); this revision answers each: 1 →
  `learn_verdict` (Decision 3), 2 + 3 → `FitSpec.learn_kind`, `list_schedules` (Decision 6), 4 →
  lift only kappa/AC1/BY (Decision 5), 5 → standalone lifted-copy test, other verdict types named,
  6 → roles + `unevaluated` policy, 7 → `apply` stage, 8 → scrape-time collector, 9 → async
  stated, 10 → cross-sectional + budget scope, 11 → inventory rows, event tools, sibling docs,
  Postgres adapter, view keyed by loop.

## Not in scope

Training or fine-tuning runs; a new judge model choice (measured by a sibling session); MLflow;
editing either sealed gate; moving any fit usecase out of trade-data; a new UI page.
