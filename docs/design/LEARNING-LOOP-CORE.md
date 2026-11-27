# The learning-loop core: one spine, one gate, one ledger

Status: DESIGN, 2026-11-07, revision 2 (answers the review). Supersedes the scope of `EVAL-LABELS-CONSOLIDATION.md`, which stays
as the detail of ONE loop kind (LLM labels, §LLM labels below). Owner of the decision: this file.
Facts were read from the main branch of each codebase involved on 2026-11-07 (the first consumer
application, its UI, its shared CI kit, a timeline library) with `git show` / `git grep`; every
load-bearing claim was checked against its `file:line` there. A claim not checked that way says
UNVERIFIED.

## What was asked, in priority order

1. **Centralize the core in one good home**: portable, zero consumer imports, storage behind a
   port, usable in an unrelated codebase, including a portable Postgres ledger (Decision 2).
2. **Make a new loop cheap**: this doc states the cost in files and lines; the first new consumer
   loop (a cross-sectional daily trend loop) measures it.
3. **Then fan out**: existing loops adopt it, judges and the consumer's chat assistant read the
   shared store.

Hard limits: the consumer's two SEALED gate documents (pre-registered, content-pinned) are never
edited; the core may only WRAP what they decide, and their strategy weight stays 0. The
consumer's existing learn engine is reused, never forked ("it must not end with two engines"). No
fine-tuning or training run; strong-model labelling goes through the deployment's existing model
gateway or self-hosted models.

## The shape

Every loop in this platform is the same five moves:

```text
  evidence in        labelled outcomes         fit / calibrate        gate             act + audit
  (series, events,  ──▶ (outcome resolved,  ──▶ (a model, a weight, ──▶ (evidence ──▶ ──▶ (promote at weight,
   replies, picks)       human / model /         a threshold, a           verdict)          show, page) + one
                         judge label)            calibration)                               ledger row per run
```

The consumer already names the stages `predict / resolve / score / consume / surface` in its loop
registry and registers ~45 loops against them. The core does not replace that vocabulary; it
supplies the parts every loop re-wrote: the gate, the run record, the labelled-example contract,
the statistics.

### Where the shape occurs (and what each is missing)

| loop | evidence | outcome / label | fit | gate today | missing |
|---|---|---|---|---|---|
| ranking-signal weights | ranker runs | resolved outcomes (end of day) | an IC-weighted EWA | a blocker list in the eval-metrics module | standard verdict row |
| probability calibration | p per pick | outcome | Platt | an `earned()` boolean in the fit usecase | verdict lives in code, not ledger |
| duration models | paths | timing of the best outcome | two fit usecases | concordance > 0.53 / eff_n ≥ 30 | same |
| meta-label | features | take/skip outcome | a meta-label model | identity under 40 samples | same |
| config grids | replay | graded path | grid | a promotion gate (not wired live) | wiring |
| source champion / challenger | source events | hits | — | promote / hold / reject | ledger row |
| threshold calibration | reactions | reaction / no reaction | a calibration usecase | a coarse gate cell | standard verdict |
| weight ledger | learned weights | — | — | a versioned weight table, operator opt-in to apply | the APPLY stage of the ranking loop: read, not replaced |
| two sealed gates and a display ladder | detectors, a daily card | base rates, session outcome | — | the sealed documents | wrap only |
| a screening board | candidates | forward outcome | — | a Wilson lower bound | gate, ledger |
| event study | calendar events | forward outcome | an event-study usecase | FDR families | ledger, gate |
| article extraction | articles | human-labelled fields (203) | — | an extraction eval pass/fail | already the precedent |
| LLM judge trust | judged replies | human verdicts (~20) | — | the judge-trust `assess` | labels (§LLM labels) |
| LLM roles (34) | replies | none | — | a sampled small judge (kappa ~0.19) | labels, coverage |
| **trend** (new) | a daily series + calendar | forward outcome | rank / logistic | — | everything: the proof consumer |

Gates in use today, three shapes and no shared type: a `GateVerdict(promote, clauses, blocker)`, a
`PromotionDecision` (`promote | hold | reject`; a clause that could not be evaluated gives HOLD,
a clause that LOOKED and disqualified gives REJECT, and REJECT wins over HOLD), a judge-trust
`Trust(model, trusted, reason, kappa…)`, plus `earned()` booleans in each fit usecase. The
consumer's research ledger then re-derives an earn state per run kind from each kind's ad-hoc
`metrics`, because no run states its verdict the same way. llm-ports itself has two more:
`evals.calibration.verdict` (a judge's calibration verdict) and `provenance.GateVerdict` (a
grounding gate's). Both stay: they are LLM-shaped results, not loop gates; `evals.calibration`
builds on `learn.stats` (Decision 1).

The ledger is already in use under the name this core must NOT take: `metrics.gate` holds
`{promote, blocker, clauses: dict[str, bool]}`, written by two of the consumer's modules and read
by its research ledger (AUTHORITATIVE), its regrade verdict, its replay handlers, its strategy
search, its chat tune agent and a UI panel. And `metrics.verdict` is not one vocabulary:
`supported | refuted | inconclusive` (regrade), `probe_pass | probe_fail` (a retrieval eval), free
text (a diagnostic). Decision 3 therefore writes under a new key.

## Decision 1 — the core's home: `llm_ports.learn`

A new subpackage of llm-ports, enforced standalone:

- **Why llm-ports and not a new repo.** llm-ports is already a published, portable library: PyPI
  release by tag, strict mypy + `pyright --verifytypes` 100%, import-linter layers, an executed
  docs site, a `test-lowest` leg, and a core that installs with no vendor. A new repo would
  rebuild all of that for ~600 lines and add a second pin to every consumer: "should probably not
  have two". The pure statistics it needs (`cohen_kappa`, `gwet_ac1`) already live here
  (`evals/calibration.py`).
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

## Decision 2 — the run ledger: the consumer's run table, behind a port

| option | for | against | verdict |
|---|---|---|---|
| **extend the consumer's run table** | exists, ~20 kinds already write it, `run_kind` is free text with no CHECK, grants for the app, the assistant and dashboards, a research registry has foreign keys onto it, a `latest_with_verdict` reader already reads `metrics->>'verdict'`, and the consumer's research-ledger page renders it | Postgres-only; a schema-specific adapter | **chosen** |
| MLflow tracking + registry | real UI, model registry | a NEW service (Postgres backend, backup, auth), a second registry beside the loop registry, and the run table's own migration already rejected it for a single operator; the consumer's "models" are small param rows, not artifacts | rejected for the consumer; possible later optional adapter for portable users |
| Langfuse | already deployed; datasets, scores, annotation queues | LLM-shaped (traces, observations); a numeric fit is not a trace | rejected as the ledger; KEPT as the LLM label store (§LLM labels) |

Portability comes from the port, not the table: `learn.RunLedger` (`append`, `latest`) with three
implementations: `MemoryRunLedger` and `JsonlRunLedger` (stdlib, in the core, so the library works
anywhere with zero setup), and the consumer's own adapter over its run repository. An MLflow
adapter is an optional extra when an outside user asks for it.

**A portable Postgres ledger** (requirement 6) is a later llm-ports adapter, `adapters/postgres/`
behind a `postgres` extra (`psycopg`), owning one table of the same shape as the consumer's run
table (`run_kind, subject, params jsonb, metrics jsonb, created_at`; the consumer's table names the
subject column `symbol` and adds `git_sha` and `notes`, so its adapter maps `subject` to
`symbol`). It is NOT in slice 1: the core proves the port with two stdlib ledgers first, and the
first outside user who wants Postgres gets the adapter. The consumer keeps its own adapter either
way: its table has grants, foreign keys and readers a generic adapter must not own.

**The consumer's ledger adapter answers `ok` only after its own commit.** Its repository's
`append` leaves the commit to its caller and the generic fit route opens a session it never
commits, so an `ok` before commit would be a row that rolls back while `Run.recorded.ok` says it
landed: the silent green this design closes. The adapter writes on its OWN session (a session
factory, not the dispatcher's session), so its commit neither commits nor discards `apply`'s
pending writes on the dispatcher's session; after `apply_failed` those writes are the
dispatcher's to roll back, and the row still lands.

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

- **Fails closed.** A None clause is never a pass under either policy. The default (`fail`) is the
  existing promotion gate's rule and the strict AND of each fit's `earned()`; `insufficient` is
  the promote/hold/reject law ("a conjunct that could not be evaluated has not been satisfied",
  HOLD).
- **promote / hold / reject map exactly**: that gate adopts with `min_sample`, a lift clause when
  the lift is positive but not robust, and a deflated bar as `floor` (failing = unproven = HOLD),
  a displacement clause and the lift clause with no lift at all as `veto` (REJECT), and
  `unevaluated="insufficient"`. Then `pass`=promote, `insufficient`=hold, `fail`=reject, and
  REJECT-beats-HOLD is rule 2 before rule 3. The clause's role is set by the gate when it
  evaluates the clause, so one clause can be a `veto` in one branch and a `floor` in another, as
  the lift clause is there.
- **Every clause is kept, not only the blocker.** `Verdict.clauses` holds every clause in order
  and `Verdict.blocking` lists every clause that is not True, so a surface renders all reasons
  (what `PromotionDecision.reasons` lists) and the clause map that "cannot hide a skipped
  conjunct".
- **Its own key in the ledger: `metrics.learn_verdict`.** A recorded run writes
  `metrics.learn_verdict = Verdict.as_dict()` (its `decision` is the verdict) and never writes
  `metrics.gate` or `metrics.verdict`, whose readers are listed above. A kind that already
  writes those keeps writing them; nothing reading them changes. The consumer's research ledger
  gains ONE generic branch: a row with `learn_verdict` maps `pass`→earned, `fail`→not_earned,
  `insufficient`→insufficient, `could_not_run`→error. It sits IMMEDIATELY after the existing
  `metrics.gate` branch and before the per-kind extractors: a run merges fit metrics into the top
  level, so a core loop whose fit reports `fdr_pass`, `dsr` or `pbo` would otherwise be read by a
  per-kind extractor instead of its verdict. A fit metric `status="failed"` still preempts it, as
  today. A kind with both keys keeps today's state (so it renders `insufficient` as `not_earned`)
  until its slice-5 PR stops writing `gate`. The ledger's rigor read gains the same one generic
  branch (a `learn_verdict` row's rigor read from its clauses), so a new loop gets a rigor
  without a per-kind edit. A row with neither key stays `informational`. `latest_with_verdict` is
  unchanged; the ledger adapter's `latest(decisions=...)` filters on
  `metrics->'learn_verdict'->>'decision'`.
- **Audit = the ledger row.** `loop.run` (and `loop.record` for a gate run outside a `Loop`)
  appends one row. A gate is a pure function `evidence -> clauses`; there is no base class.
- **Existing gates adopt by returning clauses**, thresholds unchanged: the promotion gate, the
  promote/hold/reject gate, each fit's `earned()`, the eval-metrics blocker list, the judge-trust
  `assess`, the threshold-calibration gate cell.
- **Sealed gates are wrapped, never changed.** The two sealed gates (and the display ladder) keep
  their code and documents; an adapter in the consumer reads their DECIDED outputs and records
  them as `Verdict`s, so they appear on the same ledger and page. Their own FDR code is NOT
  replaced.

## Decision 4 — the training-data contract and its read surface

```text
Example(topic, subject, value, source, by, at, reason="", ref={})
  topic   what is being learned: "llm:research.web", "trend:1d", "ranker:A-1"
  subject the thing labelled (an observation id, "A-1@2026-11-06", a pick id)
  value   the label: "good" | "bad" for verdicts, a number for an outcome
  source  human | model | judge | outcome      (outcome = resolved mechanically from the world)
  by      who: an operator id, a model id, a grader name
  ref     provenance: trace id, table + key, git sha
```

- **One contract, many physical stores, no copying.** Domain outcomes stay in their tables; LLM
  verdict labels stay in Langfuse scores; the operator's review truth stays in its own table.
  Each store is an `ExampleStore` adapter that READS its table as `Example`s; `learn.union(stores)`
  reads several as one. A copy would drift.
- **`ExampleStore`** (Protocol, never raises, named outcomes): `examples(topics, *, since,
  source)` over a sequence of topics; optional `put(examples)` (a sequence) for stores that accept
  writes (Langfuse scores, a JSONL file).
- **Pure readers in the core:** `coverage(examples, topics, target)` (every topic appears, zero
  rows included: the gap is the point); `agreement(examples, *, reference="human",
  measured=("judge", "model"), positive)` (kappa + AC1 on shared subjects);
  `few_shot(examples, k, *, topic, source="human")` (deterministic, always balanced across both
  classes).
- **The read surface for judges and the assistant** ("all of this could be used by the judge"):
  `few_shot` hands an LLM judge graded examples of its own role; the assistant's existing
  similar-cases search (pgvector over past reflections) stays the similarity search and gains an
  `ExampleStore` adapter, not a second index. The assistant's event tools are evidence READERS
  for the event-study and trend loops: a loop's `evidence()` calls the same usecases they call,
  never a copy. Similarity search is NOT in the core (it needs a vector store); a
  `SimilarExamples` Protocol is added when its first adapter is, not before.

## Decision 5 — the engine: the consumer's learning plane, only agreement + FDR math lifted

"We have a training engine already": the engine IS the consumer's learning plane, and stays
there: a fit registry dispatched by one run route, run declarations carrying a `learn_kind`,
scheduler crons, the loop registry with its stalled-loop watchdog, the weight ledger, the run
ledger. Nothing of that moves; it gains the core's types. What lifts is only math that has ONE
definition:

| function | today | in the core |
|---|---|---|
| kappa / AC1 | llm-ports `evals/calibration.py` | `learn.stats.cohen_kappa`, `gwet_ac1`; re-exported from `evals.calibration` |
| false-discovery control | the consumer's `bh_fdr`, **BY by default** (`dependent=True`; 8 callers use the default, incl. the signal-trust gate; two of its comments mislabel it Benjamini-Hochberg). Stay: the sealed gate's own copy and a scipy BY q-value helper used by discovery and event-study code | `learn.stats.bh_fdr`, **Benjamini-Yekutieli by default** (valid under any dependence between tests, as correlated signals are), `dependent=False` for plain Benjamini-Hochberg (independent tests). Byte-identical to the consumer's copy (alpha 0.10, None excluded from m, same tie order), so its callers keep the DEFAULT and their numbers do not change |

NOT lifted, on evidence: **Wilson** is already one scipy shim in the consumer, which its callers
delegate to, so a lift would add a second copy, not remove one. **Effective n** is three
DIFFERENT estimators (autocorrelation-adjusted, Kish weights, cluster count), not one formula
with three spellings. Reliability / ECE / Brier, Platt, purged CV lift in the slice whose consumer
needs them (the trend loop), with the same test. A lift is byte-for-byte on numbers: the
consumer's own tests run against the lifted function before its copy is deleted. Fit usecases
(they read tables) are NOT lifted.

## Decision 6 — ONE registry, ONE view of when training runs

- **The registry is the consumer's loop registry** and its generated loop list and read endpoint.
  Not forked. A loop built on the core declares its `LoopSpec` the same way; a helper derives its
  `score` slot (the run table where `run_kind=<kind>`, as an existing LLM loop does). The loop
  list, the in-depth overview of the ranking loop and the automation timeline each gain one
  paragraph pointing here; none is replaced.
- **A loop's run kind rides its declaration, not its route.** A run declaration's kind is the
  first path segment only, and the fit route resolves to `fit`, so a route-level `learn_kind`
  cannot name a per-model loop. Each fit-registry entry therefore declares its own `learn_kind`
  (a `FitSpec(name, fit, read, learn_kind)` replacing the bare `(fit, read)` tuple), and the fit
  dispatcher stamps that kind only for a spec that is NOT a loop: a `register_loop` spec's row
  is written by `run`, so a run is one row, never a second verdict-less row that the ledger's
  per-kind landscape (DISTINCT ON `run_kind`) would show as `informational`.
- **A fit run's work-unit name is already `fit/<model>` and stays so.** The shared unit-name rule
  keeps the full remainder after the run prefix, and a reaper query splits on that slash. No
  rename.
- **`learn_kinds()` is declared kinds UNION observed kinds**: run-declaration `learn_kind`s and
  fit-registry specs, plus the distinct `run_kind`s in the run table. Today 16 kinds are stamped
  with literal kinds and declare none; each shows trigger `undeclared`, never `manual`, so the
  view names the gap instead of asserting a trigger, and slice 4 declares them.
- **"When does training run" is a JOIN of records that already exist**, not a new store:
  `learn_kinds()` names each kind and its trigger; the LIVE schedules come from the scheduler's
  API (already read by the consumer's automation-status usecase), never the scheduler's source
  tree (it is not in the image); host timers are named as triggers but carry no next-run in this
  view; the work table holds each run's start, end and status; the run table holds its result and
  verdict. One read endpoint joins them per kind: last run, status, verdict, next run, and a
  MISSED flag (a scheduled run with no work row in its window, the rule the registry's slot
  already encodes). **Two join keys, both exact:** loop to run is the work-unit name its
  `LoopSpec` trigger spells (`fit/<model>` for a fit loop, matched exactly); schedule to loop is
  the scheduler's flow path, which the `LoopSpec` declares and the schedule list returns (it
  returns schedule and flow paths, not the path the flow POSTs). A schedule that matches no loop
  is its own `unmatched schedule` row and a loop that declares a flow with no schedule says
  `no schedule`: neither is silent. A declared kind with no schedule says `manual`, never blank;
  a declared `manual` that a host cron in fact runs is corrected in slice 4.
- **The view is keyed by registered loop, not by run kind**: a registered loop with no run-table
  stamp shows as a row with `no ledger row`, so it is visible instead of absent.
- **The view is the existing research-ledger page** in the consumer's UI, which already lists
  every kind with its earn state. It gains last / next / missed columns. No new page.

## Decision 7 — the ratchet: mapped, not adopted

The CI kit's ratchet (`RatchetSet`, `RatchetCount`) IS a gate: findings in → regression or not →
the baseline file in git is the audit. It does not adopt the ledger: it runs in CI, which cannot
reach the deployment's Postgres, and its audit trail (the baseline's git history, with a stated
reason per re-freeze) is already append-only. If a ratchet ever reports into the view, it does so
by emitting a `Verdict` JSON as a CI artifact; nothing is built for it now. The CI kit stays its
home.

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
(store the fitted map on `pass`, the identity on anything else, as the probability-calibration
fit does inside its compute step today via `earned()`) is a named stage, not buried in the fit.
`apply` returns what it did (`"stored"`, `"held"`, ...), recorded in the same row. A stage that
raises becomes `could_not_run` with the stage named (an `apply` that raises after a `pass` is
recorded as `apply_failed` in the row, keeping the verdict). Exactly one ledger row per run,
every exit included; `run` never raises. A loop that wants no action implements `apply` as a
no-op. The legacy per-model routes keep their own path until slice 5 moves each onto `run`; until
then they stamp through their existing helper as today.

**Async**: the Protocols are `async` (the stores a loop reads are async; the consumer is
asyncio), `fit` and `clauses` stay synchronous (pure). A synchronous backend runs its calls in a
thread.

Every function is pure or behind a Protocol; no clock without injection; no I/O outside the JSONL
adapters. Docs: guide `docs/guides/learning-loops.md` with an executed, self-asserting example
(a loop over synthetic data from evidence to recorded verdict), API page, ADR-0010.

## The cost of a new loop (the adoption deliverable)

After slice 3 (below), a new loop in the consumer is:

| file | new / edit | about |
|---|---|---|
| the loop module | new | the `Loop`: read evidence (a repository call), fit, return clauses: ≤ 120 lines |
| its test module | new | pure tests of fit + clauses |
| a schedule + flow | new | the cron and the thin POST, copied |
| the fit registry | +1 line | `register_loop(<Loop factory>)`: a slice-3 helper builds the `FitSpec` (`fit` = `run(loop, ledger)`, `read` = the ledger's latest DECISIVE row, `latest(decisions=("pass", "fail"))`, serving its `learn_model` only when that row is `pass` with `learn_apply` not `apply_failed`, else nothing, so a later `fail` revokes the model as the probability-calibration fit does today; `learn_kind` = `loop.kind`), dispatched by the existing fit route |
| the loop registrants | +1 entry | the `LoopSpec` (score slot derived) |

**Zero migrations, zero routes, zero UI code**: the ledger is the run table, the route is the fit
route, the research-ledger page lists any kind and its schedule. A cross-sectional loop (many
subjects, one verdict) ignores the dispatcher's subject argument, as an existing scope-global fit
already does, and puts its universe in `params`. The budget holds only for a loop that stores
nothing beyond its ledger row (its fitted `model` lives in `metrics`); a loop that needs a
per-subject table pays one migration and one repository, stated in its PR. Budget: ≤ 7 files,
≤ 250 lines including tests. The trend loop's PR reports its `git diff --stat` against this
budget; a miss is a core defect to fix, not a number to restate.

## Portability (acceptance criterion, proved in CI)

- import-linter contract `learn-standalone`: `llm_ports.learn` imports no other `llm_ports`
  module. No third-party import comes from the no-vendor-in-core contract plus the subprocess
  test below.
- `import llm_ports.learn` still runs `llm_ports/__init__` (pydantic, jinja2, pyyaml), so the
  contract alone does not prove portability. `tests/contract/test_learn_standalone.py` copies
  `src/llm_ports/learn/` into a temporary directory under an empty `llm_ports/__init__.py`
  (no import is rewritten) and runs it in a subprocess whose `sys.modules` must then hold no
  third-party module: the "lift it into another codebase" claim, executed.
- `tests/contract/test_core_imports.py` (the `core-only` CI job, a bare install) walks `llm_ports.learn`
  with the rest of the core and asserts it loads no vendor module; `examples/learning_loop.py` runs on every pytest leg.
- The vocabulary pre-commit hooks already reject consumer and deployment names in
  `src/llm_ports/`, and market and deployment names across the docs, tests and examples too.
- Docs say how to plug in a ledger: implement two methods.

## Telemetry (what fails silently, where it lands, what watches it)

| silent failure | lands in | watched from outside by |
|---|---|---|
| a loop stops running | no run-table row / no work row in window | the stalled-loop watchdog (exists) via the derived score slot; MISSED flag on the schedule view |
| a loop runs but cannot decide (data gone, store down) | `metrics.learn_verdict.decision = could_not_run` with reason | the research ledger shows `error`; a gauge on the service's watched tick (every 5 min) computes `learn_could_not_run_streak{kind}` from the ledger, independent of any page being open; alert at ≥ 3. An in-service gauge dies with the service, so it exports `learn_gauge_updated_ts` and the operations stack alerts on its staleness (`absent()` or age > 15 min): the outside watcher for this row and the next two |
| a loop never earns: stuck at `insufficient` forever (the sample never grows) | `learn_verdict.decision = insufficient`, every run | the same gauge exports `learn_days_insufficient{kind}` (days since the kind's first consecutive `insufficient`); alert at ≥ 30 days, so a floor that cannot be reached is named rather than waited on |
| an `apply` fails after a pass | `metrics.learn_apply = apply_failed` | the gauge's `learn_apply_failed_total{kind}`; alert > 0 |
| a gate silently passes on missing evidence | impossible by construction: None fails closed | unit tests of `decide` both directions (a pass and every fail mode) |
| a ledger write fails | the consumer's stamp helper swallows it today | `run()` returns the ledger outcome; the consumer's runner counts `learn_ledger_write_failed_total{kind}`; alert > 0 |
| a lifted statistic changes a number | n/a | the consumer's existing tests run against the lifted function before the copy is deleted |

## Slices (PRs), in order

1. **llm-ports: `llm_ports.learn` core** (types, gate, stats, examples, stores, loop runner with
   its `apply` stage, standalone contract, guide, ADR-0010, API page) → release 1.5.0.
2. **llm-ports: LLM labels on the core**: re-scope WIP `04678fd` (below) → 1.6.0.
3. **The consumer: adoption plumbing**: pin; its ledger adapter; `FitSpec` with `learn_kind` +
   `register_loop` (stamps only non-loop specs); `learn_kinds()` (declared UNION observed); the
   research ledger reads `metrics.learn_verdict` generically (and its rigor read); derived score
   slot; its BY-default `bh_fdr` copy replaced by `learn.stats.bh_fdr` with its DEFAULT (BY:
   numbers unchanged for its 8 callers; `dependent=False` would loosen the signal-trust gate) and
   its comments that call it Benjamini-Hochberg corrected (not the sealed gate's copy, not the
   q-value helper); the learn gauge and its staleness timestamp.
4. **The consumer, its UI and its operations stack: schedule view**: the schedule endpoint (the
   scheduler's schedule list, keyed by registered loop), research-ledger columns, the three
   alerts.
5. **The consumer: existing gates return `Verdict`s**: the promotion gate, the
   promote/hold/reject gate, fit `earned()`s, judge trust; sealed-gate wrappers. Thresholds
   unchanged.
6. **The consumer: LLM label loop** (original Step 4): queue, coverage, LABELS card (§LLM labels).
7. **The consumer: the trend loop**, first new consumer loop, weight 0, display-only on the
   research ledger; measures the adoption budget. Reads the consumer's daily-series view joined to
   the calendar; forward outcomes resolved per the consumer's own fill protocol. A roadmap line in
   the consumer's plans when this doc lands.
8. **Fan-out**: judges read `few_shot`; the assistant's similar-cases adapter; remaining loops.

Slices 1-2 are core, 3-5 make it easy, 6-8 go wide: the operator's order.

## LLM labels (one loop kind)

`EVAL-LABELS-CONSOLIDATION.md` holds the inventory and the label decisions; they stand with these
changes:

- `ports.Label` / `ports.LabelStore` (WIP `04678fd`) are NOT added: a label is a `learn.Example`
  (topic `llm:<role>`, `evals.labels.topic`; source human / model / judge; the trace id in
  `ref["trace"]`) read through `learn.ExampleStore`. `evals.labels` keeps only what is
  LLM-specific: `topic` / `role_of`, `draw` (stratified by the judge's verdict, keyed by the core's
  `coverage`), `label_of(JudgeVerdict, source="model"|"judge")`, and the `LabelQueue` port
  (`enqueue(topic, subjects)`, the review surface). `coverage` and `agreement` are the core's
  (`learn.examples`); the test double is `learn.stores.MemoryExampleStore`, not a new fake.
- `adapters/langfuse/labels.py` is `LangfuseExampleStore`, a `WritableExampleStore` and a
  `LabelQueue` (scores `label-<role>` on the production generation, `label.model-<role>` and
  `label.judge-<role>` written by `put`, and since 1.12.0 a person's `label-<role>` relayed by
  `put` with `by`, ADR-0014; one annotation queue per role; SDK calls in a worker thread). `adapters/langfuse/experiments.py` ships as drafted. Both in slice 2 (1.6.0).
- The judge-trust gate becomes a `Loop` in the consumer (slice 6): examples = human labels from
  the store, clauses = `labelled >= 20` (floor), `kappa >= 0.6`, `seeded_bad_all_caught` (via
  `evals.calibration.verdict`, as the judge-trust `assess` does today), recorded as `run_kind =
  'llm_judge_trust'`; the judge-trusted gauge reads its latest verdict instead of a baked JSONL.
- Langfuse stays the LLM label store (UI exists, the label sits on the reply it labels, already
  backed up). A replay read from a Langfuse export was cut at 4,000 characters (sibling notes
  2026-11-07; source of the cut UNVERIFIED): the queue job records the length it enqueued and the
  label loop refuses a label whose stored reply length differs.

## Corrections to the inputs

- The training-engine inventory said the judge-trust gate was not on main. It is: it merged
  before this revision, a 109-line module.
- The prior reviewer's findings file was empty on disk; its findings are not recoverable. This
  doc got a fresh review (DO NOT SHIP, 11 findings); this revision answers each: 1 →
  `learn_verdict` (Decision 3), 2 + 3 → `FitSpec.learn_kind`, the scheduler's schedule list
  (Decision 6), 4 → lift only kappa/AC1/BY (Decision 5), 5 → standalone lifted-copy test, other
  verdict types named, 6 → roles + `unevaluated` policy, 7 → `apply` stage, 8 → scrape-time
  collector, 9 → async stated, 10 → cross-sectional + budget scope, 11 → inventory rows, event
  tools, sibling docs, Postgres adapter, view keyed by loop.

## Not in scope

Training or fine-tuning runs; a new judge model choice (measured separately); MLflow; editing
either sealed gate; moving any fit usecase out of the consumer; a new UI page.
