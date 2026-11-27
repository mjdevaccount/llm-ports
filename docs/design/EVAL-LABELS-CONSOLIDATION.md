# One system for LLM evaluation, judging and labelled data

Status: SUPERSEDED in scope by `LEARNING-LOOP-CORE.md` (2026-11-07), which owns the decisions; this file stays as the detail of the LLM-label loop, amended there. Originally DESIGN, 2026-11-07. Scope: every codebase in the first consumer's deployment that grades, judges or labels LLM
output. Owner of the decision: this file. The inventory was read from each codebase's main branch
with `git ls-files` / `git grep` and from the deployment host (the Langfuse API and the consumer's
eval-run table) on 2026-11-07 ~04:40Z.

## Why

The deployed judge (a small self-hosted model) agrees with a human at kappa ~0.19 on a 20-reply
hand-graded set and fails 10 of 14 replies a human graded good. No judge, small or large, can be
shown calibrated per role on that set: those 20 replies are almost the whole human-labelled
corpus, about two per role across 34 registered roles. Models will not get better here until the
labelled data grows, and that data is spread over eight homes in three codebases. The work is: one
system first, then grow the labels on it.

## The decision

**One system, three parts, no second harness:**

| part | home | holds |
|---|---|---|
| mechanics | **llm-ports** `llm_ports.evals` (+ its ports and adapters) | everything that is not about one consumer: judging, golden runs, calibration statistics, the deterministic board, and (new) the labelled-example model, coverage and the label store port |
| storage + labelling UI | **Langfuse** on the deployment host (v4.72.0, already deployed, already holds traces, judge scores and 3 datasets) | every label: verdict labels as scores on the production generation, reference outputs as dataset items |
| configuration | **the consumer application** (its evals package and its judge rules) | the consumer's roles, rubrics, graders, golden pack data, which roles get queued and how many, the jobs and gauges that run it |

"We already have one" is right: llm-ports 1.4.0 already holds the judge (`evals.judge`, with
`JudgeLane`), golden-set runs (`evals.golden`), calibration (`evals.calibration`: kappa, AC1,
seeded-bad replies), the deterministic board (`GauntletEvaluator`), the judge adapter
(`adapters.pydantic_evals`) and replay recording (`adapters.pydantic_ai.replay`); the consumer
pins it (`==1.4.0`). What the library does NOT have is any notion of a **label**: nothing models
"a human (or a stronger model) said this reply is good", nothing counts labels per role, nothing
puts a reply in front of a labeller. That gap is the whole build.

## Inventory

### llm-ports (the one) — keep, extend

| piece | what it does | verdict |
|---|---|---|
| `evals/judge.py` | `Seen`, `judge_input`, `rubric` (4 laws), `sample_key`/`selected` 1-in-N draw, `JudgeLane` background judging off `Hooks.on_ok` | keep |
| `evals/golden.py` | fire a pack's golden items through `call.generate`, grade on one board | keep |
| `evals/calibration.py` | `cohen_kappa`, `gwet_ac1`, `seed_placeholder`, `seed_figure`, `tally`, `verdict` (exit 0/1/2) | keep; `labels.agreement` calls it |
| `evals/_gauntlet.py` | `GauntletEvaluator`: deterministic graders over rows into one board | keep |
| `evals/refute.py` | claim refutation | keep |
| `adapters/pydantic_evals.py` | `PydanticEvalsJudge` (the `Judge` port over pydantic-evals' LLM judge) | keep; also the strong-model labeller |
| `adapters/pydantic_ai/replay.py` | record / replay model traffic | keep |
| `adapters/langfuse/` | tracer + prompt source | keep; gains `labels.py` |
| `docs/guides/judge-and-refute.md` | judge guide | keep; links the new labels guide |

### The consumer application — its configuration stays, generic duplicates go

The earlier slice (llm-ports#39, 1.4.0) already switched most of it: the consumer's judge runs
`JudgeLane`, its pack-golden runner delegates to `golden`, its kappa and calibration scripts and
its judge-trust gate call `calibration`, and its recorded model calls `replay.recording`. What is
left:

**Generic, still in the consumer -> moves to llm-ports, then deleted (PR 1 + PR 2):**

| piece | size | what it does | goes to |
|---|---|---|---|
| a Langfuse experiment helper | 76 lines | records a graded eval as a Langfuse dataset experiment (`run_experiment`), exits 2 when the SDK silently dropped an item | `llm_ports.adapters.langfuse.experiments` (`run`, `record_rows`, a dropped item = a named outcome, not a shrunken ratio). Its three eval-script callers import it from there |
| a local `citations_resolve` grader | ~15 lines | the cited-keys-resolve grader | `golden.citations_resolve` (already in llm-ports, same contract) |

**Dead or duplicated -> deleted (PR 2):**

| piece | why |
|---|---|
| a promptfoo harness (config, dataset builder, a provider shim, asserts, cases, prompts) | a second harness (Node) for one role; nothing invokes it (no Makefile, CI or scheduler reference); hard-codes a host IP. That role is covered by the call graders on every reply and by the sampled judge |
| a triage bake-off script | its own copy of the triage scorer; nothing invokes it |
| a citation spike script | scratch spike with its own pydantic-evals judge; nothing invokes it |
| an unlabelled sample file | a copy (every `gold` null) of the extraction sample; nothing reads it |

**Consumer-specific -> stays as configuration and data (graders, rubrics, golden sets, schedules):**

| piece | what it is |
|---|---|
| the judge wiring, its rules, its trust gate and the judge adapter | which roles, time-window eligibility, rubric fill-in, the row written, the trust gate. The trust gate reads a calibration JSONL at runtime today; PR 3 points it at `labels.agreement` over the store |
| per-surface graders, the pack-golden runner, the recorded model | per-surface graders and wiring over the llm-ports mechanics |
| the eval runs, their routers and repositories | the scheduled runs (a weekly prompt eval, a daily gauntlet, a weekly extraction eval) and their rows in the consumer's eval-run and run-ledger tables |
| a weekly prompt-eval script | builds Langfuse datasets `topic_gold_v1`, `triage_gold_v1`, `mention_gold_v1` from real inputs, re-runs them through Gemini, scores json/schema/accuracy; accuracy uses expected outputs "hand-labelled in the Langfuse UI" — the label store this design adopts, already in use for three roles |
| golden-set builders and replays (about a dozen scripts) | build or replay the consumer's golden sets |
| a faithfulness-threshold calibrator | calibrates the HHEM faithfulness model's threshold (its own torch env). A threshold-vs-label calibration; kept in the consumer because it needs torch. Noted as a candidate for `calibration` later |
| two calibration CLIs | thin CLIs over `calibration` for the review judge and the sampled judge |
| golden JSONL files, pack `golden_v1.json` files, golden test fixtures | data (labels move per "Where labels live" below; golden INPUTS stay in git for CI) |
| the operator's review-label endpoint and its table | the operator's existing labelling surface for post-hoc reviews |

**Not LLM evals** (named by the grep, out of scope): retrieval recall/MRR and the embed-space
parity gate (it already uses `llm_ports.rag.metrics`), shadow re-projection evals, a Langfuse
parity check, and the Langfuse cost reads.

### A local model host's repository

| piece | what it does | verdict |
|---|---|---|
| a goldens file (19 rows), a scorer, a runner, a README | stdlib-only eval of the host's research middleware (`-web` tier): live-figure fidelity, respects handed ground truth, cites a URL, no filler, tool routing read from the host's own log. Last commit 2026-07-31; nothing invokes it | FOLD (PR 6): the 19 goldens become the Langfuse dataset `golden:research.web`; the output checks become graders run by `GauntletEvaluator` over the web roles' production replies; the tool-routing check (it reads the host's own log) stays a serving probe in that repository, not an eval. The eval directory is deleted with a pointer |

### The operations stack

| piece | verdict |
|---|---|
| Langfuse v4.72.0, backed up by the stack's backup job | THE label store. No change but one alert rule (below) |
| a judge-verdict-fail alert, gated by a judge-trusted gauge | keep; the trust gauge becomes computable from labels (`labels.agreement`), not a hand flag |

### Langfuse, live (2026-11-07)

`/api/public/health` = 4.72.0. Datasets: 3 (`summary_golden_v1`, `triage_gold_v1`,
`topic_gold_v1`). Annotation queues: **0**. Score configs: **0**. The SDK in the service image
(langfuse 5.1.0) exposes `annotation_queues`, `score_configs`, datasets and scores. The labelling
surface exists and is unused.

### Labelled data that exists today (2026-11-07)

| store | shape | n | role(s) |
|---|---|---|---|
| the eval-run table, rows `pack='judge:<role>'` | MACHINE verdicts (the small judge) | 1,087 over 20 roles (pass 25) | extract.article 734, vision.describe 97, research.web 63, review.postmortem 48, ... |
| a judge-calibration JSONL | HUMAN verdict on a reply (with Langfuse span id), beside seeded-bad rows | 51 lines, 20 human verdicts | 11 roles, ~2 each |
| the operator's review table | HUMAN per-field truth about a reviewed decision (its own UI) | 50 | review.postmortem |
| a labelled extraction JSONL | HUMAN/reference expected fields | 203 | extract.article |
| Langfuse datasets `triage_gold_v1`, `topic_gold_v1`, `summary_golden_v1` | reference items, expected outputs hand-labelled in the Langfuse UI | (counted in PR 3) | triage.mention, triage.topic, summary.note |
| pack `golden_v1.json` files | golden INPUTS (slots), no labels | 3 packs | review.decision, summary.note, review.postmortem |
| a triage golden JSONL | HUMAN expected flag + calls | 18 | triage.mention |
| Langfuse `brief_golden_v1`, `mention_gold_v1` | reference items | (counted in PR 3) | brief.*, extract.mentions |

Human verdict labels on LLM replies: **~20**. That is the number this design exists to raise.

## Where labels physically live, and why Langfuse

Two label shapes, one store:

1. **Verdict labels** (was this reply good?) are Langfuse **scores** on the production
   generation observation. Human: score name `label-<role>`, source `ANNOTATION`, written in
   the Langfuse annotation-queue UI. Strong model: the same name, source `API`, metadata
   `{"labeller": "model", "by": "<model id>"}`. The deployed judge keeps its existing
   `<role>.llm_judge` score. One categorical score config per role (`good` / `bad`), one
   annotation queue per role (`label-<role>`).
2. **Reference outputs** (the right answer for this input) are Langfuse **dataset items** with
   `expected_output`, one dataset per role named `golden:<role>`, `source_observation_id` set when
   the item came from production. The five existing datasets keep their names (see PR 2 notes).

Why Langfuse and not a table or jsonl:

- **The UI already exists.** Annotation queues are the labelling surface: the reply, its prompt,
  its system prompt and images are already on the trace, side by side with the score form. A table
  would need that UI built; jsonl has no UI at all.
- **The label sits on the thing labelled.** A score is keyed by observation id, so a label can
  never drift from the reply it judges, and the judge's own score sits beside it for agreement.
- **It is already the store**: traces, judge scores and three datasets live there, three of the
  consumer's scripts already write datasets, and it is backed up by the operations stack.
- **Against, and the answer:** CI cannot reach Langfuse. CI does not need labels: CI runs the
  deterministic graders over golden INPUTS (`golden_v1.json`, in git). Labels feed measurement
  on the deployment host (agreement, coverage), which runs where Langfuse is.
- **Against, and the answer:** a judge replay read a reply cut at exactly 4,000 characters from a
  Langfuse export (judge-failure notes, 2026-11-07; the cut's source is UNVERIFIED — the library
  tracer does not cut outputs, `adapters/langfuse/tracer.py`). The queue job therefore enqueues
  the GENERATION observation and records the output length it saw; PR 3 verifies on the host that a
  queued reply longer than 4,000 characters reads back whole before any label counts.
- **Stays outside, by design:** the operator's review table labels the DECISION reviewed (was it
  timely, was its size appropriate), the operator's record with its own UI. It is domain truth, not
  a verdict on a reply. Coverage COUNTS it for `review.postmortem` through a second, read-only
  source; it is not copied.

## What llm-ports gains (PR 1, release 1.5.0)

- `llm_ports.evals.labels` (pure, `evals` layer):
  - `Label`: role, subject (observation id), value (`good` / `bad`), labeller (`human` / `model` /
    `judge`), by, reason, at.
  - `coverage(labels, roles, *, target)` -> per role: human / model / judge counts, target, gap.
    Every role in `roles` appears, a zero row included: the gap is the point.
  - `draw(candidates, *, have, target, per_run)` -> which unlabelled production outputs to queue
    next, per role, deterministic (`judge.sample_key`), **stratified by the judge's verdict** so
    both classes are present: a kappa needs both, and a judge that fails 93% would otherwise fill
    the queue with fails.
  - `agreement(labels, *, against="human")` -> per role `{n, kappa, ac1}` for judge-vs-human and
    model-vs-human, through `calibration.cohen_kappa` / `gwet_ac1`. A statistic the data cannot
    define is None (calibration's rule).
  - `label_of(verdict, *, role, subject, by)`: a `JudgeVerdict` from any `Judge` (the strong-model
    labeller is just another `Judge`) as a model `Label`.
- Port `LabelStore` (`ports`): `enqueue(role, subjects)`, `labels(since=)`, `put(label)`. Never
  raises; each returns a named outcome (ADR-0001), e.g. `could_not_reach` with the reason.
- Adapter `adapters/langfuse/labels.py` `LangfuseLabelStore`: get-or-create the per-role score
  config and annotation queue, add observations to the queue, write model labels as API scores,
  read `label-<role>` and `<role>.llm_judge` scores back as `Label`s.
- `adapters/langfuse/experiments.py`: `run` / `record_rows`, moved from the consumer's experiment
  helper (record an offline-graded eval as a dataset experiment; an item the SDK dropped is a
  named outcome, never a shrunken ratio).
- `testing.FakeLabelStore`; guide `docs/guides/labels.md`; ADR-0010 "labels are scores on the
  labelled observation"; API page; CHANGELOG.

Not in llm-ports: which roles, the targets, the Gemini budget, the schedule. Those belong to the
consumer.

## What the consumer does (PRs 2-4)

- **PR 2 (consolidate, no behaviour change):** import `experiments` and `golden.citations_resolve`
  from llm-ports; delete the local experiment helper, the local `citations_resolve`, the promptfoo
  harness, the bake-off and spike scripts and the unlabelled sample; drop their tests where
  llm-ports now carries them. Prove: the existing suites green, `wc -l` before / after.
- **PR 3 (labels):** below.
- **Rename into one scheme:** the Langfuse datasets `triage_gold_v1` / `topic_gold_v1` /
  `mention_gold_v1` / `summary_golden_v1` / `brief_golden_v1` keep their names (renaming a
  Langfuse dataset loses its run history); the scheme `golden:<role>` applies to datasets created
  from here on, and the coverage reader maps the five legacy names to their roles explicitly.

## Labelled-data expansion (PR 3-5)

- **The queue** (a nightly scheduled job behind an idempotent run endpoint): per registered role,
  read the last 7 days of generations from Langfuse, `labels.draw` up to `per_run` (default 5) to
  close the gap to `target` (default 50 human per role), `LabelStore.enqueue`. The operator labels
  in the Langfuse UI, five minutes at a time, a role at a time.
- **The cheap labeller**: the same job then asks a strong model to label up to its daily budget
  (Gemini free tier: 20 requests/day/model, measured 2026-11-07; or a larger self-hosted model in
  an A/B window, which evicts the small judge) through `PydanticEvalsJudge` + `labels.label_of`
  -> `put`. Model labels are counted SEPARATELY and never pass for human ones; `agreement`
  measures the strong model against the human labels before any of its labels are trusted for a
  role.
- **Seed the store** once from what exists: the judge-calibration JSONL (20 human verdicts with
  span ids) -> `label-<role>` scores; the labelled extraction JSONL (203) -> dataset
  `golden:extract.article`; then the jsonl files are deleted and their readers switched.
- **Coverage** (a read endpoint): `labels.coverage` over the consumer's role registry (34 roles)
  + the operator's review table for `review.postmortem`, and `labels.agreement` per role.

## Telemetry (what fails silently, where it lands, what watches it)

| silent failure | lands in | watched from outside by |
|---|---|---|
| queue job stops running | `llm_label_job_last_success_timestamp` stops moving | a Prometheus staleness alert (>36h) + the consumer's loop registry: the job is registered as a learning loop, so the registry's stalled-loop watchdog reads it |
| job runs but enqueues nothing / Langfuse refuses | `llm_label_enqueued_total{role}` flat, outcome reason on the job row | same alert reads the per-run outcome; a `could_not_reach` outcome exits 2, never 0 |
| labels stop arriving (nobody labels) | `llm_labels{role,labeller}` flat | visible on the UI's LABELS card (gap per role); not paged: an unlabelled day is the operator's choice |
| a judge drifts from humans | `llm_judge_agreement{role}` | the existing judge-trusted gate reads it instead of a hand flag |

## UI

The consumer's LLM page gains a LABELS card beside its existing gauntlet boards: per role human /
model / judge counts, gap to target, judge-vs-human kappa when defined, and a link to the role's
Langfuse annotation queue. The labelling itself happens in Langfuse's queue UI (no new UI is
built for it).

## PR plan

1. llm-ports: `evals.labels`, `LabelStore` port, `LangfuseLabelStore`, `langfuse.experiments`, fake, guide, ADR-0010 ->
   release 1.5.0.
2. The consumer: switch every generic duplicate to llm-ports and delete it (table above).
3. The consumer: pin 1.5.0; label queue job + coverage endpoint + gauges + loop registration; seed
   the store; the scheduled job.
4. The operations stack: the label-job staleness alert (+ a promtool case).
5. The consumer's UI: LABELS card.
6. The local model host: fold its eval directory (dataset + graders in the consumer), keep the
   routing probe.

### Deviations as built

- **The web output checks became CALL graders, not consumer graders.** "Cites a URL" and "no
  refusal" hold for any research tier, so they are generic: `cites_url` and `no_refusal` in
  `llm_ports.call.graders` (1.8.0), with `cites_url` in `WEB_ONLY_GRADERS` so a call without
  `web=True` drops it. Scoring every production reply at call time beats an offline
  `GauntletEvaluator` pass over a sample, and the consumer's existing grader-failure-rate rule
  watches it with no new alert. The consumer's seven web structured roles carry them; before
  enabling, 14 days of live replies were measured: `research.web` 119/119 cited a URL,
  `event.brief` 0/9 (every stored brief has `sources = []`), a defect the grader now surfaces.
  "No filler" was not generic enough to move; it stays a consumer grader candidate.
- **Golden-set seeding is `experiments.seed` (1.9.0)**, idempotent by item id, replacing the
  consumer's three hand-rolled dataset-item loops.
- **Label names use `-`, not `:` (1.9.1).** Langfuse refuses a score-config name outside
  letters, digits, `_`, space, `.`, `(`, `)`, `-` (1-80), and a human label's score is named after
  its config, so `label:<role>` failed the first config create with a 400 and the first queue run
  enqueued nothing. Names are now `label-<role>`, `label.model-<role>`, `label.judge-<role>`;
  reads accept both spellings, and the adapter refuses a role whose name the server would refuse
  before writing anything.

## Not in scope

No fine-tuning or training run: the data comes first. No new judge model decision (measured
separately). The consumer's learning-loop registry is the platform's list of learning loops (of
every kind); it is not an eval system and is not merged into this one. The label loop is
REGISTERED there as one row, which is what makes it watched.
