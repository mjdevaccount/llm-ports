# One system for LLM evaluation, judging and labelled data

Status: DESIGN, 2026-11-07. Scope: every repo of the estate that grades, judges or labels LLM
output. Owner of the decision: this file. Inventory read from each repo's `origin/main` with
`git ls-files` / `git grep` and from the box (Langfuse API, `quant.prompt_eval_run`) on
2026-11-07 ~04:40Z.

## Why

The box judge (a 9B model) agrees with a human at kappa ~0.19 on a 20-reply hand-graded set and
fails 10 of 14 replies a human graded good. No judge, small or large, can be shown calibrated per
role on that set: those 20 replies are almost the whole human-labelled corpus, about two per role
across 34 registered roles. Models will not get better here until the labelled data grows, and
that data is spread over eight homes in three repos. The work is: one system first, then grow the
labels on it.

## The decision

**One system, three parts, no second harness:**

| part | home | holds |
|---|---|---|
| mechanics | **llm-ports** `llm_ports.evals` (+ its ports and adapters) | everything that is not about EDGE: judging, golden runs, calibration statistics, the deterministic board, and (new) the labelled-example model, coverage and the label store port |
| storage + labelling UI | **Langfuse** on the box (v4.72.0, already deployed, already holds traces, judge scores and 3 datasets) | every label: verdict labels as scores on the production generation, reference outputs as dataset items |
| configuration | **trade-data** `service/evals/` + `service/platform/llm_judge_rules.py` | EDGE's roles, rubrics, graders, golden pack data, which roles get queued and how many, the jobs and gauges that run it |

"We already have one" is right: llm-ports 1.4.0 already holds the judge (`evals.judge`, with
`JudgeLane`), golden-set runs (`evals.golden`), calibration (`evals.calibration`: kappa, AC1,
seeded-bad replies), the deterministic board (`GauntletEvaluator`), the judge adapter
(`adapters.pydantic_evals`) and replay recording (`adapters.pydantic_ai.replay`); trade-data pins
it (`pyproject.toml:50`, `==1.4.0`). What the library does NOT have is any notion of a **label**:
nothing models "a human (or a stronger model) said this reply is good", nothing counts labels per
role, nothing puts a reply in front of a labeller. That gap is the whole build.

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

### trade-data — EDGE configuration stays, generic duplicates go

Read from trade-data `origin/main` 6e941731. The earlier slice (llm-ports#39, 1.4.0) already
switched most of it: `service/platform/llm_judge.py:89` runs `JudgeLane`,
`service/evals/pack_golden.py:33` delegates to `golden`, `scripts/judge_kappa.py:32`,
`scripts/llm_judge_calibrate.py:52` and `service/platform/llm_judge_trust.py:41` call
`calibration`, `service/evals/recorded_model.py:19` calls `replay.recording`. What is left:

**Generic, still in trade-data -> moves to llm-ports, then deleted (PR 1 + PR 2):**

| piece | lines | what it does | goes to |
|---|---|---|---|
| `scripts/_langfuse_run.py` | 76 | records a graded eval as a Langfuse dataset experiment (`run_experiment`), exits 2 when the SDK silently dropped an item | `llm_ports.adapters.langfuse.experiments` (`run`, `record_rows`, a dropped item = a named outcome, not a shrunken ratio). Callers `day_brief_eval.py`, `narrator_eval.py`, `eval_prompts.py` import it from there |
| `service/evals/autopsy_graders.py` `citations_resolve` (:84) | ~15 | the cited-keys-resolve grader | `golden.citations_resolve` (already in llm-ports, same contract) |

**Dead or duplicated -> deleted (PR 2):**

| piece | why |
|---|---|
| `evals/` (promptfoo: `promptfooconfig.yaml`, `gen_dataset.py`/`.sql`, `providers/box.js`, `asserts/`, `cases/`, `prompts/`, `README.md`) | a second harness (Node) for `post.axes`; nothing invokes it (no Makefile, CI or Windmill reference); hard-codes a box IP. `post.axes` is covered by the call graders on every reply and by the sampled judge |
| `scripts/triage_bakeoff.py` | its own copy of the triage scorer (:81, duplicates `scripts/_triage_score.py`); nothing invokes it |
| `scripts/ai_spike_cite.py` | scratch spike with its own pydantic-evals judge (:208); nothing invokes it |
| `docs/churn_golden_sample.jsonl` | an unlabelled copy (every `gold` null) of the churn sample; nothing reads it |

**EDGE-specific -> stays as configuration and data (graders, rubrics, golden sets, schedules):**

| piece | what it is |
|---|---|
| `service/platform/llm_judge.py`, `llm_judge_rules.py`, `llm_judge_trust.py`, `adapters/pydantic_evals_judge.py` (`BoxJudge`) | the judge wiring: which roles, RTH eligibility, rubric fill-in, the row written, the trust gate. `llm_judge_trust` reads `judge_calibration.jsonl` at runtime today; PR 3 points it at `labels.agreement` over the store |
| `service/evals/{autopsy,dossier,day_brief,narrator}_graders.py`, `pack_golden.py`, `recorded_model.py` | per-surface graders and wiring over the llm-ports mechanics |
| `service/usecases/{prompt_eval,llm_gauntlet,churn_eval}.py`, routers `prompt_eval_router.py`, `churn_eval_router.py`, `repositories/{prompt_eval_repo,evals_repo}.py` | the runs (Windmill `prompt_eval` Sat 17:40, `llm_gauntlet` daily 17:20, `churn_eval` Sun 06:30) and their rows (`quant.prompt_eval_run`, `quant.learn_runs`) |
| `scripts/eval_prompts.py` (Windmill `eval_prompts`, Mon 06:00) | builds Langfuse datasets `macro_gold_v1`, `triage_gold_v1`, `mention_gold_v1` from real inputs, re-runs them through Gemini, scores json/schema/accuracy; accuracy uses expected outputs "hand-labelled in the Langfuse UI" — the label store this design adopts, already in use for three roles |
| `scripts/{day_brief,narrator,pack}_golden.py`, `harvest_golden_llm.py`, `day_brief_eval.py`, `narrator_eval.py`, `day_brief_plant.py`, `churn_golden_sample.py`, `triage_eval_golden.py`, `triage_golden_replay.py`, `_triage_score.py` | build or replay EDGE golden sets |
| `scripts/day_brief_calibrate.py` | calibrates the HHEM faithfulness model's threshold (its own torch env). A threshold-vs-label calibration; kept EDGE because it needs torch. Noted as a candidate for `calibration` later |
| `scripts/judge_kappa.py`, `scripts/llm_judge_calibrate.py` | thin CLIs over `calibration` for the autopsy judge and the sampled judge |
| `prompts/golden/*.jsonl`, `prompts/packs/*/golden_v1.json`, `docs/triage_golden.jsonl`, `tests/golden/**` | data (labels move per "Where labels live" below; golden INPUTS stay in git for CI) |
| `POST /journal/autopsy_label/{trade_id}` (`service/admin/journal_router.py:119`) -> `quant.autopsy_calibration` | the operator's existing labelling surface for trade autopsies |

**Not LLM evals** (named by the grep, out of scope): `scripts/eval_multileg.py`,
`scripts/eval_resign.py`, `scripts/reproject_shadow_eval.py`, `scripts/langfuse_parity.py`,
`scripts/rag_eval.py` (retrieval recall/MRR and the embed-space parity gate; it already uses
`llm_ports.rag.metrics`), `service/adapters/langfuse_{metrics,reads,observations}.py` (cost reads).

### mattpc-llm-stack

| piece | what it does | verdict |
|---|---|---|
| `evals/goldens.jsonl` (19 rows), `score.py`, `run_eval.py`, `README.md` | stdlib-only eval of the box's research middleware (`-web` tier): live-price fidelity, respects handed ground truth, cites a URL, no filler, tool routing read from `research.log`. Last commit 2026-07-31; nothing invokes it (`git grep` outside `evals/`: no hits) | FOLD (PR 6): the 19 goldens become the Langfuse dataset `golden:research.web`; the output checks become graders in trade-data `service/evals/web_graders.py` run by `GauntletEvaluator` over the web roles' production replies; the tool-routing check (reads the box's own log) stays a serving probe in mattpc-llm-stack under `probes/`, not an eval. `evals/` is deleted with a pointer |

### trade-infra

| piece | verdict |
|---|---|
| Langfuse v4.72.0 (`docker-compose.yml:388,446`), backed up by `infra/backup/lib_langfuse.sh` | THE label store. No change but one alert rule (below) |
| `LlmJudgeVerdictFail` alert, gated by `edge_llm_judge_trusted` (trade-infra#815) | keep; the trust gauge becomes computable from labels (`labels.agreement`), not a hand flag |

### Langfuse, live (box API, 2026-11-07)

`/api/public/health` = 4.72.0. Datasets: 3 (`narrator_golden_v1`, `triage_gold_v1`,
`macro_gold_v1`). Annotation queues: **0**. Score configs: **0**. The SDK in the service image
(langfuse 5.1.0) exposes `annotation_queues`, `score_configs`, datasets and scores. The labelling
surface exists and is unused.

### Labelled data that exists today (box + repos, 2026-11-07)

| store | shape | n | role(s) |
|---|---|---|---|
| `quant.prompt_eval_run` `pack='judge:<role>'` | MACHINE verdicts (the 9B judge) | 1,087 over 20 roles (pass 25) | churn.extract 734, narrative.vision 97, narrative.research 63, trade.autopsy 48, ... |
| `prompts/golden/judge_calibration.jsonl` | HUMAN verdict on a reply (with Langfuse span id), beside seeded-bad rows | 51 lines, 20 human verdicts | 11 roles, ~2 each |
| `quant.autopsy_calibration` | HUMAN per-field truth about a trade (Journal UI) | 50 | trade.autopsy |
| `prompts/golden/churn_golden_labeled.jsonl` | HUMAN/reference expected fields | 203 | churn.extract |
| Langfuse datasets `triage_gold_v1`, `macro_gold_v1`, `narrator_golden_v1` (live API: 3 datasets) | reference items, expected outputs hand-labelled in the Langfuse UI | (counted in PR 3) | triage.mention, triage.macro, narrator.note |
| `prompts/packs/*/golden_v1.json` | golden INPUTS (slots), no labels | 3 packs | decision.autopsy, narrator.note, trade.autopsy |
| `docs/triage_golden.jsonl` | HUMAN expected flag + calls | 18 | triage.mention |
| Langfuse `day_brief_golden_v1`, `mention_gold_v1` | reference items | (counted in PR 3) | day_brief.*, deep_extract.mentions |

Human verdict labels on LLM replies: **~20**. That is the number this design exists to raise.

## Where labels physically live, and why Langfuse

Two label shapes, one store:

1. **Verdict labels** (was this reply good?) are Langfuse **scores** on the production
   generation observation. Human: score name `label:<role>`, source `ANNOTATION`, written in
   the Langfuse annotation-queue UI. Strong model: the same name, source `API`, metadata
   `{"labeller": "model", "by": "<model id>"}`. The deployed judge keeps its existing
   `<role>.llm_judge` score. One categorical score config per role (`good` / `bad`), one
   annotation queue per role (`label:<role>`).
2. **Reference outputs** (the right answer for this input) are Langfuse **dataset items** with
   `expected_output`, one dataset per role named `golden:<role>`, `source_observation_id` set when
   the item came from production. The five existing datasets keep their names (see PR 2 notes).

Why Langfuse and not a table or jsonl:

- **The UI already exists.** Annotation queues are the labelling surface: the reply, its prompt,
  its system prompt and images are already on the trace, side by side with the score form. A table
  would need that UI built; jsonl has no UI at all.
- **The label sits on the thing labelled.** A score is keyed by observation id, so a label can
  never drift from the reply it judges, and the judge's own score sits beside it for agreement.
- **It is already the store**: traces, judge scores and three datasets live there, scripts already
  write datasets (`scripts/day_brief_eval.py:102`, `scripts/narrator_eval.py:50`,
  `scripts/eval_prompts.py:266`), and it is backed up (`trade-infra infra/backup/lib_langfuse.sh`).
- **Against, and the answer:** CI cannot reach Langfuse. CI does not need labels: CI runs the
  deterministic graders over golden INPUTS (`golden_v1.json`, in git). Labels feed measurement
  on the box (agreement, coverage), which runs where Langfuse is.
- **Against, and the answer:** a judge replay read a reply cut at exactly 4,000 characters from a
  Langfuse export (llm-judge-fail notes, 2026-11-07; the cut's source is UNVERIFIED — the library
  tracer does not cut outputs, `adapters/langfuse/tracer.py`). The queue job therefore enqueues
  the GENERATION observation and records the output length it saw; PR 3 verifies on the box that a
  queued reply longer than 4,000 characters reads back whole before any label counts.
- **Stays outside, by design:** `quant.autopsy_calibration` labels the TRADE (did the entry chase,
  was size appropriate), the operator's record with its own Journal UI. It is domain truth, not a
  verdict on a reply. Coverage COUNTS it for `trade.autopsy` through a second, read-only source;
  it is not copied.

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
  read `label:<role>` and `<role>.llm_judge` scores back as `Label`s.
- `adapters/langfuse/experiments.py`: `run` / `record_rows`, moved from trade-data
  `scripts/_langfuse_run.py` (record an offline-graded eval as a dataset experiment; an item the
  SDK dropped is a named outcome, never a shrunken ratio).
- `testing.FakeLabelStore`; guide `docs/guides/labels.md`; ADR-0010 "labels are scores on the
  labelled observation"; API page; CHANGELOG.

Not in llm-ports: which roles, the targets, the Gemini budget, the schedule. Those are EDGE.

## What trade-data does (PRs 2-4)

- **PR 2 (consolidate, no behaviour change):** import `experiments` and `golden.citations_resolve`
  from llm-ports; delete `scripts/_langfuse_run.py`, the local `citations_resolve`, `evals/`,
  `scripts/triage_bakeoff.py`, `scripts/ai_spike_cite.py`, `docs/churn_golden_sample.jsonl`; drop
  their tests where llm-ports now carries them. Prove: the existing suites green, `wc -l` before /
  after.
- **PR 3 (labels):** below.
- **Rename into one scheme:** the Langfuse datasets `triage_gold_v1` / `macro_gold_v1` /
  `mention_gold_v1` / `narrator_golden_v1` / `day_brief_golden_v1` keep their names (renaming a
  Langfuse dataset loses its run history); the scheme `golden:<role>` applies to datasets created
  from here on, and the coverage reader maps the five legacy names to their roles explicitly.

## Labelled-data expansion (PR 3-5)

- **The queue** (`service/usecases/llm_labels.py`, Windmill flow `llm_label_queue`, nightly
  after 21:35Z, idempotent `/admin/run/llm_label_queue`): per registered role, read the last 7
  days of generations from Langfuse, `labels.draw` up to `per_run` (default 5) to close the gap to
  `target` (default 50 human per role), `LabelStore.enqueue`. The operator labels in the
  Langfuse UI, five minutes at a time, a role at a time.
- **The cheap labeller**: the same job then asks a strong model to label up to its daily budget
  (Gemini free tier: 20 requests/day/model, measured 2026-11-07; or llm-box `qwen3.6-27b` in an
  A/B window, it evicts the 9B) through `PydanticEvalsJudge` + `labels.label_of` -> `put`. Model
  labels are counted SEPARATELY and never pass for human ones; `agreement` measures the strong model
  against the human labels before any of its labels are trusted for a role.
- **Seed the store** once from what exists: `judge_calibration.jsonl` (20 human verdicts with span
  ids) -> `label:<role>` scores; `churn_golden_labeled.jsonl` (203) -> dataset
  `golden:churn.extract`; then the jsonl files are deleted and their readers switched.
- **Coverage** (`GET /llm/labels/coverage`): `labels.coverage` over the role registry
  (`service/platform/llm_roles.py`, 34 roles) + `quant.autopsy_calibration` for trade.autopsy, and
  `labels.agreement` per role.

## Telemetry (what fails silently, where it lands, what watches it)

| silent failure | lands in | watched from outside by |
|---|---|---|
| queue job stops running | `edge_llm_label_job_last_success_timestamp` stops moving | Prometheus alert `LlmLabelJobStale` (trade-infra, >36h) + the loop registry: the flow is registered as loop `llm-labels` in `docs/LEARNING-LOOPS.md`, so the `loop-stalled` watchdog reads it |
| job runs but enqueues nothing / Langfuse refuses | `edge_llm_label_enqueued_total{role}` flat, outcome reason on the job row | same alert reads the per-run outcome; a `could_not_reach` outcome exits 2, never 0 |
| labels stop arriving (nobody labels) | `edge_llm_labels{role,labeller}` flat | visible on the LLM page LABELS card (gap per role); not paged: an unlabelled day is the operator's choice |
| a judge drifts from humans | `edge_llm_judge_agreement{role}` | the existing `edge_llm_judge_trusted` gate reads it instead of a hand flag |

## UI

trade-ui LLM page (`src/pages/Work.tsx`), GAUNTLET tab, a LABELS card beside the existing boards
(`src/edge/LlmGauntletPanel.tsx`): per role human / model / judge counts, gap to target,
judge-vs-human kappa when defined, and a link to the role's Langfuse annotation queue. The
labelling itself happens in Langfuse's queue UI (no new UI is built for it).

## PR plan

1. llm-ports: `evals.labels`, `LabelStore` port, `LangfuseLabelStore`, `langfuse.experiments`, fake, guide, ADR-0010 ->
   release 1.5.0.
2. trade-data: switch every generic duplicate to llm-ports and delete it (table above).
3. trade-data: pin 1.5.0; label queue job + coverage endpoint + gauges + loop registration; seed
   the store; Windmill flow.
4. trade-infra: `LlmLabelJobStale` alert (+ promtool case).
5. trade-ui: LABELS card.
6. mattpc-llm-stack: fold `evals/` (dataset + graders in trade-data), keep the routing probe.

## Not in scope

No fine-tuning or training run: the data comes first. No new judge model decision (a sibling
session measures Gemini 3.6 Flash). `docs/LEARNING-LOOPS.md` is the platform's registry of
learning loops (trading and LLM alike); it is not an eval system and is not merged into this one.
The label loop is REGISTERED there as one row, which is what makes it watched.
