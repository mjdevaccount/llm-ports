---
status: accepted
date: 2026-11-09
---

# Prompt tuning runs on the `gepa` core behind a caller's evaluator; the study UI is launched on demand

Decided in 1.20.0.

## Context and problem statement

A consumer wants its prompt packs to improve against a measured metric: the proper score of the
claims a pack extracts, the share it can check, a grounding gate. GEPA (reflective prompt
evolution) is the method: it scores a candidate prompt on minibatches, asks a reflection model to
rewrite one component from the feedback, and keeps rewrites that beat their parent. Two packages
ship it:

- `dspy` (MIT): `dspy.GEPA` optimizes a DSPy *program* (signatures and modules). It brings about
  forty dependencies, LiteLLM among them.
- `gepa` 0.1.4 (MIT): the algorithm alone, with no required dependency. It takes any
  `GEPAAdapter` (`evaluate(batch, candidate) -> outputs, scores, trajectories`).

A consumer's packs are templates rendered for agents with typed outputs, not DSPy programs, so
`dspy.GEPA` would mean rewriting each pack as a program first.

Separately, the walk-forward tuner (ADR-0015) keeps its Optuna study in `SqlJournal`, one table
behind a transaction-mode pooler. Stock optuna-dashboard opens an RDB URL or a journal file, not
that table, so the study had no UI.

## Decision

1. `adapters.gepa` wraps the `gepa` core. `optimize(seed, train, val, evaluate=, reflect=)` takes
   the caller's batch evaluator (one `Rollout(output, score, feedback)` per item) and a reflection
   callable (`str -> str`), and returns `Optimized | Insufficient | Unavailable`. The caller
   supplies both models, so GEPA's LiteLLM path is never used and the consumer's gateway stays
   the one egress.
2. GEPA logs a failed reflection and continues; it never raises it. `optimize` therefore counts
   every reflection call and failure. Every call failing is `Unavailable`; some failing is
   reported on `Optimized.reflection_failures`. A search that rewrote nothing because its
   reflection model was down does not look like a search that found nothing better.
3. `Optimized.improved` compares the best candidate to the seed on the validation items. The
   adapter does not promote. Promotion is the caller's decision, made through its own review
   path.
4. `adapters.optuna` gains `dashboard_app(storage)` / `serve_dashboard(storage)`, which run
   optuna-dashboard over any Optuna storage, `journal_storage(...)` included (extra
   `[optuna-dashboard]`). It runs on demand and binds loopback by default. It is not a resident
   service.

## Considered options

- **`dspy.GEPA`**: rejected for the program rewrite and the dependency weight. A consumer that
  already has DSPy installed has the `gepa` core too (DSPy depends on it).
- **A hand-written evolution loop**: rejected. GEPA's Pareto selection, minibatch acceptance and
  merge are the method.
- **Optuna RDB storage on a direct DSN, so the stock dashboard opens it**: rejected. It adds a
  second, non-pooled connection class, and Optuna's own Alembic-managed tables would sit outside
  the consumer's migration ledger.

## Consequences

- Good: a pack is tuned with no change to how it is rendered or called. The evaluator is the one
  domain-specific piece.
- Good: a down reflection model is a named outcome, not a silent zero-improvement run.
- Bad: GEPA's own checkpoint (`run_dir`) is a directory of files. A caller that needs the run
  history in its database records `Optimized` itself.
