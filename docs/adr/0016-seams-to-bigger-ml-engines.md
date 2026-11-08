---
status: accepted
date: 2026-11-08
---

# Seams to bigger ML engines: run ids, artifact URIs, a model registry port and conformance suites

Decided in 1.15.0.

## Context and problem statement

`learn` and `forecast` run on the standard library (ADR-0010, ADR-0013), and their vendor
integrations (Optuna, scikit-learn, OpenTelemetry, Langfuse) are adapters behind extras. A
consumer may later move its loops onto a full ML platform: MLflow tracking and model registry
first, then W&B, Ray Tune, or a managed platform (Vertex AI, SageMaker). Four things stopped such
an engine from plugging in as an adapter without touching the core or a consumer:

1. A ledger write returned only a count, so a run had no id an external tracker could link to.
2. A fit was JSON only; an engine produces a model file or a URI.
3. Promotion (champion / challenger) was written ad hoc inside each loop's `apply`, with no port
   a registry could implement.
4. Nothing outside a backend checked that it kept its port's contract (never raise, an
   unreadable store is not an empty one, the decision filter), so a new backend could fail open
   and look like it worked.

## Decision drivers

- The core stays standard library only; an engine is an adapter behind an extra.
- No consumer breaks: every addition is an optional, defaulted field or a new name.
- Each port maps onto the engine's own standard concept, not a lowest common denominator.
- A backend that breaks a contract must fail a check, not a production read.
- `Loop.fit` stays pure and its result plain data, so an engine can run it in a remote worker.

## Considered options

1. Adopt MLflow (or another engine) as the core's storage.
2. Leave each consumer to write its own bridge per engine.
3. Seams in the core, an engine as an adapter, and conformance suites any backend must pass.

## Decision outcome

Option 3. Option 1 would make every install carry a tracking stack and tie the loop's contract
to one vendor's data model. Option 2 is the duplication the ports exist to prevent.

- `Written.ref` and `LedgerRow.ref` (optional, not part of equality): the id a backend gave the
  row. `MemoryRunLedger` and `JsonlRunLedger` mint a UUID; MLflow's ref is the run id.
  `Run.recorded.ref` is what a registry version's `run_ref` links to.
- `Fit.artifact_uri` (optional): where a model too big for `model` lives. `loop.run` writes it to
  the row as `learn_artifact_uri`, only when set, so existing rows keep their shape.
- `ModelRegistry`: `register` (a new version, numbered from 1 per name), `set_alias`,
  `by_alias`, `get`. Its shape is MLflow's registry with aliases (a version never changes, an
  alias moves), which W&B's artifact aliases and Vertex AI's model registry aliases also have.
  `learn.registry` ships `MemoryModelRegistry` and `JsonlModelRegistry` (an append-only event
  log, so every alias move stays on the record). Existing loops are not moved onto it.
- `llm_ports.testing.conformance`: `check_run_ledger`, `check_example_store`,
  `check_model_registry` and their broken-storage twins. Each writes under fresh random names, so
  it runs against a shared or live store. Every in-repo backend runs them, and the suite's own
  tests prove it fails a backend that breaks each contract.
- `adapters.mlflow` (extra `[mlflow]`): `MlflowRunLedger` and `MlflowModelRegistry` over
  `mlflow-skinny`, passing the suites against a local SQLite store.

### Port to standard

| Port | Standard it maps onto | Notes |
|---|---|---|
| `RunLedger` | MLflow tracking runs; W&B runs | kind = experiment, subject and decision = tags, scalar params and numeric metrics logged, the whole row a JSON artifact |
| `ModelRegistry` | MLflow model registry aliases; W&B artifact aliases; Vertex AI model aliases | `source` = `artifact_uri`, the fitted JSON a version tag |
| `ExampleStore` | a feature store's offline (point-in-time) read | `since` and `source` are the time and provenance filters |
| estimators (`forecast._estimator.Params`) | the scikit-learn estimator API | so skl2onnx, joblib and Ray's sklearn integrations accept them (ADR-0013) |
| tuning (`adapters.optuna`) | Optuna studies | Ray Tune runs Optuna through `OptunaSearch` (ADR-0015) |
| traces (`adapters.otel`) | OpenTelemetry GenAI semantic conventions | |
| provenance (`provenance`) | PROV-O | ADR-0006 |

### Measured, for the MLflow adapter (mlflow-skinny 3.18.0)

- MLflow 3 refuses its file store unless `MLFLOW_ALLOW_FILE_STORE=true` ("maintenance mode"), and
  the skinny client registers the `sqlite://` scheme only when SQLAlchemy is importable; with
  SQLAlchemy and Alembic (both already Optuna's dependencies) it opens a local SQLite store with
  no server. The extra pins those three.
- A fresh SQLite store takes about 10 s to create (MLflow runs its migrations); a run takes
  about 50 ms afterwards.
- Run tag values over 8000 characters are truncated with a warning, not refused, so the row's
  truth is an artifact, never a tag. Search filters have no `IN` for tags, so a multi-decision
  filter is applied client-side; runs with equal start times have no stable order, so the
  ledger stamps strictly increasing milliseconds.

### Consequences

- Good: the next engine (W&B, a managed registry) is one adapter module and a green
  conformance run; no consumer changes.
- Good: a backend that fails open fails a named invariant in CI.
- Bad: `JsonlModelRegistry` reads its version number from the file; two writers on one file can
  register the same number (one writer per file).
- Bad: two `MlflowRunLedger` objects writing one kind in the same millisecond have no defined
  order between those two rows.
- Neutral: `Fit.model` is still the core's model format; an engine model lives at
  `artifact_uri` and the core never loads it.
