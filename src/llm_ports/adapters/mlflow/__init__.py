"""The learning loop's run ledger and model registry on MLflow (extra `[mlflow]`).

`MlflowRunLedger(client)` records each `RunLedger.append` as one MLflow run, so loops land in
MLflow's tracking UI and API next to every other experiment:

- experiment: `prefix + kind` (created on first append; `artifact_location` sets where its
  runs' artifacts go).
- tags: `llm_ports.kind`, `llm_ports.subject`, `llm_ports.decision` (the recorded verdict's
  decision, absent on an ungated run) and `mlflow.runName`.
- params / metrics: every scalar param and every finite numeric metric whose name MLflow
  accepts. A verdict, a model or a string metric is not a number MLflow can chart, so the whole
  row is also JSON in the run's tags (`ROW_TAG/000`, `ROW_TAG/001`, ... of at most 8000 characters
  each, then `ROW_PARTS_TAG` = the count, written last), which `latest` reads back from the search
  result itself: no artifact store, no download. A row over `MAX_ROW_JSON` characters, or a
  subject over 8000 (MLflow would cut its tag short), is refused. `latest` matches the row's own
  kind, so two prefixes whose experiment names collide (`a` + `bc`, `ab` + `c`) never cross-read.
- `Written.ref` / `LedgerRow.ref`: the MLflow run id.

A run ends `FINISHED` once every part is written, `FAILED` when a write fails part-way; `latest`
reads `FINISHED` runs whose row is complete, so a half-written or foreign run never reads as a
row. Runs end through `update_run` (MLflow's `set_terminated` prints a URL to stdout on a
tracking server), so their `end_time` is unset. Runs are ordered by start time in milliseconds,
stamped strictly increasing per ledger object; two ledger objects writing one kind in the same
millisecond have no defined order.

`MlflowModelRegistry(client)` maps `ModelRegistry` onto MLflow's model registry: a registered
model per name, a model version per `register`, aliases as MLflow aliases. A version's `source`
is the `artifact_uri` when one is given (a tracking server refuses a local path as a source, so
use a remote URI such as `s3://...` there, or `runs:/<id>/...`, whose run id is passed along as
MLflow's server requires); with none it is a backing run in the experiment
`registry_experiment` (`runs:/<id>/llm_ports`), since MLflow requires a source. The fitted
`model` is JSON in the version tag `llm_ports.model` (at most `MAX_MODEL_JSON` characters; a
larger model belongs at an `artifact_uri`), `run_ref` in `llm_ports.run_ref`. `register` does not
wait for the version to turn `READY` (MLflow's default waits and logs while it does); a backend that
copies the artifact first, Databricks for one, reports it `PENDING_REGISTRATION` until then.

The client is the caller's (`mlflow.MlflowClient(tracking_uri, registry_uri)`): a tracking
server, Databricks, or a local `sqlite:///` file. MLflow's client is synchronous; every call runs
in a worker thread, and one ledger or registry object is safe to share between threads. Against
an unreachable server MLflow retries each call before failing (`MLFLOW_HTTP_REQUEST_MAX_RETRIES`,
`MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR`, `MLFLOW_HTTP_REQUEST_TIMEOUT`; minutes by default), so a
loop that must report `unavailable` promptly sets those lower in its environment.

Invariants:
    - Never raises: MLflow refusing a value (`INVALID_PARAMETER_VALUE`) is a `request_fault`;
      any other failure, a server down included, is `unavailable`. A param or metric MLflow
      would refuse (its name, an unconvertible number) is left out of MLflow's columns, never
      a failed write; it stays in the row.
    - Its calls write nothing to stdout or stderr.
    - Passes `llm_ports.testing.conformance` (`check_run_ledger`, `check_model_registry` and their
      broken-storage checks) against a tracking server and a local SQLite store.

Rationale: ADR-0016.
"""

from llm_ports.adapters.mlflow.ledger import MAX_ROW_JSON, ROW_PARTS_TAG, ROW_TAG, MlflowRunLedger
from llm_ports.adapters.mlflow.registry import MAX_MODEL_JSON, MlflowModelRegistry

__all__ = ["MAX_MODEL_JSON", "MAX_ROW_JSON", "ROW_PARTS_TAG", "ROW_TAG", "MlflowModelRegistry", "MlflowRunLedger"]
