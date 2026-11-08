# Bigger ML engines

The learning loop's storage is three ports, each with a standard-library backend. A bigger engine
(MLflow first; W&B, Ray, a managed platform later) is an adapter that implements the same port,
so a loop and its consumers do not change when the engine does. Why it is shaped this way:
[ADR-0016](../adr/0016-seams-to-bigger-ml-engines.md).

| Port | Standard library | MLflow (`[mlflow]`) | Maps onto |
|---|---|---|---|
| `RunLedger` | `MemoryRunLedger`, `JsonlRunLedger` | `MlflowRunLedger` | tracking runs |
| `ModelRegistry` | `MemoryModelRegistry`, `JsonlModelRegistry` | `MlflowModelRegistry` | registry versions and aliases |
| `ExampleStore` | `MemoryExampleStore`, `JsonlExampleStore` | | a feature store's offline read |

The rest of the stack already speaks the standards an engine reads: estimators follow the
scikit-learn API, tuning is an Optuna study (Ray Tune wraps it with `OptunaSearch`), traces use
the OpenTelemetry GenAI conventions, provenance is PROV-O.

## Lineage: run, version, alias

`run(loop, ledger)` returns the ledger's answer in `Run.recorded`; its `ref` is the row's id (a
UUID in the standard-library ledgers, the run id in MLflow). Register the fit with that ref, and
promotion is moving an alias:

--8<-- "examples/ml_engine_seams.py"

A model too big for JSON goes to a file or bucket first; set `Fit.artifact_uri` and pass the
same URI to `register`. The ledger row records it as `learn_artifact_uri`.

## Swapping in MLflow

```{.python test="skip"}
from mlflow import MlflowClient

from llm_ports.adapters.mlflow import MlflowModelRegistry, MlflowRunLedger

client = MlflowClient(tracking_uri="sqlite:///mlflow.db", registry_uri="sqlite:///mlflow.db")
ledger = MlflowRunLedger(client, prefix="loops/", artifact_location="file:///srv/mlflow/artifacts")
registry = MlflowModelRegistry(client)
```

Each row is one MLflow run in the experiment `prefix + kind`, tagged with its subject and
decision; scalar params and numeric metrics are logged so MLflow can chart them, and the whole
row is JSON in the run's tags (`llm_ports.row/000`, ...), read back with no artifact store. A
tracking server works the same way (`tracking_uri="http://..."`); there, an `artifact_uri`
passed to `register` must be a remote URI (`s3://...`), since MLflow refuses a local path as a
model version's source. A version registered with no `artifact_uri` gets a backing run in the
experiment `llm_ports.registry` as its source.

## Adding a backend

A new backend passes the conformance suite before anything reads from it:

```{.python test="skip"}
from llm_ports.testing.conformance import check_run_ledger, check_unavailable_run_ledger

assert await check_run_ledger(MyLedger(live_connection))
assert await check_unavailable_run_ledger(MyLedger(closed_connection))
```

Each check writes under fresh random names, so it runs against a shared or production store, and
raises `ContractViolation` naming the first invariant the backend breaks: a raise across the
port, an unreadable store read as empty, a filter ignored. Rules for an engine adapter:

- The core stays standard library; the engine's client is imported only in its adapter, behind
  its extra.
- `Loop.fit` stays pure and `Fit` plain data, so an engine can run the fit in a remote worker.
- Map each port onto the engine's own concept (experiment, run, alias), not a private schema.
