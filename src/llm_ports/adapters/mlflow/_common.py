"""What the MLflow ledger and registry share: failure mapping and experiment lookup."""

from __future__ import annotations

from llm_ports._extras import extra
from llm_ports.learn import StoreOutcome

with extra("mlflow"):
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException


def outcome_of(exc: Exception) -> StoreOutcome:
    """Return what an MLflow failure means: MLflow refusing the input, or MLflow unusable."""
    if isinstance(exc, MlflowException) and exc.error_code == "INVALID_PARAMETER_VALUE":
        return "request_fault"
    return "unavailable"


def reason_of(exc: Exception) -> str:
    return f"mlflow: {type(exc).__name__}: {exc}"[:500]


def has_code(exc: Exception, wanted: str) -> bool:
    return isinstance(exc, MlflowException) and exc.error_code == wanted


def experiment_id(client: MlflowClient, name: str, artifact_location: str | None, *, create: bool) -> str | None:
    found = client.get_experiment_by_name(name)
    if found is not None:
        return str(found.experiment_id)
    if not create:
        return None
    try:
        return str(client.create_experiment(name, artifact_location=artifact_location))
    except MlflowException as exc:
        if not has_code(exc, "RESOURCE_ALREADY_EXISTS"):
            raise
        again = client.get_experiment_by_name(name)  # another writer created it first
        if again is None:
            raise
        return str(again.experiment_id)


__all__: list[str] = []  # shared by the ledger and registry; no surface of its own
