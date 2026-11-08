"""The MLflow ledger and registry pass the conformance suites over a local SQLite store."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from llm_ports.adapters.mlflow import MAX_MODEL_JSON, ROW_ARTIFACT, MlflowModelRegistry, MlflowRunLedger
from llm_ports.learn import VERDICT_KEY, Clause, Fit, Verdict, gate, run
from llm_ports.testing.conformance import (
    check_model_registry,
    check_run_ledger,
    check_unavailable_model_registry,
    check_unavailable_run_ledger,
)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[MlflowClient, str]]:
    """One SQLite store for the module: MLflow migrates a fresh database in seconds, not millis."""
    root = tmp_path_factory.mktemp("mlflow")
    uri = "sqlite:///" + (root / "mlflow.db").as_posix()
    yield MlflowClient(tracking_uri=uri, registry_uri=uri), (root / "artifacts").as_uri()


@pytest.fixture
def ledger(store: tuple[MlflowClient, str]) -> MlflowRunLedger:
    client, artifacts = store
    return MlflowRunLedger(client, prefix="t/", artifact_location=artifacts)


class _Down:
    """A client whose server is gone: every call raises."""

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            raise MlflowException("connection refused", error_code="TEMPORARILY_UNAVAILABLE")

        return call


async def test_the_ledger_passes_its_contract(ledger: MlflowRunLedger) -> None:
    assert len(await check_run_ledger(ledger)) == 9


async def test_the_registry_passes_its_contract(store: tuple[MlflowClient, str]) -> None:
    assert len(await check_model_registry(MlflowModelRegistry(store[0]))) == 8


async def test_a_server_down_is_reported_not_raised() -> None:
    assert len(await check_unavailable_run_ledger(MlflowRunLedger(_Down()))) == 2  # type: ignore[arg-type]  # a stand-in client
    assert len(await check_unavailable_model_registry(MlflowModelRegistry(_Down()))) == 2  # type: ignore[arg-type]  # a stand-in client


async def test_rows_land_as_mlflow_runs(ledger: MlflowRunLedger, store: tuple[MlflowClient, str]) -> None:
    client = store[0]
    many = {f"p{i}": i for i in range(150)}  # over one log_batch's 100 params
    metrics = {"n": 3, "err": 0.5, "label": "x", "nan": float("nan"), "bad key!": 1, VERDICT_KEY: {"decision": "pass"}}
    wrote = await ledger.append("land", subject="SPY", params={**many, "obj": {"a": 1}}, metrics=metrics)
    assert wrote.ok and wrote.ref
    got = client.get_run(wrote.ref)
    assert got.data.metrics == {"n": 3.0, "err": 0.5}
    assert len(got.data.params) == 150 and "obj" not in got.data.params
    assert got.data.tags["llm_ports.subject"] == "SPY" and got.data.tags["llm_ports.decision"] == "pass"
    assert client.get_experiment(got.info.experiment_id).name == "t/land"
    row = (await ledger.latest("land", subject="SPY", decisions=("pass",))).row
    assert row is not None and row.ref == wrote.ref and row.params["obj"] == {"a": 1}
    assert (await ledger.latest("land", subject="QQQ")).row is None


async def test_a_run_without_its_row_artifact_is_skipped(
    ledger: MlflowRunLedger, store: tuple[MlflowClient, str]
) -> None:
    client = store[0]
    first = await ledger.append("skip", subject="", params={"seq": 1}, metrics={})
    exp = client.get_experiment_by_name("t/skip")
    assert exp is not None and first.ok
    stray = client.create_run(exp.experiment_id, start_time=10**13)  # newest, and not ours
    client.log_dict(stray.info.run_id, ["not", "a", "row"], ROW_ARTIFACT)
    client.set_terminated(stray.info.run_id)
    row = (await ledger.latest("skip")).row
    assert row is not None and row.params == {"seq": 1}


async def test_refusals_are_request_faults(store: tuple[MlflowClient, str]) -> None:
    reg = MlflowModelRegistry(store[0])
    big = {"w": "x" * MAX_MODEL_JSON}
    assert "use artifact_uri" in str((await reg.register("big", model=big)).reason)
    assert (await reg.register("", model={})).reason == "empty model name"
    assert (await reg.register("m", model={}, tags={"k": 1})).reason == "tags must map str to str"  # type: ignore[dict-item]  # a wrong tag type is the input under test
    assert (await reg.register("refuse", model={})).ok
    assert (await reg.set_alias("refuse", "", 1)).reason == "empty alias"
    assert (await reg.set_alias("refuse", "v1", 1)).outcome == "request_fault"  # MLflow reserves v<N>


async def test_a_failed_write_ends_the_run_failed(ledger: MlflowRunLedger, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ledger.client, "log_dict", boom)
    wrote = await ledger.append("half", subject="", params={}, metrics={})
    assert wrote.outcome == "unavailable" and "disk full" in str(wrote.reason)
    assert (await ledger.latest("half")).row is None  # the half-written run is FAILED, never read


async def test_a_loop_run_links_to_a_registered_version(
    ledger: MlflowRunLedger, store: tuple[MlflowClient, str]
) -> None:
    reg = MlflowModelRegistry(store[0])

    class Loop:
        kind, subject, params = "linked", "", {"window": 5}

        async def evidence(self) -> list[float]:
            return [1.0, 2.0]

        def fit(self, xs: list[float]) -> Fit:
            return Fit({"mean": sum(xs) / len(xs)}, {"n": len(xs)}, artifact_uri="file:///m.json")

        def clauses(self, fit: Fit, xs: list[float]) -> list[Clause]:
            return [gate.at_least("n", fit.metrics["n"], 2, role="floor")]

        async def apply(self, verdict: Verdict, fit: Fit) -> str:
            return "stored"

    got = await run(Loop(), ledger)
    version = await reg.register("linked", model=got.fit.model if got.fit else {}, run_ref=got.recorded.ref)
    await reg.set_alias("linked", "champion", int(str(version.ref)))
    champion = (await reg.by_alias("linked", "champion")).version
    assert champion is not None and champion.run_ref == got.recorded.ref and champion.model == {"mean": 1.5}
    row = (await ledger.latest("linked")).row
    assert row is not None and row.metrics["learn_artifact_uri"] == "file:///m.json"
    assert row.metrics["learn_model"] == {"mean": 1.5}
