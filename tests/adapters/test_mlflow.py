"""The MLflow ledger and registry pass the conformance suites on a tracking server and on SQLite.

Every test runs twice: against a real `mlflow server` (REST handlers, server-side validation,
the client's stdout printing) and against a local `sqlite:///` store (no server). A local store
alone cannot fire on the REST paths, which is where the adapter's first version broke.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST

from llm_ports.adapters.mlflow import (
    MAX_MODEL_JSON,
    MAX_ROW_JSON,
    ROW_PARTS_TAG,
    ROW_TAG,
    MlflowModelRegistry,
    MlflowRunLedger,
)
from llm_ports.learn import VERDICT_KEY, Clause, Fit, Verdict, gate, run
from llm_ports.testing.conformance import (
    check_model_registry,
    check_run_ledger,
    check_unavailable_model_registry,
    check_unavailable_run_ledger,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Stop the server and the worker it spawned."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(30)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real tracking server on SQLite, for the module."""
    root = tmp_path_factory.mktemp("mlflow-server")
    port = _free_port()
    args = [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"]
    args += ["--backend-store-uri", "sqlite:///" + (root / "m.db").as_posix()]
    args += ["--default-artifact-root", (root / "artifacts").as_uri()]
    with (root / "server.log").open("wb") as log:
        proc = subprocess.Popen(  # our own interpreter, our own arguments
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
            start_new_session=sys.platform != "win32",
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()  # localhost
                    break
                except OSError:
                    if proc.poll() is not None or time.monotonic() > deadline:
                        pytest.fail(
                            "mlflow server did not start:\n" + (root / "server.log").read_text(errors="replace")
                        )
                    time.sleep(0.5)
            yield f"http://127.0.0.1:{port}"
        finally:
            _stop(proc)


@pytest.fixture(scope="module", params=["server", "sqlite"])
def client(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> MlflowClient:
    if request.param == "server":
        uri = str(request.getfixturevalue(server.__name__))  # started only when a test asks for it
    else:
        uri = "sqlite:///" + (tmp_path_factory.mktemp("mlflow-sqlite") / "m.db").as_posix()
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@pytest.fixture
def ledger(client: MlflowClient, tmp_path_factory: pytest.TempPathFactory) -> MlflowRunLedger:
    return MlflowRunLedger(client, prefix="t/", artifact_location=tmp_path_factory.mktemp("art").as_uri())


class _Down:
    """A client whose server is gone: every call raises."""

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            raise MlflowException("connection refused", error_code="TEMPORARILY_UNAVAILABLE")

        return call


async def test_the_ledger_passes_its_contract(ledger: MlflowRunLedger) -> None:
    assert len(await check_run_ledger(ledger)) == 9


async def test_the_registry_passes_its_contract(client: MlflowClient) -> None:
    assert len(await check_model_registry(MlflowModelRegistry(client))) == 8


async def test_a_server_down_is_reported_not_raised() -> None:
    assert len(await check_unavailable_run_ledger(MlflowRunLedger(_Down()))) == 2  # type: ignore[arg-type]  # a stand-in client
    assert len(await check_unavailable_model_registry(MlflowModelRegistry(_Down()))) == 2  # type: ignore[arg-type]  # a stand-in client


async def test_nothing_is_printed(
    ledger: MlflowRunLedger, client: MlflowClient, capfd: pytest.CaptureFixture[str]
) -> None:
    capfd.readouterr()
    assert (await ledger.append("quiet", subject="", params={"a": 1}, metrics={"n": 1})).ok
    assert (await ledger.latest("quiet")).row is not None
    reg = MlflowModelRegistry(client)
    assert (await reg.register("quiet", model={"w": 1})).ok
    assert (await reg.set_alias("quiet", "champion", 1)).ok
    assert (await reg.by_alias("quiet", "champion")).version is not None
    assert capfd.readouterr() == ("", "")  # MLflow's set_terminated prints a URL on a server


async def test_rows_land_as_mlflow_runs(ledger: MlflowRunLedger, client: MlflowClient) -> None:
    many = {f"p{i}": i for i in range(150)}  # over one log_batch's 100 params
    odd = {"a//b": 1, "x/": 1, "/lead": 1, "a/./b": 1, ".": 1, "../up": 1, "bad key!": 1}  # MLflow refuses these
    metrics = {
        "n": 3,
        "err": 0.5,
        "label": "x",
        "nan": float("nan"),
        "huge": 10**400,
        VERDICT_KEY: {"decision": "pass"},
    }
    wrote = await ledger.append("land", subject="SPY", params={**many, **odd, "obj": {"a": 1}}, metrics=metrics)
    assert wrote.ok and wrote.ref
    got = client.get_run(wrote.ref)
    assert got.info.status == "FINISHED"
    assert got.data.metrics == {"n": 3.0, "err": 0.5}
    assert set(got.data.params) == set(many)
    assert got.data.tags["llm_ports.subject"] == "SPY" and got.data.tags["llm_ports.decision"] == "pass"
    assert client.get_experiment(got.info.experiment_id).name == "t/land"
    row = (await ledger.latest("land", subject="SPY", decisions=("pass",))).row
    assert row is not None and row.ref == wrote.ref and row.params["obj"] == {"a": 1} and row.params["a//b"] == 1
    assert row.metrics["huge"] == 10**400
    assert (await ledger.latest("land", subject="QQQ")).row is None


async def test_a_large_row_spans_parts_and_many_metrics_span_batches(ledger: MlflowRunLedger) -> None:
    params = {"blob": "x" * 50_000}  # seven 8000-character parts
    metrics = {f"m{i}": float(i) for i in range(1001)}  # over one batch's metrics
    assert (await ledger.append("wide", subject="", params=params, metrics=metrics)).ok
    row = (await ledger.latest("wide")).row
    assert row is not None and row.params == params and len(row.metrics) == 1001
    too_big = await ledger.append("wide", subject="", params={"blob": "x" * MAX_ROW_JSON}, metrics={})
    assert too_big.outcome == "request_fault" and "max" in str(too_big.reason)


async def test_a_run_without_a_complete_row_is_skipped(ledger: MlflowRunLedger, client: MlflowClient) -> None:
    first = await ledger.append("skip", subject="", params={"seq": 1}, metrics={})
    exp = client.get_experiment_by_name("t/skip")
    assert exp is not None and first.ok
    for start in range(10**13 - 30, 10**13 - 5):  # more foreign runs than one search page
        stray = client.create_run(exp.experiment_id, start_time=start)
        client.update_run(stray.info.run_id, "FINISHED")
    for start, tags in (
        (10**13, {}),  # a foreign run: no row at all
        (10**13 + 1, {ROW_PARTS_TAG: "2", f"{ROW_TAG}/000": "{"}),  # cut short: a part missing
        (10**13 + 2, {ROW_PARTS_TAG: "1", f"{ROW_TAG}/000": '["not", "a", "row"]'}),
        (10**13 + 3, {ROW_PARTS_TAG: "x"}),
    ):
        stray = client.create_run(exp.experiment_id, start_time=start)
        client.log_batch(stray.info.run_id, tags=[RunTag(k, v) for k, v in tags.items()])
        client.update_run(stray.info.run_id, "FINISHED")
    row = (await ledger.latest("skip")).row
    assert row is not None and row.params == {"seq": 1}


async def test_a_failed_write_ends_the_run_failed(
    ledger: MlflowRunLedger, client: MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ledger.client, "log_batch", boom)
    wrote = await ledger.append("half", subject="", params={}, metrics={})
    assert wrote.outcome == "unavailable" and "disk full" in str(wrote.reason)
    monkeypatch.undo()
    assert (await ledger.latest("half")).row is None  # the half-written run is FAILED, never read
    exp = client.get_experiment_by_name("t/half")
    assert exp is not None
    assert [r.info.status for r in client.search_runs([exp.experiment_id])] == ["FAILED"]


async def test_concurrent_appends_keep_order_and_one_experiment(ledger: MlflowRunLedger, client: MlflowClient) -> None:
    wrote = await asyncio.gather(*(ledger.append("par", subject="", params={"i": i}, metrics={}) for i in range(8)))
    assert all(w.ok for w in wrote)
    exp = client.get_experiment_by_name("t/par")
    assert exp is not None
    starts = [r.info.start_time for r in client.search_runs([exp.experiment_id])]
    assert len(set(starts)) == 8  # strictly increasing stamps, never two in one millisecond


async def test_registry_refusals_are_request_faults(client: MlflowClient, tmp_path: Path) -> None:
    reg = MlflowModelRegistry(client)
    big = {"w": "x" * MAX_MODEL_JSON}
    assert "use artifact_uri" in str((await reg.register("big", model=big)).reason)
    assert (await reg.register("", model={})).reason == "empty model name"
    assert (await reg.register("m", model={}, tags={"k": 1})).reason == "tags must map str to str"  # type: ignore[dict-item]  # a wrong tag type is the input under test
    path = await reg.register("m", model={}, artifact_uri=tmp_path)  # type: ignore[arg-type]  # a Path is the input under test
    assert path.outcome == "request_fault" and "artifact_uri" in str(path.reason)
    assert (await reg.register("m", model={}, run_ref=7)).outcome == "request_fault"  # type: ignore[arg-type]  # an int is the input under test
    assert (await reg.register("refuse", model={})).ok
    assert (await reg.set_alias("refuse", "", 1)).reason == "empty alias"
    assert (await reg.set_alias("refuse", "v1", 1)).outcome == "request_fault"  # MLflow reserves v<N>
    for version in (0, -1):
        assert (await reg.get("refuse", version)) == (await reg.get("refuse", 99))


async def test_a_version_with_no_artifact_has_a_backing_run(client: MlflowClient) -> None:
    reg = MlflowModelRegistry(client, registry_experiment="t/registry")
    wrote = await reg.register("backed", model={"w": 2}, run_ref="ledger-run")
    assert wrote.ok
    mv = client.get_model_version("backed", str(wrote.ref))
    assert mv.run_id and mv.source == f"runs:/{mv.run_id}/llm_ports"
    assert client.get_run(mv.run_id).info.status == "FINISHED"
    got = (await reg.get("backed", 1)).version
    assert got is not None and got.artifact_uri is None and got.run_ref == "ledger-run"


async def test_a_loop_run_links_to_a_registered_version(ledger: MlflowRunLedger, client: MlflowClient) -> None:
    reg = MlflowModelRegistry(client)

    class Loop:
        kind, subject, params = "linked", "", {"window": 5}

        async def evidence(self) -> list[float]:
            return [1.0, 2.0]

        def fit(self, xs: list[float]) -> Fit:
            return Fit({"mean": sum(xs) / len(xs)}, {"n": len(xs)}, artifact_uri="s3://models/m.json")

        def clauses(self, fit: Fit, xs: list[float]) -> list[Clause]:
            return [gate.at_least("n", fit.metrics["n"], 2, role="floor")]

        async def apply(self, verdict: Verdict, fit: Fit) -> str:
            return "stored"

    got = await run(Loop(), ledger)
    assert got.fit is not None
    version = await reg.register(
        "linked", model=got.fit.model, artifact_uri=got.fit.artifact_uri, run_ref=got.recorded.ref
    )
    await reg.set_alias("linked", "champion", int(str(version.ref)))
    champion = (await reg.by_alias("linked", "champion")).version
    assert champion is not None and champion.run_ref == got.recorded.ref and champion.model == {"mean": 1.5}
    assert champion.artifact_uri == "s3://models/m.json"
    row = (await ledger.latest("linked")).row
    assert row is not None and row.metrics["learn_artifact_uri"] == "s3://models/m.json"
    assert row.metrics["learn_model"] == {"mean": 1.5}


class _Over:
    """The real client, with some methods replaced: another writer's race, or a store gone odd."""

    def __init__(self, client: MlflowClient, **calls: Any) -> None:
        self._client, self._calls = client, calls

    def __getattr__(self, name: str) -> Any:
        return self._calls.get(name) or getattr(self._client, name)


def _raise(code: int, *, then: Any = None) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        if then is not None:
            then(*args, **kwargs)
        raise MlflowException("raced", error_code=code)

    return call


async def test_another_writer_creating_the_experiment_first_is_not_a_failure(client: MlflowClient) -> None:
    assert (await MlflowRunLedger(client, prefix="t/").append("race", subject="", params={}, metrics={})).ok
    looks = iter([None])
    racy = _Over(
        client,
        get_experiment_by_name=lambda name: next(looks, None) or client.get_experiment_by_name(name),
        create_experiment=_raise(RESOURCE_ALREADY_EXISTS),
    )
    assert (await MlflowRunLedger(racy, prefix="t/").append("race", subject="", params={}, metrics={})).ok  # type: ignore[arg-type]  # a stand-in client
    vanished = _Over(
        client, get_experiment_by_name=lambda name: None, create_experiment=_raise(RESOURCE_ALREADY_EXISTS)
    )
    gone = await MlflowRunLedger(vanished).append("vanished", subject="", params={}, metrics={})  # type: ignore[arg-type]  # a stand-in client
    assert gone.outcome == "unavailable"
    refused = _Over(client, get_experiment_by_name=lambda name: None, create_experiment=_raise(INVALID_PARAMETER_VALUE))
    bad = await MlflowRunLedger(refused).append("refused", subject="", params={}, metrics={})  # type: ignore[arg-type]  # a stand-in client
    assert bad.outcome == "request_fault"


async def test_another_writer_creating_the_model_first_is_not_a_failure(client: MlflowClient) -> None:
    racy = _Over(
        client,
        get_registered_model=_raise(RESOURCE_DOES_NOT_EXIST),
        create_registered_model=_raise(RESOURCE_ALREADY_EXISTS, then=client.create_registered_model),
    )
    assert (await MlflowModelRegistry(racy).register("raced", model={})).ok  # type: ignore[arg-type]  # a stand-in client
    odd = _Over(
        client,
        get_registered_model=_raise(RESOURCE_DOES_NOT_EXIST),
        create_registered_model=_raise(INVALID_PARAMETER_VALUE),
    )
    assert (await MlflowModelRegistry(odd).register("odd", model={})).outcome == "request_fault"  # type: ignore[arg-type]  # a stand-in client


async def test_a_failed_registration_ends_its_backing_run_failed(client: MlflowClient) -> None:
    refusing = _Over(client, create_model_version=_raise(INVALID_PARAMETER_VALUE))
    reg = MlflowModelRegistry(refusing, registry_experiment="t/failed-registry")  # type: ignore[arg-type]  # a stand-in client
    assert (await reg.register("refused", model={})).outcome == "request_fault"
    exp = client.get_experiment_by_name("t/failed-registry")
    assert exp is not None
    assert [r.info.status for r in client.search_runs([exp.experiment_id])] == ["FAILED"]


async def test_a_version_another_tool_wrote_reads_with_an_empty_model(client: MlflowClient) -> None:
    client.create_registered_model("foreign")
    client.create_model_version("foreign", "s3://b/m", tags={"llm_ports.model": "not json"}, await_creation_for=0)
    got = (await MlflowModelRegistry(client).get("foreign", 1)).version
    assert got is not None and got.model == {} and got.artifact_uri == "s3://b/m"
