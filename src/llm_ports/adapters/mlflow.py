"""The learning loop's run ledger and model registry on MLflow (extra `[mlflow]`).

`MlflowRunLedger(client)` records each `RunLedger.append` as one MLflow run, so loops land in
MLflow's tracking UI and API next to every other experiment:

- experiment: `prefix + kind` (created on first append; `artifact_location` sets where its
  runs' artifacts go).
- tags: `llm_ports.kind`, `llm_ports.subject`, `llm_ports.decision` (the recorded verdict's
  decision, absent on an ungated run) and `mlflow.runName`.
- params / metrics: every scalar param with a name MLflow accepts (`log_param`), every finite
  numeric metric (`log_metric`); a verdict, a model, a string metric is not a number MLflow can
  chart, so the whole row is the JSON artifact `llm_ports/row.json`, which `latest` reads back.
- `Written.ref` / `LedgerRow.ref`: the MLflow run id.

A run that fails part-way is ended `FAILED`; `latest` reads `FINISHED` runs only, so a half-written
run never reads as a row. Runs are ordered by start time in milliseconds, stamped strictly
increasing per ledger object; two ledger objects writing one kind in the same millisecond have no
defined order.

`MlflowModelRegistry(client)` maps `ModelRegistry` onto MLflow's model registry: a registered
model per name, a model version per `register` (its `source` is the `artifact_uri`, `""` when
none), aliases as MLflow aliases. The fitted `model` is JSON in the version tag `llm_ports.model`
(at most `MAX_MODEL_JSON` characters; a larger model belongs at an `artifact_uri`), `run_ref` in
`llm_ports.run_ref`.

The client is the caller's (`mlflow.MlflowClient(tracking_uri, registry_uri)`): a tracking
server, Databricks, or a local `sqlite:///` file. MLflow's client is synchronous; every call runs
in a worker thread.

Invariants:
    - Never raises: MLflow refusing a value (`INVALID_PARAMETER_VALUE`) is a `request_fault`;
      any other failure, a server down included, is `unavailable`.
    - Passes `llm_ports.testing.conformance` (`check_run_ledger`, `check_model_registry` and their
      broken-storage checks).

Rationale: ADR-0016.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llm_ports._extras import extra
from llm_ports.learn import Decision, LedgerRow, ModelVersion, RowRead, StoreOutcome, VersionRead, Written

with extra("mlflow"):
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param
    from mlflow.exceptions import MlflowException

#: Where each run's full row lives among its artifacts.
ROW_ARTIFACT = "llm_ports/row.json"
#: The largest `model` JSON a registry version tag holds (MLflow's tag value limit).
MAX_MODEL_JSON = 100_000
_NAME = re.compile(r"^[\w\-./ ]{1,250}$")
_PAGE = 100


def _outcome(exc: Exception) -> StoreOutcome:
    """Return what an MLflow failure means: MLflow refusing the input, or MLflow unusable."""
    if isinstance(exc, MlflowException) and exc.error_code == "INVALID_PARAMETER_VALUE":
        return "request_fault"
    return "unavailable"


def _why(exc: Exception) -> str:
    return f"mlflow: {type(exc).__name__}: {exc}"[:500]


def _missing(exc: Exception) -> bool:
    return isinstance(exc, MlflowException) and exc.error_code == "RESOURCE_DOES_NOT_EXIST"


def _quoted(value: str) -> str | None:
    """Return `value` as an MLflow filter string literal, or None when no quote fits it."""
    for q in ("'", '"'):
        if q not in value:
            return f"{q}{value}{q}"
    return None


def _scalar_params(params: Mapping[str, Any]) -> list[Param]:
    return [
        Param(k, str(v))
        for k, v in params.items()
        if _NAME.match(k) and isinstance(v, str | int | float | bool) and len(str(v)) <= 6000
    ]


def _numeric_metrics(metrics: Mapping[str, Any], ms: int) -> list[Metric]:
    return [
        Metric(k, float(v), ms, 0)
        for k, v in metrics.items()
        if _NAME.match(k) and isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    ]


class MlflowRunLedger:
    """A `RunLedger` over MLflow tracking: one run per row (see the module docstring)."""

    def __init__(
        self,
        client: MlflowClient,
        *,
        prefix: str = "",
        artifact_location: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Record runs through `client`.

        Args:
            client: The caller's MLflow client.
            prefix: Prepended to a kind to name its experiment.
            artifact_location: Where a new experiment's artifacts go; None for the server's default.
            clock: Wall-clock seconds, for each run's start time; injectable for tests.
        """
        self.client: MlflowClient = client
        self.prefix: str = prefix
        self._artifacts = artifact_location
        self._clock = clock
        self._last_ms = 0
        self._experiments: dict[str, str] = {}

    def _stamp(self) -> int:
        self._last_ms = max(int(self._clock() * 1000), self._last_ms + 1)
        return self._last_ms

    def _experiment(self, kind: str, *, create: bool) -> str | None:
        name = self.prefix + kind
        if name not in self._experiments:
            found = self.client.get_experiment_by_name(name)
            if found is not None:
                self._experiments[name] = str(found.experiment_id)
            elif create:
                self._experiments[name] = str(self.client.create_experiment(name, artifact_location=self._artifacts))
        return self._experiments.get(name)

    def _append(self, kind: str, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]) -> Written:
        row = {"kind": kind, "subject": subject, "params": dict(params), "metrics": dict(metrics)}
        try:
            json.dumps(row)
        except (TypeError, ValueError) as exc:
            return Written(0, "request_fault", f"not JSON-able: {exc}")
        ms = self._stamp()
        decision = LedgerRow(kind, subject, params, metrics, datetime.now(UTC)).decision
        tags = {"llm_ports.kind": kind, "llm_ports.subject": subject, "mlflow.runName": subject or kind}
        if decision is not None:
            tags["llm_ports.decision"] = decision
        run_id: str | None = None
        try:
            experiment = self._experiment(kind, create=True)
            run_id = str(self.client.create_run(cast(str, experiment), start_time=ms, tags=tags).info.run_id)
            scalars = _scalar_params(params)
            for i in range(0, max(len(scalars), 1), _PAGE):
                self.client.log_batch(
                    run_id, metrics=_numeric_metrics(metrics, ms) if i == 0 else [], params=scalars[i : i + _PAGE]
                )
            self.client.log_dict(run_id, row, ROW_ARTIFACT)
            self.client.set_terminated(run_id, "FINISHED", end_time=int(self._clock() * 1000))
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            if run_id is not None:
                with contextlib.suppress(Exception):  # the write already failed; the outcome says so
                    self.client.set_terminated(run_id, "FAILED")
            return Written(0, _outcome(exc), _why(exc))
        return Written(1, "ok", ref=run_id)

    async def append(
        self, kind: str, *, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> Written:
        """Record one run as an MLflow run. Never raises."""
        return await asyncio.to_thread(self._append, kind, subject, params, metrics)

    def _row(self, run: Any) -> LedgerRow | None:
        """Rebuild a row from a run's artifact; None when the run carries no readable one."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self.client.download_artifacts(run.info.run_id, ROW_ARTIFACT, tmp)
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("params"), dict)
            or not isinstance(raw.get("metrics"), dict)
        ):
            return None
        at = datetime.fromtimestamp(run.info.start_time / 1000, UTC)
        got = cast(dict[str, Any], raw)
        return LedgerRow(
            str(got.get("kind", "")),
            str(got.get("subject", "")),
            got["params"],
            got["metrics"],
            at,
            str(run.info.run_id),
        )

    def _latest(self, kind: str, subject: str | None, decisions: Sequence[Decision] | None) -> RowRead:
        try:
            experiment = self._experiment(kind, create=False)
            if experiment is None:
                return RowRead(None, "ok")
            where = ["attributes.status = 'FINISHED'"]
            literal = _quoted(subject) if subject is not None else None
            if literal is not None:
                where.append(f"tags.`llm_ports.subject` = {literal}")
            if decisions is not None and len(decisions) == 1:
                where.append(f"tags.`llm_ports.decision` = '{decisions[0]}'")
            token: str | None = None
            while True:
                page = self.client.search_runs(
                    [experiment],
                    " and ".join(where),
                    max_results=_PAGE,
                    order_by=["attributes.start_time DESC"],
                    page_token=token,
                )
                for run in page:
                    tags = run.data.tags
                    if subject is not None and tags.get("llm_ports.subject") != subject:
                        continue
                    if decisions is not None and tags.get("llm_ports.decision") not in decisions:
                        continue
                    row = self._row(run)
                    if row is not None:
                        return RowRead(row, "ok")
                token = page.token
                if not token:
                    return RowRead(None, "ok")
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return RowRead(None, "unavailable", _why(exc))

    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        """Read the newest matching `FINISHED` run. Never raises; an unknown kind has no rows."""
        return await asyncio.to_thread(self._latest, kind, subject, decisions)


class MlflowModelRegistry:
    """A `ModelRegistry` over MLflow's model registry (see the module docstring)."""

    def __init__(self, client: MlflowClient) -> None:
        """Register models through `client` (its registry URI)."""
        self.client: MlflowClient = client

    def _register(
        self,
        name: str,
        model: Mapping[str, Any],
        artifact_uri: str | None,
        run_ref: str | None,
        tags: Mapping[str, str],
    ) -> Written:
        try:
            body = json.dumps(dict(model), sort_keys=True)
        except (TypeError, ValueError) as exc:
            return Written(0, "request_fault", f"model not JSON-able: {exc}")
        if len(body) > MAX_MODEL_JSON:
            return Written(
                0, "request_fault", f"model JSON is {len(body)} chars (max {MAX_MODEL_JSON}); use artifact_uri"
            )
        if not name or not all(isinstance(k, str) and isinstance(v, str) for k, v in tags.items()):
            return Written(0, "request_fault", "empty model name" if not name else "tags must map str to str")
        labels = {**tags, "llm_ports.model": body}
        if run_ref is not None:
            labels["llm_ports.run_ref"] = run_ref
        try:
            try:
                self.client.get_registered_model(name)
            except MlflowException as exc:
                if not _missing(exc):
                    raise
                self.client.create_registered_model(name)
            mv = self.client.create_model_version(name, artifact_uri or "", tags=labels)
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return Written(0, _outcome(exc), _why(exc))
        return Written(1, "ok", ref=str(mv.version))

    async def register(
        self,
        name: str,
        *,
        model: Mapping[str, Any],
        artifact_uri: str | None = None,
        run_ref: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Written:
        """Create an MLflow model version. Never raises."""
        return await asyncio.to_thread(self._register, name, model, artifact_uri, run_ref, tags or {})

    def _version(self, name: str, version: int) -> ModelVersion | None:
        try:
            mv = self.client.get_model_version(name, str(version))
        except MlflowException as exc:
            if _missing(exc):
                return None
            raise
        tags: dict[str, str] = dict(mv.tags or {})
        model = json.loads(tags.pop("llm_ports.model", "{}"))
        run_ref = tags.pop("llm_ports.run_ref", None)
        at = datetime.fromtimestamp(int(mv.creation_timestamp) / 1000, UTC)
        return ModelVersion(
            name, int(mv.version), model if isinstance(model, dict) else {}, mv.source or None, run_ref, at, tags
        )

    def _set_alias(self, name: str, alias: str, version: int) -> Written:
        try:
            if not alias:
                return Written(0, "request_fault", "empty alias")
            if self._version(name, version) is None:
                return Written(0, "request_fault", f"no version {version} of {name!r}")
            self.client.set_registered_model_alias(name, alias, str(version))
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return Written(0, _outcome(exc), _why(exc))
        return Written(1, "ok", ref=str(version))

    async def set_alias(self, name: str, alias: str, version: int) -> Written:
        """Set an MLflow alias. Never raises; MLflow refuses `latest` and `v<N>` as aliases."""
        return await asyncio.to_thread(self._set_alias, name, alias, version)

    def _by_alias(self, name: str, alias: str) -> VersionRead:
        try:
            try:
                aliases = self.client.get_registered_model(name).aliases or {}
            except MlflowException as exc:
                if _missing(exc):
                    return VersionRead(None, "ok")
                raise
            version = aliases.get(alias)
            return VersionRead(self._version(name, int(version)) if version is not None else None, "ok")
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return VersionRead(None, "unavailable", _why(exc))

    async def by_alias(self, name: str, alias: str) -> VersionRead:
        """Read the version an MLflow alias points at. Never raises."""
        return await asyncio.to_thread(self._by_alias, name, alias)

    def _get(self, name: str, version: int) -> VersionRead:
        try:
            return VersionRead(self._version(name, version), "ok")
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return VersionRead(None, "unavailable", _why(exc))

    async def get(self, name: str, version: int) -> VersionRead:
        """Read one MLflow model version. Never raises."""
        return await asyncio.to_thread(self._get, name, version)


__all__ = ["MAX_MODEL_JSON", "ROW_ARTIFACT", "MlflowModelRegistry", "MlflowRunLedger"]
