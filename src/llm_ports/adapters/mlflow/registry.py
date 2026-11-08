"""`MlflowModelRegistry`: the model registry port on MLflow's registry (see the package docstring)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters.mlflow._common import experiment_id, has_code, outcome_of, reason_of
from llm_ports.learn import ModelVersion, VersionRead, Written
from llm_ports.learn.registry import _bad_version, _refusal

with extra("mlflow"):
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

#: The largest `model` JSON a registry version tag holds.
MAX_MODEL_JSON = 100_000
_BACKING = "llm_ports"


def _run_of(artifact_uri: str) -> str | None:
    """Return the run a `runs:/<id>/...` URI names (a server refuses one registered without it)."""
    if not artifact_uri.startswith("runs:/"):
        return None
    return artifact_uri.removeprefix("runs:/").split("/", 1)[0] or None


class MlflowModelRegistry:
    """A `ModelRegistry` over MLflow's model registry (see the module docstring)."""

    def __init__(self, client: MlflowClient, *, registry_experiment: str = "llm_ports.registry") -> None:
        """Register models through `client` (its registry URI).

        Args:
            client: The caller's MLflow client.
            registry_experiment: The experiment holding the backing run of a version registered
                with no `artifact_uri`.
        """
        self.client: MlflowClient = client
        self.registry_experiment: str = registry_experiment
        self._experiment: str | None = None
        self._lock = threading.Lock()

    def _backing_run(self, name: str) -> str:
        with self._lock:
            if self._experiment is None:
                self._experiment = experiment_id(self.client, self.registry_experiment, None, create=True)
        tags = {"llm_ports.registered_model": name, "mlflow.runName": name}
        return str(self.client.create_run(str(self._experiment), tags=tags).info.run_id)

    def _ensure_model(self, name: str) -> None:
        try:
            self.client.get_registered_model(name)
        except MlflowException as exc:
            if not has_code(exc, "RESOURCE_DOES_NOT_EXIST"):
                raise
            try:
                self.client.create_registered_model(name)
            except MlflowException as again:
                if not has_code(again, "RESOURCE_ALREADY_EXISTS"):  # another writer created it first
                    raise

    def _register(
        self,
        name: str,
        model: Mapping[str, Any],
        artifact_uri: str | None,
        run_ref: str | None,
        tags: Mapping[str, str],
    ) -> Written:
        refused = _refusal(name, model, tags, artifact_uri, run_ref)
        if refused is not None:
            return Written(0, "request_fault", refused)
        body = json.dumps(dict(model), sort_keys=True)
        if len(body) > MAX_MODEL_JSON:
            return Written(
                0, "request_fault", f"model JSON is {len(body)} chars (max {MAX_MODEL_JSON}); use artifact_uri"
            )
        labels = {**tags, "llm_ports.model": body}
        if run_ref is not None:
            labels["llm_ports.run_ref"] = run_ref
        backing: str | None = None
        try:
            self._ensure_model(name)
            if artifact_uri is not None:
                mv = self.client.create_model_version(
                    name, artifact_uri, run_id=_run_of(artifact_uri), tags=labels, await_creation_for=0
                )
            else:
                backing = self._backing_run(name)
                mv = self.client.create_model_version(
                    name, f"runs:/{backing}/{_BACKING}", run_id=backing, tags=labels, await_creation_for=0
                )
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            if backing is not None:
                with contextlib.suppress(Exception):  # the write already failed; the outcome says so
                    self.client.update_run(backing, "FAILED")
            return Written(0, outcome_of(exc), reason_of(exc))
        if backing is not None:  # the version exists: a run left RUNNING must not turn it into a retry
            with contextlib.suppress(Exception):
                self.client.update_run(backing, "FINISHED")
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
        if version < 1:
            return None
        try:
            mv = self.client.get_model_version(name, str(version))
        except MlflowException as exc:
            if has_code(exc, "RESOURCE_DOES_NOT_EXIST"):
                return None
            raise
        tags: dict[str, str] = dict(mv.tags or {})
        try:
            model = json.loads(tags.pop("llm_ports.model", "{}"))
        except ValueError:  # a version some other tool wrote
            model = {}
        run_ref = tags.pop("llm_ports.run_ref", None)
        source = mv.source or None
        if mv.run_id and source == f"runs:/{mv.run_id}/{_BACKING}":
            source = None  # our backing run, not a caller's artifact
        at = datetime.fromtimestamp(int(mv.creation_timestamp) / 1000, UTC)
        return ModelVersion(name, int(mv.version), model if isinstance(model, dict) else {}, source, run_ref, at, tags)

    def _set_alias(self, name: str, alias: str, version: int) -> Written:
        if (bad := _bad_version(version)) is not None:
            return Written(0, "request_fault", bad)
        try:
            if not alias:
                return Written(0, "request_fault", "empty alias")
            if self._version(name, version) is None:
                return Written(0, "request_fault", f"no version {version} of {name!r}")
            self.client.set_registered_model_alias(name, alias, str(version))
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return Written(0, outcome_of(exc), reason_of(exc))
        return Written(1, "ok", ref=str(version))

    async def set_alias(self, name: str, alias: str, version: int) -> Written:
        """Set an MLflow alias. Never raises; MLflow refuses `latest` and `v<N>` as aliases."""
        return await asyncio.to_thread(self._set_alias, name, alias, version)

    def _by_alias(self, name: str, alias: str) -> VersionRead:
        try:
            try:
                aliases = self.client.get_registered_model(name).aliases or {}
            except MlflowException as exc:
                if has_code(exc, "RESOURCE_DOES_NOT_EXIST"):
                    return VersionRead(None, "ok")
                raise
            version = aliases.get(alias)
            return VersionRead(self._version(name, int(version)) if version is not None else None, "ok")
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return VersionRead(None, "unavailable", reason_of(exc))

    async def by_alias(self, name: str, alias: str) -> VersionRead:
        """Read the version an MLflow alias points at. Never raises."""
        return await asyncio.to_thread(self._by_alias, name, alias)

    def _get(self, name: str, version: int) -> VersionRead:
        if (bad := _bad_version(version)) is not None:
            return VersionRead(None, "request_fault", bad)
        try:
            return VersionRead(self._version(name, version), "ok")
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return VersionRead(None, "unavailable", reason_of(exc))

    async def get(self, name: str, version: int) -> VersionRead:
        """Read one MLflow model version. Never raises."""
        return await asyncio.to_thread(self._get, name, version)


__all__ = ["MAX_MODEL_JSON", "MlflowModelRegistry"]
