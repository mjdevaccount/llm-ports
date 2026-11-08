"""Model registries that need nothing but the standard library.

- `MemoryModelRegistry`: in-process, for tests and one-shot scripts.
- `JsonlModelRegistry`: an append-only JSON-lines event log (`register` and `alias` events),
  folded on read, so every alias move stays on the record.

A tracker's registry (MLflow's, for one) is an adapter that implements the same four methods.

Invariants:
    - Never raises for a backend failure: an unreadable or unwritable file is an `unavailable`
      outcome; a model that is not JSON-able, an empty name or alias, an `artifact_uri` or
      `run_ref` of the wrong type, or an alias set to a version that does not exist is a
      `request_fault`.
    - Versions of one name are numbered from 1 and never change once registered; only aliases move.
    - One writer per `JsonlModelRegistry` file: the next version number is read from the file.

Rationale: ADR-0016.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llm_ports.learn._types import ModelVersion, VersionRead, Written
from llm_ports.learn.stores import _read_lines, _when


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _State:
    """Versions and aliases, built by applying events in order."""

    def __init__(self) -> None:
        self.versions: dict[str, list[ModelVersion]] = {}
        self.aliases: dict[tuple[str, str], int] = {}

    def next_version(self, name: str) -> int:
        return len(self.versions.get(name, ())) + 1

    def get(self, name: str, version: int) -> ModelVersion | None:
        held = self.versions.get(name, [])
        return held[version - 1] if 1 <= version <= len(held) else None

    def by_alias(self, name: str, alias: str) -> ModelVersion | None:
        version = self.aliases.get((name, alias))
        return self.get(name, version) if version is not None else None


def _refusal(
    name: str, model: Mapping[str, Any], tags: Mapping[str, str], artifact_uri: object, run_ref: object
) -> str | None:
    """Return why a registration is refused, or None when it is acceptable."""
    if not isinstance(name, str) or not name:
        return "empty model name"
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in tags.items()):
        return "tags must map str to str"
    if artifact_uri is not None and (not isinstance(artifact_uri, str) or not artifact_uri):
        return f"artifact_uri must be a non-empty str or None, not {type(artifact_uri).__name__}"
    if run_ref is not None and not isinstance(run_ref, str):
        return f"run_ref must be a str or None, not {type(run_ref).__name__}"
    try:
        json.dumps(dict(model))
    except (TypeError, ValueError) as exc:
        return f"model not JSON-able: {exc}"
    return None


class MemoryModelRegistry:
    """A `ModelRegistry` held in memory."""

    def __init__(self, *, now: Callable[[], datetime] = _utcnow) -> None:
        """Create an empty registry.

        Args:
            now: The clock that stamps each version; injectable for tests.
        """
        self._state = _State()
        self._now = now

    async def register(
        self,
        name: str,
        *,
        model: Mapping[str, Any],
        artifact_uri: str | None = None,
        run_ref: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Written:
        """Register a new version. Never raises."""
        refused = _refusal(name, model, tags or {}, artifact_uri, run_ref)
        if refused is not None:
            return Written(0, "request_fault", refused)
        version = self._state.next_version(name)
        mv = ModelVersion(name, version, dict(model), artifact_uri, run_ref, self._now(), dict(tags or {}))
        self._state.versions.setdefault(name, []).append(mv)
        return Written(1, "ok", ref=str(version))

    async def set_alias(self, name: str, alias: str, version: int) -> Written:
        """Point `alias` at `version`. Never raises."""
        if not alias or self._state.get(name, version) is None:
            return Written(0, "request_fault", f"no version {version} of {name!r}" if alias else "empty alias")
        self._state.aliases[(name, alias)] = version
        return Written(1, "ok", ref=str(version))

    async def by_alias(self, name: str, alias: str) -> VersionRead:
        """Read the version `alias` points at. Never raises."""
        return VersionRead(self._state.by_alias(name, alias), "ok")

    async def get(self, name: str, version: int) -> VersionRead:
        """Read one version. Never raises."""
        return VersionRead(self._state.get(name, version), "ok")


class JsonlModelRegistry:
    """A `ModelRegistry` in a JSON-lines event log: appended, never rewritten."""

    def __init__(self, path: str | Path, *, now: Callable[[], datetime] = _utcnow) -> None:
        """Use the registry at `path` (created on first write).

        Args:
            path: The file.
            now: The clock that stamps each event; injectable for tests.
        """
        self.path: Path = Path(path)
        self._now = now

    def _fold(self) -> _State | str:
        """Return the state the log describes, or why the log could not be read."""
        rows = _read_lines(self.path)
        if isinstance(rows, str):
            return rows
        state = _State()
        for raw in rows:
            _apply(state, raw)
        return state

    def _append(self, event: dict[str, Any]) -> str | None:
        """Append one event; return why it failed, or None."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, sort_keys=True) + "\n")
        except OSError as exc:
            return f"{self.path}: {exc}"
        return None

    async def register(
        self,
        name: str,
        *,
        model: Mapping[str, Any],
        artifact_uri: str | None = None,
        run_ref: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Written:
        """Append a `register` event. Never raises."""
        refused = _refusal(name, model, tags or {}, artifact_uri, run_ref)
        if refused is not None:
            return Written(0, "request_fault", refused)
        state = self._fold()
        if isinstance(state, str):
            return Written(0, "unavailable", state)
        version = state.next_version(name)
        failed = self._append(
            {
                "op": "register",
                "name": name,
                "version": version,
                "model": dict(model),
                "artifact_uri": artifact_uri,
                "run_ref": run_ref,
                "tags": dict(tags or {}),
                "at": self._now().isoformat(),
            }
        )
        return Written(0, "unavailable", failed) if failed else Written(1, "ok", ref=str(version))

    async def set_alias(self, name: str, alias: str, version: int) -> Written:
        """Append an `alias` event. Never raises."""
        state = self._fold()
        if isinstance(state, str):
            return Written(0, "unavailable", state)
        if not alias or state.get(name, version) is None:
            return Written(0, "request_fault", f"no version {version} of {name!r}" if alias else "empty alias")
        event = {"op": "alias", "name": name, "alias": alias, "version": version, "at": self._now().isoformat()}
        failed = self._append(event)
        return Written(0, "unavailable", failed) if failed else Written(1, "ok", ref=str(version))

    async def by_alias(self, name: str, alias: str) -> VersionRead:
        """Read the version `alias` points at. Never raises; a missing file is an empty registry."""
        state = self._fold()
        if isinstance(state, str):
            return VersionRead(None, "unavailable", state)
        return VersionRead(state.by_alias(name, alias), "ok")

    async def get(self, name: str, version: int) -> VersionRead:
        """Read one version. Never raises; a missing file is an empty registry."""
        state = self._fold()
        if isinstance(state, str):
            return VersionRead(None, "unavailable", state)
        return VersionRead(state.get(name, version), "ok")


def _apply(state: _State, raw: Mapping[str, Any]) -> None:
    """Apply one logged event; a malformed or out-of-sequence event is skipped."""
    name, version = raw.get("name"), raw.get("version")
    if not isinstance(name, str) or not isinstance(version, int):
        return
    if raw.get("op") == "alias":
        alias = raw.get("alias")
        if isinstance(alias, str) and state.get(name, version) is not None:
            state.aliases[(name, alias)] = version
        return
    model, at, tags = raw.get("model"), _when(raw.get("at")), raw.get("tags")
    if raw.get("op") != "register" or version != state.next_version(name) or not isinstance(model, dict) or at is None:
        return
    artifact_uri, run_ref = raw.get("artifact_uri"), raw.get("run_ref")
    state.versions.setdefault(name, []).append(
        ModelVersion(
            name,
            version,
            cast(dict[str, Any], model),
            artifact_uri if isinstance(artifact_uri, str) else None,
            run_ref if isinstance(run_ref, str) else None,
            at,
            {str(k): str(v) for k, v in cast(dict[str, Any], tags).items()} if isinstance(tags, dict) else {},
        )
    )


__all__ = ["JsonlModelRegistry", "MemoryModelRegistry"]
