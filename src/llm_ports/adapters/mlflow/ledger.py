"""`MlflowRunLedger`: the learning loop's run ledger on MLflow tracking (see the package docstring)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import posixpath
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters.mlflow._common import experiment_id, outcome_of, reason_of
from llm_ports.learn import Decision, LedgerRow, RowRead, Written

with extra("mlflow"):
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag

#: The run-tag prefix of a row's JSON parts (`llm_ports.row/000`, ...).
ROW_TAG = "llm_ports.row"
#: The run tag holding how many parts the row has; written after them.
ROW_PARTS_TAG = "llm_ports.row.parts"
#: The largest row JSON a run holds.
MAX_ROW_JSON = 1_000_000
_PART = 8000  # MLflow's run-tag value limit
_NAME = re.compile(r"[\w\-./ ]{1,250}")  # MLflow's portable key characters (no colon: Windows servers)
_BATCH = 100  # params / tags per log_batch
_BATCH_METRICS = 800  # so one batch stays within MLflow's 1000 entities
_READ_PAGE = 20  # runs per search page: each carries its whole row


def _key(k: object) -> bool:
    """Return whether MLflow accepts `k` as a param or metric name (its own `path_not_unique` rule)."""
    if not isinstance(k, str) or not _NAME.fullmatch(k):
        return False
    norm = posixpath.normpath(k)
    return norm == k and norm != "." and not norm.startswith(("..", "/"))


def _number(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    try:
        f = float(v)
    except OverflowError:  # an int past float range
        return None
    return f if math.isfinite(f) else None


def _quoted(value: str) -> str | None:
    """Return `value` as an MLflow filter string literal, or None when no quote fits it."""
    for q in ("'", '"'):
        if q not in value:
            return f"{q}{value}{q}"
    return None


def _row_of(run: Any) -> LedgerRow | None:
    """Rebuild a row from a run's tags; None when the run carries no complete one."""
    tags: Mapping[str, str] = run.data.tags
    try:
        parts = int(tags[ROW_PARTS_TAG])
        raw = json.loads("".join(tags[f"{ROW_TAG}/{i:03d}"] for i in range(parts)))
    except (KeyError, ValueError):  # a foreign run, or one cut short; JSONDecodeError is a ValueError
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("params"), dict) or not isinstance(raw.get("metrics"), dict):
        return None
    at = datetime.fromtimestamp(run.info.start_time / 1000, UTC)
    return LedgerRow(
        str(raw.get("kind", "")), str(raw.get("subject", "")), raw["params"], raw["metrics"], at, str(run.info.run_id)
    )


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
        self._lock = threading.Lock()

    def _stamp(self) -> int:
        with self._lock:
            self._last_ms = max(int(self._clock() * 1000), self._last_ms + 1)
            return self._last_ms

    def _experiment(self, kind: str, *, create: bool) -> str | None:
        name = self.prefix + kind
        with self._lock:
            if name not in self._experiments:
                found = experiment_id(self.client, name, self._artifacts, create=create)
                if found is None:
                    return None
                self._experiments[name] = found
            return self._experiments[name]

    def _append(self, kind: str, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]) -> Written:
        try:
            body = json.dumps({"kind": kind, "subject": subject, "params": dict(params), "metrics": dict(metrics)})
        except (TypeError, ValueError) as exc:
            return Written(0, "request_fault", f"not JSON-able: {exc}")
        if len(subject) > _PART:  # MLflow would cut the subject tag short, and the filter then misses it
            return Written(0, "request_fault", f"subject is {len(subject)} chars (max {_PART})")
        if len(body) > MAX_ROW_JSON:
            return Written(0, "request_fault", f"row JSON is {len(body)} chars (max {MAX_ROW_JSON})")
        ms = self._stamp()
        decision = LedgerRow(kind, subject, params, metrics, datetime.now(UTC)).decision
        tags = {"llm_ports.kind": kind, "llm_ports.subject": subject, "mlflow.runName": subject or kind}
        if decision is not None:
            tags["llm_ports.decision"] = decision
        chunks = [body[i : i + _PART] for i in range(0, len(body), _PART)]
        row = [RunTag(f"{ROW_TAG}/{i:03d}", c) for i, c in enumerate(chunks)] + [
            RunTag(ROW_PARTS_TAG, str(len(chunks)))
        ]
        ps = [Param(k, str(v)) for k, v in params.items() if _key(k) and isinstance(v, str | int | float | bool)]
        ps = [p for p in ps if len(p.value) <= 6000]
        nums = [Metric(k, n, ms, 0) for k, v in metrics.items() if _key(k) and (n := _number(v)) is not None]
        run_id: str | None = None
        try:
            experiment = self._experiment(kind, create=True)
            run_id = str(self.client.create_run(str(experiment), start_time=ms, tags=tags).info.run_id)
            batches = max(
                math.ceil(len(ps) / _BATCH), math.ceil(len(row) / _BATCH), math.ceil(len(nums) / _BATCH_METRICS)
            )
            for i in range(batches):
                self.client.log_batch(
                    run_id,
                    metrics=nums[i * _BATCH_METRICS : (i + 1) * _BATCH_METRICS],
                    params=ps[i * _BATCH : (i + 1) * _BATCH],
                    tags=row[i * _BATCH : (i + 1) * _BATCH],
                )
            self.client.update_run(run_id, "FINISHED")
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            if run_id is not None:
                with contextlib.suppress(Exception):  # the write already failed; the outcome says so
                    self.client.update_run(run_id, "FAILED")
            return Written(0, outcome_of(exc), reason_of(exc))
        return Written(1, "ok", ref=run_id)

    async def append(
        self, kind: str, *, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> Written:
        """Record one run as an MLflow run. Never raises."""
        return await asyncio.to_thread(self._append, kind, subject, params, metrics)

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
                    max_results=_READ_PAGE,
                    order_by=["attributes.start_time DESC"],
                    page_token=token,
                )
                for run in page:
                    if decisions is not None and run.data.tags.get("llm_ports.decision") not in decisions:
                        continue
                    row = _row_of(run)  # its kind, not the experiment name: prefix "a" + "bc" is "ab" + "c"
                    if row is not None and row.kind == kind and (subject is None or row.subject == subject):
                        return RowRead(row, "ok")
                token = page.token
                if not token:
                    return RowRead(None, "ok")
        except Exception as exc:  # MLflow's own failure becomes a named outcome
            return RowRead(None, "unavailable", reason_of(exc))

    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        """Read the newest matching `FINISHED` run. Never raises; an unknown kind has no rows."""
        return await asyncio.to_thread(self._latest, kind, subject, decisions)


__all__ = ["MAX_ROW_JSON", "ROW_PARTS_TAG", "ROW_TAG", "MlflowRunLedger"]
