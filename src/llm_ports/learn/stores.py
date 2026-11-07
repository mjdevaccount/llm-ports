"""Ledgers and example stores that need nothing but the standard library.

- `MemoryRunLedger`, `MemoryExampleStore`: in-process, for tests and one-shot scripts.
- `JsonlRunLedger`, `JsonlExampleStore`: one JSON object per line in a file, so a loop runs and
  keeps its history anywhere with zero setup.
- `union`: read several example stores as one.

A database-backed ledger is an adapter in the consuming application: it implements the two
`RunLedger` methods over its own table.

Invariants:
    - Never raises for a backend failure: an unreadable or unwritable file is an `unavailable`
      outcome; a value that is not JSON-able is a `request_fault`.
    - An unreadable store never reads as an empty one.

Rationale: ADR-0010.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llm_ports.learn._protocols import ExampleStore
from llm_ports.learn._types import Decision, Example, ExamplesRead, LedgerRow, RowRead, Source, Written


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _matches(row: LedgerRow, kind: str, subject: str | None, decisions: Sequence[Decision] | None) -> bool:
    if row.kind != kind or (subject is not None and row.subject != subject):
        return False
    return decisions is None or row.decision in decisions


def _wanted(ex: Example, topics: Sequence[str], since: datetime | None, source: Source | None) -> bool:
    if ex.topic not in topics or (source is not None and ex.source != source):
        return False
    return since is None or (ex.at is not None and ex.at >= since)


class MemoryRunLedger:
    """A `RunLedger` held in a list (`rows`), oldest first."""

    def __init__(self, *, now: Callable[[], datetime] = _utcnow) -> None:
        """Create an empty ledger.

        Args:
            now: The clock that stamps each row; injectable for tests.
        """
        self.rows: list[LedgerRow] = []
        self._now = now

    async def append(
        self, kind: str, *, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> Written:
        """Record one run. Never raises."""
        self.rows.append(LedgerRow(kind, subject, dict(params), dict(metrics), self._now()))
        return Written(1, "ok")

    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        """Read the newest matching row. Never raises."""
        hits = [r for r in self.rows if _matches(r, kind, subject, decisions)]
        return RowRead(hits[-1] if hits else None, "ok")


class JsonlRunLedger:
    """A `RunLedger` in a JSON-lines file: one row per line, appended, never rewritten."""

    def __init__(self, path: str | Path, *, now: Callable[[], datetime] = _utcnow) -> None:
        """Use the ledger at `path` (created on first append).

        Args:
            path: The file.
            now: The clock that stamps each row; injectable for tests.
        """
        self.path = Path(path)
        self._now = now

    async def append(
        self, kind: str, *, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> Written:
        """Append one row. Never raises."""
        row = {"kind": kind, "subject": subject, "params": params, "metrics": metrics, "at": self._now().isoformat()}
        try:
            line = json.dumps(row, sort_keys=True)
        except (TypeError, ValueError) as exc:
            return Written(0, "request_fault", f"not JSON-able: {exc}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            return Written(0, "unavailable", f"{self.path}: {exc}")
        return Written(1, "ok")

    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        """Read the newest matching row. Never raises; a missing file is an empty ledger."""
        rows = _read_lines(self.path)
        if isinstance(rows, str):
            return RowRead(None, "unavailable", rows)
        found: LedgerRow | None = None
        for raw in rows:
            row = _row(raw)
            if row is not None and _matches(row, kind, subject, decisions):
                found = row
        return RowRead(found, "ok")


class MemoryExampleStore:
    """A writable `ExampleStore` held in a list (`items`)."""

    def __init__(self, examples: Iterable[Example] = ()) -> None:
        """Create a store holding `examples`."""
        self.items: list[Example] = list(examples)

    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        """Read matching examples. Never raises."""
        return ExamplesRead(tuple(e for e in self.items if _wanted(e, topics, since, source)), "ok")

    async def put(self, examples: Sequence[Example]) -> Written:
        """Store examples. Never raises."""
        self.items.extend(examples)
        return Written(len(examples), "ok")


class JsonlExampleStore:
    """A writable `ExampleStore` in a JSON-lines file: one example per line."""

    def __init__(self, path: str | Path) -> None:
        """Use the store at `path` (created on first put)."""
        self.path = Path(path)

    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        """Read matching examples. Never raises; a missing file is an empty store."""
        rows = _read_lines(self.path)
        if isinstance(rows, str):
            return ExamplesRead((), "unavailable", rows)
        found = (_example(raw) for raw in rows)
        return ExamplesRead(tuple(e for e in found if e is not None and _wanted(e, topics, since, source)), "ok")

    async def put(self, examples: Sequence[Example]) -> Written:
        """Append examples. Never raises."""
        try:
            lines = [json.dumps(_example_dict(e), sort_keys=True) for e in examples]
        except (TypeError, ValueError) as exc:
            return Written(0, "request_fault", f"not JSON-able: {exc}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.writelines(line + "\n" for line in lines)
        except OSError as exc:
            return Written(0, "unavailable", f"{self.path}: {exc}")
        return Written(len(lines), "ok")


class _Union:
    """Several example stores read as one."""

    def __init__(self, stores: Sequence[ExampleStore]) -> None:
        self._stores = tuple(stores)

    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        out: list[Example] = []
        for i, store in enumerate(self._stores):
            got = await store.examples(topics, since=since, source=source)
            if not got.ok:
                return ExamplesRead((), "unavailable", f"store {i} ({type(store).__name__}): {got.reason}")
            out += got.examples
        return ExamplesRead(tuple(out), "ok")


def union(*stores: ExampleStore) -> ExampleStore:
    """Return one read-only store over `stores`.

    Returns:
        A store whose read is every store's read concatenated, or `unavailable` (naming the store)
        when any one cannot be read: a partial read would undercount a topic's labels silently.
    """
    return _Union(stores)


def _read_lines(path: Path) -> list[dict[str, Any]] | str:
    """Return the file's JSON objects (a missing file has none), or a failure reason."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        return f"{path}: {exc}"
    out: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            return f"{path}:{n}: not JSON"
        if isinstance(obj, dict):
            out.append(cast(dict[str, Any], obj))
    return out


def _when(raw: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def _row(raw: Mapping[str, Any]) -> LedgerRow | None:
    at = _when(raw.get("at"))
    params, metrics = raw.get("params"), raw.get("metrics")
    if at is None or not isinstance(params, dict) or not isinstance(metrics, dict):
        return None
    return LedgerRow(
        str(raw.get("kind", "")),
        str(raw.get("subject", "")),
        cast(dict[str, Any], params),
        cast(dict[str, Any], metrics),
        at,
    )


def _example_dict(e: Example) -> dict[str, Any]:
    return {
        "topic": e.topic,
        "subject": e.subject,
        "value": e.value,
        "source": e.source,
        "by": e.by,
        "at": e.at.isoformat() if e.at else None,
        "reason": e.reason,
        "ref": dict(e.ref),
    }


def _example(raw: Mapping[str, Any]) -> Example | None:
    value, source, ref = raw.get("value"), raw.get("source"), raw.get("ref")
    if source not in ("human", "model", "judge", "outcome") or not isinstance(value, str | int | float):
        return None
    refs = {str(k): str(v) for k, v in cast(dict[str, Any], ref).items()} if isinstance(ref, dict) else {}
    return Example(
        topic=str(raw.get("topic", "")),
        subject=str(raw.get("subject", "")),
        value=value,
        source=cast(Source, source),
        by=str(raw.get("by", "")),
        at=_when(raw.get("at")),
        reason=str(raw.get("reason", "")),
        ref=refs,
    )


__all__ = ["JsonlExampleStore", "JsonlRunLedger", "MemoryExampleStore", "MemoryRunLedger", "union"]
