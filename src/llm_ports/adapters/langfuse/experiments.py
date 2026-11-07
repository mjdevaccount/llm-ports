"""Record a graded evaluation as a Langfuse dataset experiment (extra `[langfuse]`).

`seed` writes a golden set into a dataset, idempotently: each item's id is the caller's or a hash
of its input, and Langfuse upserts an item by id, so a re-seed rewrites rather than duplicates.
`run` drives the SDK's own experiment runner (`Langfuse.run_experiment`): it opens one trace per
dataset item, links the trace to the item, and books each grade as a score on it. `record_rows`
records rows already graded offline (a golden-set run that graded a recording) as one run on a
dataset: the task hands back the row's public output plus its id, and the evaluator looks the row
up by that id, because an SDK evaluator sees input, output, expected output and metadata, never the
item.

The SDK logs a task or evaluator that raised and drops or leaves that item ungraded, so a run that
only counted what came back would report a clean ratio over a shrunken denominator. Here a run that
graded fewer items than it was handed is an `invalid` outcome naming the shortfall.

Example:
    run = record_rows(client, dataset="summary_golden_v1", run_name="pack-1a2b", rows=rows,
                      output=lambda r: {"text": r["text"]}, grade=lambda r: [("echo", r["echo"], "NUMERIC")])
    if not run.ok:
        ...  # the run did not finish looking; `run.reason` says why

Invariants:
    - Never raises for a server or SDK failure: the failure is the outcome.
    - `max_concurrency` defaults to 1: an offline grade drives one model or CLI at a time.

Rationale: ADR-0001.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_ports._extras import extra
from llm_ports.ports import Outcome
from llm_ports.ports.outcomes import failure_reason

with extra("langfuse"):
    from langfuse import Evaluation

#: What one grader says about one row: `(score name, value, data type)`; the data type is one of
#: Langfuse's `NUMERIC`, `CATEGORICAL` or `BOOLEAN`.
Grade = tuple[str, float, str]


@dataclass(frozen=True)
class ExperimentRun:
    """What `run` and `record_rows` return.

    Attributes:
        outcome: `ok` when every item was graded; `invalid` when the SDK dropped or left an item
            ungraded; `unavailable` when the run itself failed.
        handed: Items the run was handed.
        graded: Items that came back with at least one grade.
        reason: One line naming the failure; None on `ok`.
        result: The SDK's `ExperimentResult` (its run URL, item results); None when the run failed.
    """

    outcome: Outcome
    handed: int
    graded: int = 0
    reason: str | None = None
    result: Any = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


def run(
    client: Any,
    *,
    run_name: str,
    items: Sequence[Any],
    task: Callable[..., Any],
    grade: Callable[..., Iterable[Grade]],
    metadata: Mapping[str, str] | None = None,
    max_concurrency: int = 1,
) -> ExperimentRun:
    """Run `task` over dataset `items` and book `grade`'s scores on each item's trace.

    Args:
        client: A Langfuse SDK 5 client, built by the caller.
        run_name: The experiment and run name.
        items: Dataset items (`client.get_dataset(name).items`), so each trace links to its item.
        task: The SDK task: called with `item=` (and other keywords), returns the item's output.
        grade: Called with the SDK evaluator's keywords (`output=`, `expected_output=`, ...);
            returns the item's `Grade`s.
        metadata: Run metadata.
        max_concurrency: Items in flight at once.

    Returns:
        The run, whose outcome is `ok` only when every handed item was graded. Never raises.
    """

    def evaluator(**kw: Any) -> list[Any]:
        return [Evaluation(name=n, value=v, data_type=dt) for n, v, dt in grade(**kw)]  # type: ignore[arg-type]  # SDK takes a Literal; Grade carries a str

    try:
        res = client.run_experiment(
            name=run_name,
            run_name=run_name,
            data=list(items),
            task=task,
            evaluators=[evaluator],
            max_concurrency=max_concurrency,
            metadata=dict(metadata or {}),
        )
        client.flush()
    except Exception as exc:  # a server or SDK failure is the run's outcome
        return ExperimentRun("unavailable", len(items), reason=failure_reason(exc))
    graded = sum(1 for r in res.item_results if r.evaluations)
    if graded < len(items):
        reason = f"{len(items) - graded}/{len(items)} items dropped or ungraded (the task or an evaluator raised)"
        return ExperimentRun("invalid", len(items), graded, reason, res)
    return ExperimentRun("ok", len(items), graded, None, res)


def record_rows(
    client: Any,
    *,
    dataset: str,
    run_name: str,
    rows: Sequence[Mapping[str, Any]],
    output: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    grade: Callable[[Mapping[str, Any]], Iterable[Grade]],
    metadata: Mapping[str, str] | None = None,
    id_field: str = "id",
) -> ExperimentRun:
    """Record rows graded offline as one run on `dataset`.

    Only items whose id matches a row's `id_field` are run; an item with no row is skipped, not
    counted.

    Args:
        client: A Langfuse SDK 5 client.
        dataset: The dataset's name.
        run_name: The run name.
        rows: The graded rows, each carrying `id_field`.
        output: A row to the public output recorded on its trace.
        grade: A row to its `Grade`s.
        metadata: Run metadata.
        id_field: The row key that matches a dataset item's id.

    Returns:
        The run (see `run`); `unavailable` when the dataset could not be read. Never raises.
    """
    by_id = {r[id_field]: r for r in rows}
    try:
        items = [i for i in client.get_dataset(dataset).items if i.id in by_id]
    except Exception as exc:  # the store is down or the dataset is missing
        return ExperimentRun("unavailable", 0, reason=failure_reason(exc))

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        return {"id": item.id, **output(by_id[item.id])}

    def graded(*, output: Mapping[str, Any], **_: Any) -> Iterable[Grade]:
        return grade(by_id[output["id"]])

    return run(client, run_name=run_name, items=items, task=task, grade=graded, metadata=metadata)


@dataclass(frozen=True)
class Seeded:
    """What `seed` returns.

    Attributes:
        outcome: `ok` when every item was written; `invalid` when some writes failed;
            `unavailable` when the dataset could be neither created nor read.
        handed: Items `seed` was handed.
        written: Items the store accepted.
        ids: The id of every item written, in input order.
        reason: One line naming the failure; None on `ok`.
    """

    outcome: Outcome
    handed: int
    written: int = 0
    ids: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


def input_id(value: Any) -> str:
    """A stable id for a dataset item: the first 32 hex digits of the SHA-256 of its canonical JSON.

    Args:
        value: The item's input; any JSON-serialisable value (a non-JSON leaf is stringified).

    Returns:
        The id; equal inputs (key order aside) give equal ids. Never raises.
    """
    blob = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def seed(
    client: Any,
    *,
    dataset: str,
    items: Sequence[Mapping[str, Any]],
    input: Callable[[Mapping[str, Any]], Any],  # the SDK's own field name
    expected: Callable[[Mapping[str, Any]], Any] | None = None,
    metadata: Callable[[Mapping[str, Any]], Any] | None = None,
    item_id: Callable[[Mapping[str, Any]], str] | None = None,
    description: str | None = None,
) -> Seeded:
    """Write `items` into `dataset`, creating the dataset when missing; a re-seed upserts.

    Args:
        client: A Langfuse SDK 5 client.
        dataset: The dataset's name.
        items: The golden rows, in any shape the projections read.
        input: A row to the item's input.
        expected: A row to the item's expected output; omitted, none is written.
        metadata: A row to the item's metadata; omitted, none is written.
        item_id: A row to the item's id; omitted, `input_id` of the item's input, so the same input
            is the same item across seeds.
        description: The dataset's description, written on creation.

    Returns:
        `ok` with every id when every item was written; `invalid` naming the shortfall when some
        writes failed; `unavailable` when the dataset could be neither created nor read. Never
        raises.

    Invariants:
        - Idempotent: seeding the same rows twice leaves one item per id.
        - A failed write never stops the rest; the outcome counts it.
    """
    try:
        client.create_dataset(name=dataset, description=description)
    except Exception as exc:  # the name may already exist; only an unreadable dataset is fatal
        try:
            client.get_dataset(dataset)
        except Exception:
            return Seeded("unavailable", len(items), reason=failure_reason(exc))
    ids: list[str] = []
    failed: list[str] = []
    for row in items:
        value = input(row)
        iid = item_id(row) if item_id else input_id(value)
        kw: dict[str, Any] = {"dataset_name": dataset, "id": iid, "input": value}
        if expected is not None:
            kw["expected_output"] = expected(row)
        if metadata is not None:
            kw["metadata"] = metadata(row)
        try:
            client.create_dataset_item(**kw)
        except Exception as exc:  # one write failing is counted, not fatal
            failed.append(failure_reason(exc))
            continue
        ids.append(iid)
    try:
        client.flush()
    except Exception as exc:  # a flush that fails leaves the writes unconfirmed
        return Seeded("invalid", len(items), len(ids), tuple(ids), f"flush failed: {failure_reason(exc)}")
    if failed:
        reason = f"{len(failed)}/{len(items)} items not written: {failed[0]}"
        return Seeded("invalid", len(items), len(ids), tuple(ids), reason)
    return Seeded("ok", len(items), len(ids), tuple(ids))


__all__ = ["ExperimentRun", "Grade", "Seeded", "input_id", "record_rows", "run", "seed"]
