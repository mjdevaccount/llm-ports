"""experiments -- a graded eval recorded as a Langfuse dataset experiment; a dropped item is a named
outcome, never a clean ratio over a shrunken denominator. Against a fake SDK client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from llm_ports.adapters.langfuse import experiments as ex
from llm_ports.testing.langfuse import RefusingLangfuse


class FakeClient:
    """The two SDK calls `record_rows` makes, plus `flush`; `drop` names items the SDK drops."""

    def __init__(self, ids: list[str], drop: frozenset[str] = frozenset()) -> None:
        self.items = [SimpleNamespace(id=i, input={"q": i}) for i in ids]
        self.drop = drop
        self.calls: list[dict[str, Any]] = []
        self.flushed = 0

    def get_dataset(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(name=name, items=self.items)

    def run_experiment(self, **kw: Any) -> SimpleNamespace:
        self.calls.append(kw)
        results = []
        for item in kw["data"]:
            out = kw["task"](item=item)
            evals = [] if item.id in self.drop else [e for f in kw["evaluators"] for e in f(output=out)]
            results.append(SimpleNamespace(item=item, output=out, evaluations=evals))
        return SimpleNamespace(item_results=results)

    def flush(self) -> None:
        self.flushed += 1


ROWS = [{"id": "a", "text": "one", "echo": 0.0}, {"id": "b", "text": "two", "echo": 1.0}]


def _record(c: Any, rows: list[dict[str, Any]] = ROWS) -> ex.ExperimentRun:
    return ex.record_rows(
        c,
        dataset="d1",
        run_name="r1",
        rows=rows,
        output=lambda r: {"text": r["text"]},
        grade=lambda r: [("echo", r["echo"], "NUMERIC")],
        metadata={"git": "abc"},
    )


def test_every_row_graded_is_ok_and_books_each_rows_own_grade() -> None:
    c = FakeClient(["a", "b", "z"])  # "z" has no row: skipped, not counted
    run = _record(c)
    assert run.ok and (run.handed, run.graded, run.reason) == (2, 2, None)
    call = c.calls[0]
    assert call["name"] == call["run_name"] == "r1" and call["max_concurrency"] == 1
    assert call["metadata"] == {"git": "abc"} and [i.id for i in call["data"]] == ["a", "b"]
    by_item = {r.item.id: r for r in run.result.item_results}
    assert by_item["b"].output == {"id": "b", "text": "two"}
    ev = by_item["b"].evaluations[0]
    assert (ev.name, ev.value, ev.data_type) == ("echo", 1.0, "NUMERIC")
    assert c.flushed == 1


def test_a_dropped_item_is_invalid_and_says_how_many() -> None:
    run = _record(FakeClient(["a", "b"], drop=frozenset({"b"})))
    assert run.outcome == "invalid" and not run.ok
    assert (run.handed, run.graded) == (2, 1)
    assert run.reason is not None and run.reason.startswith("1/2 items")


def test_a_server_that_refuses_is_unavailable_never_a_raise() -> None:
    run = _record(RefusingLangfuse())
    assert run.outcome == "unavailable" and run.handed == 0
    assert run.reason is not None and "ConnectionError" in run.reason


def test_a_run_that_raises_is_unavailable() -> None:
    class Boom(FakeClient):
        def run_experiment(self, **kw: Any) -> SimpleNamespace:
            raise TimeoutError("read timed out")

    run = _record(Boom(["a", "b"]))
    assert run.outcome == "unavailable" and run.handed == 2 and run.graded == 0
    assert run.reason is not None and "TimeoutError" in run.reason
