"""The learning loop's ports: a run ledger, a store of labelled examples, and a loop.

- `RunLedger`: where every run of every loop is recorded, one row per run.
- `ExampleStore` / `WritableExampleStore`: where labelled examples are read (and, for some
  stores, written). One contract over many physical stores; a consumer reads, never copies.
- `Loop`: one learning loop's three stages, which `loop.run` drives.

Async, as the stores a loop reads usually are; a synchronous backend runs its calls in a thread.

Invariants:
    - No I/O: the module holds Protocols only.
    - A ledger or store never raises for a backend failure; the failure is the result's `outcome`.

Rationale: ADR-0010.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from llm_ports.learn._types import Clause, Decision, Example, ExamplesRead, RowRead, Source, Written


class RunLedger(Protocol):
    """The record of every run: what ran, on what, with which knobs, and what it decided."""

    async def append(
        self, kind: str, *, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> Written:
        """Record one run.

        Args:
            kind: The loop's run kind.
            subject: What the run was about; "" for a loop with one subject.
            params: The knobs the run used; JSON-able.
            metrics: What it measured; JSON-able. A gated run carries `verdict` and `gate`.

        Returns:
            `written == 1` on success, else an `unavailable` or `request_fault` outcome.
        """
        ...

    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        """Read the newest row of one kind.

        Args:
            kind: The run kind.
            subject: Only rows about this subject; None for any.
            decisions: Only rows whose `metrics.verdict` is one of these; None for any row. A
                gate's decisive read passes `("pass", "fail")`, so a later `could_not_run` does not
                close what a decisive run opened.

        Returns:
            The row (None when there is none), or an `unavailable` outcome.
        """
        ...


class ExampleStore(Protocol):
    """Labelled examples, read by topic."""

    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        """Read the examples of `topics`, labelled at or after `since`, from `source` or any source.

        Returns:
            The examples, or an `unavailable` outcome; an unreadable store never reads as empty.
        """
        ...


class WritableExampleStore(ExampleStore, Protocol):
    """A store that also accepts examples written by code."""

    async def put(self, examples: Sequence[Example]) -> Written:
        """Store examples.

        Returns:
            How many landed, or a `request_fault` naming an example the store refuses (a store
            may refuse a `human` label written by code: a person did not give it).
        """
        ...


@dataclass(frozen=True)
class Fit:
    """What a loop's fit stage produced.

    Attributes:
        model: The fitted parameters a consumer applies (a weight, a threshold, coefficients);
            JSON-able. `{}` for a loop that measures without fitting.
        metrics: What the fit measured (sample size, error, lift); JSON-able.
    """

    model: Mapping[str, Any] = field(default_factory=dict[str, Any])
    metrics: Mapping[str, Any] = field(default_factory=dict[str, Any])


class Loop[E](Protocol):
    """One learning loop: read evidence, fit, and say which clauses the fit must clear.

    `loop.run` drives the stages, decides the verdict with `gate.decide` and records one ledger
    row, so a loop holds only what is specific to it.
    """

    @property
    def kind(self) -> str:
        """The run kind its ledger rows carry."""
        ...

    @property
    def subject(self) -> str:
        """What it learns about (a symbol, a role, a model); "" for one subject."""
        ...

    @property
    def params(self) -> Mapping[str, Any]:
        """The knobs of this run (window, thresholds); JSON-able."""
        ...

    async def evidence(self) -> E:
        """Read the evidence (examples, outcomes, series). May raise; `run` records the failure."""
        ...

    def fit(self, evidence: E) -> Fit:
        """Fit or measure. May raise; `run` records the failure."""
        ...

    def clauses(self, fit: Fit, evidence: E) -> Sequence[Clause]:
        """Return the gate's evaluated clauses for this fit. May raise; `run` records the failure."""
        ...


__all__ = ["ExampleStore", "Fit", "Loop", "RunLedger", "WritableExampleStore"]
