"""The learning loop's ports: a run ledger, a store of labelled examples, and a loop.

- `RunLedger`: where every run of every loop is recorded, one row per run.
- `ExampleStore` / `WritableExampleStore`: where labelled examples are read (and, for some
  stores, written). One contract over many physical stores; a consumer reads, never copies.
- `ModelRegistry`: numbered versions of a fitted model and named aliases over them (`champion`,
  `challenger`), the shape a tracker's model registry has, so promotion can live in one place.
- `Loop`: one learning loop's four stages (evidence, fit, clauses, apply), which `loop.run` drives.

Async, as the stores a loop reads usually are; a synchronous backend runs its calls in a thread.

Invariants:
    - No I/O: the module holds Protocols only.
    - A ledger, store or registry never raises for a backend failure; the failure is the result's
      `outcome`. `llm_ports.testing.conformance` checks a backend against these contracts.

Rationale: ADR-0010, ADR-0016.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from llm_ports.learn._types import (
    Clause,
    Decision,
    Example,
    ExamplesRead,
    RowRead,
    Source,
    Verdict,
    VersionRead,
    Written,
)


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
            metrics: What it measured; JSON-able. A gated run carries its verdict under
                `VERDICT_KEY`.

        Returns:
            `written == 1` on success, else an `unavailable` or `request_fault` outcome. A ledger
            that ids its rows puts the id in `ref`; the row read back carries the same `ref`.
        """
        ...

    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        """Read the newest row of one kind.

        Args:
            kind: The run kind.
            subject: Only rows about this subject; None for any.
            decisions: Only rows whose recorded verdict's decision is one of these (a row with no
                verdict never matches); None for any row. A
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
        artifact_uri: Where a model too big for `model` was written (a file, a bucket, a
            tracker's artifact store); None when `model` is the whole model. `loop.run` records
            it in the ledger row.

    Invariants:
        - Plain data, so a fit made in a worker process pickles back whole.
    """

    model: Mapping[str, Any] = field(default_factory=dict[str, Any])
    metrics: Mapping[str, Any] = field(default_factory=dict[str, Any])
    artifact_uri: str | None = None


class ModelRegistry(Protocol):
    """Numbered versions of fitted models, with named aliases (`champion`) pointing at one each.

    Mirrors a tracker's model registry (versions from 1 per name; an alias moves, a version never
    changes), so a backend is a thin mapping.
    """

    async def register(
        self,
        name: str,
        *,
        model: Mapping[str, Any],
        artifact_uri: str | None = None,
        run_ref: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Written:
        """Register a new version of `name` (created on first use).

        Args:
            name: The registered model's name.
            model: The fitted parameters; JSON-able.
            artifact_uri: Where a model too big for `model` lives; None when `model` is whole.
            run_ref: The ledger row that produced it (`Written.ref`).
            tags: Free string labels.

        Returns:
            `written == 1` and the new version number as a decimal string in `ref`, else a
            `request_fault` (a value the registry refuses) or `unavailable` outcome.
        """
        ...

    async def set_alias(self, name: str, alias: str, version: int) -> Written:
        """Point `alias` of `name` at `version`, moving it from any earlier version.

        Returns:
            `written == 1` on success; `request_fault` when the version does not exist or is not an
            `int`; else
            `unavailable`.
        """
        ...

    async def by_alias(self, name: str, alias: str) -> VersionRead:
        """Read the version `alias` of `name` points at.

        Returns:
            The version, `ok` with None when the model or alias does not exist, or `unavailable`.
        """
        ...

    async def get(self, name: str, version: int) -> VersionRead:
        """Read one version of `name`.

        Returns:
            The version, `ok` with None when it does not exist, `request_fault` when `version` is
            not an `int` (`Written.ref` is its decimal string), or `unavailable`.
        """
        ...


class Loop[E](Protocol):
    """One learning loop: read evidence, fit, say which clauses the fit must clear, act on it.

    `loop.run` drives the stages, decides the verdict with `gate.decide`, calls `apply` and
    records one ledger row, so a loop holds only what is specific to it. A loop whose verdict
    needs a policy other than the default (`unevaluated`) sets the `unevaluated` attribute.
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
        """Fit or measure. May raise; `run` records the failure.

        Pure (no I/O, no shared state), so a heavier engine can ship it to a remote worker.
        """
        ...

    def clauses(self, fit: Fit, evidence: E) -> Sequence[Clause]:
        """Return the gate's evaluated clauses for this fit. May raise; `run` records the failure."""
        ...

    async def apply(self, verdict: Verdict, fit: Fit) -> str:
        """Act on the verdict: store the fit on `pass`, keep the incumbent otherwise, or nothing.

        Called once per run that reached a decision (never on `could_not_run`), before the run
        is recorded, so the row says what was done.

        Returns:
            One word for the record (`stored`, `held`, `noop`, ...). May raise; `run` records
            `apply_failed` and keeps the verdict.
        """
        ...


__all__ = ["ExampleStore", "Fit", "Loop", "ModelRegistry", "RunLedger", "WritableExampleStore"]
