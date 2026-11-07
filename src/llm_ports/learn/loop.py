"""Run a learning loop once: evidence, fit, gate, and exactly one ledger row, whatever happens.

`run` is the part every loop used to rewrite: it drives a `Loop`'s three stages, decides the
verdict with `gate.decide`, and records the run. A stage that raises does not escape: the run is
recorded as `could_not_run` with the stage and the exception's type and message, because a run
that left no row is indistinguishable from a run that never happened.

Ledger row written for every run:

- `params`: the loop's `params`.
- `metrics`: the fit's `metrics`, then `verdict` (the decision), `gate` (`Verdict.as_dict`),
  `model` (the fit's `model`) and `seconds`. Those four keys are reserved: they replace a fit
  metric of the same name.

Invariants:
    - Exactly one `RunLedger.append` per call.
    - Never raises for a loop's failure or a ledger's; both are in the returned `Run`.

Rationale: ADR-0010.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_ports.learn._protocols import Fit, Loop, RunLedger
from llm_ports.learn._types import Verdict, Written
from llm_ports.learn.gate import could_not_run, decide

#: Metric keys `run` writes; a fit metric of the same name is replaced.
RESERVED = ("verdict", "gate", "model", "seconds")


@dataclass(frozen=True)
class Run:
    """One run of a loop.

    Attributes:
        kind: The loop's run kind.
        subject: The loop's subject.
        verdict: The gate's verdict (`could_not_run` when a stage failed).
        fit: The fit, or None when the run stopped before it finished.
        recorded: The ledger's answer to the row this run wrote.
        seconds: Wall time of the stages, excluding the ledger write.
    """

    kind: str
    subject: str
    verdict: Verdict
    fit: Fit | None
    recorded: Written
    seconds: float


async def run[E](loop: Loop[E], ledger: RunLedger, *, clock: Callable[[], float] = time.monotonic) -> Run:
    """Run `loop` once and record it in `ledger`.

    Args:
        loop: The loop.
        ledger: Where the run is recorded.
        clock: A monotonic clock in seconds, for `seconds`; injectable for tests.

    Returns:
        The run. Never raises for a stage's exception (recorded as `could_not_run`) or a ledger
        failure (`Run.recorded` carries it).
    """
    start = clock()
    fit: Fit | None = None
    stage = "evidence"
    try:
        evidence = await loop.evidence()
        stage = "fit"
        fit = loop.fit(evidence)
        stage = "clauses"
        verdict = decide(loop.kind, loop.clauses(fit, evidence), subject=loop.subject)
    except Exception as exc:  # a loop's own failure becomes a recorded could_not_run
        verdict = could_not_run(loop.kind, f"{stage}: {type(exc).__name__}: {exc}"[:500], subject=loop.subject)
        if stage != "clauses":
            fit = None
    seconds = clock() - start
    metrics: dict[str, Any] = dict(fit.metrics) if fit is not None else {}
    metrics.update(
        verdict=verdict.decision,
        gate=verdict.as_dict(),
        model=dict(fit.model) if fit is not None else {},
        seconds=round(seconds, 3),
    )
    try:
        recorded = await ledger.append(loop.kind, subject=loop.subject, params=dict(loop.params), metrics=metrics)
    except Exception as exc:  # a ledger that broke its never-raise contract is still recorded
        recorded = Written(0, "unavailable", f"ledger raised {type(exc).__name__}: {exc}"[:500])
    return Run(loop.kind, loop.subject, verdict, fit, recorded, seconds)


__all__ = ["RESERVED", "Run", "run"]
