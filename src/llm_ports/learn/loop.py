"""Run a learning loop once: evidence, fit, gate, apply, and exactly one ledger row, whatever happens.

`run` is the part every loop used to rewrite: it drives a `Loop`'s stages, decides the verdict
with `gate.decide`, lets the loop act on it, and records the run. A stage that raises does not
escape: the run is recorded as `could_not_run` with the stage and the exception's type and
message, because a run that left no row is indistinguishable from a run that never happened.
An `apply` that raises keeps the verdict and is recorded as `apply_failed`.

Ledger row written for every run:

- `params`: the loop's `params`.
- `metrics`: the fit's `metrics`, then the reserved keys `RESERVED`: `learn_verdict`
  (`Verdict.as_dict`), `learn_model` (the fit's `model`), `learn_apply` (what `apply` returned,
  `apply_failed`, or "" when it was not called), `learn_apply_reason` and `learn_seconds`, and
  `learn_artifact_uri` (the fit's `artifact_uri`) only when the fit set one. A fit metric of a
  reserved name is replaced.

`record` writes the same verdict row for a gate run outside a `Loop`. `Run.as_dict` is a run as
plain fields (an admin endpoint's response) and `failed` says whether it should read red.

Invariants:
    - Exactly one `RunLedger.append` per call of `run` or `record`.
    - Never raises for a loop's failure (a stage, an attribute, a malformed fit) or a
      ledger's; both are in the returned value.

Rationale: ADR-0010.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from llm_ports.learn._protocols import Fit, Loop, RunLedger
from llm_ports.learn._types import VERDICT_KEY, Verdict, Written
from llm_ports.learn.gate import Unevaluated, could_not_run, decide

#: Metric keys `run` writes; a fit metric of the same name is replaced.
RESERVED = (VERDICT_KEY, "learn_model", "learn_apply", "learn_apply_reason", "learn_seconds", "learn_artifact_uri")


@dataclass(frozen=True)
class Run:
    """One run of a loop.

    Attributes:
        kind: The loop's run kind.
        subject: The loop's subject.
        verdict: The gate's verdict (`could_not_run` when a stage failed).
        fit: The fit, or None when the run stopped before it finished.
        applied: What `apply` returned; `apply_failed` when it raised; "" when not called.
        recorded: The ledger's answer to the row this run wrote; its `ref` is the row's id when
            the ledger mints one (what a model registry's `run_ref` links to).
        seconds: Wall time of the stages, excluding the ledger write.
    """

    kind: str
    subject: str
    verdict: Verdict
    fit: Fit | None
    applied: str
    recorded: Written
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return the run as plain fields, for an HTTP response or a scheduler's log.

        Returns:
            `kind`, `subject`, `decision`, `reason`, `blocker` (from the verdict), `applied`,
            `recorded` (the ledger's outcome), `recorded_reason`, `ref` (the row's id, or None),
            `metrics` (the fit's; `{}` without a fit) and `seconds`. `failed` reads it.
        """
        return {
            "kind": self.kind,
            "subject": self.subject,
            "decision": self.verdict.decision,
            "reason": self.verdict.reason,
            "blocker": self.verdict.blocker,
            "applied": self.applied,
            "recorded": self.recorded.outcome,
            "recorded_reason": self.recorded.reason,
            "ref": self.recorded.ref,
            "metrics": dict(self.fit.metrics) if self.fit is not None else {},
            "seconds": self.seconds,
        }


def failed(result: Run | Mapping[str, Any]) -> bool:
    """Return whether a run should read red to whatever scheduled it.

    Red is a run that measured nothing (`could_not_run`), whose ledger row did not land, or
    whose `apply` raised (`apply_failed`). A `fail` or `insufficient` verdict is a measurement,
    not a failure, and reads green.

    Args:
        result: A `Run`, or its `Run.as_dict` (as a scheduler holds it after an HTTP hop).

    Returns:
        True when the run is red. Never raises; a mapping missing a key reads red.
    """
    got = result.as_dict() if isinstance(result, Run) else result
    return got.get("decision") == "could_not_run" or got.get("recorded") != "ok" or got.get("applied") == "apply_failed"


def _error(stage: str, exc: Exception) -> str:
    """Return one line naming a failed stage and its exception."""
    return f"{stage}: {type(exc).__name__}: {exc}"[:500]


async def record(
    ledger: RunLedger,
    kind: str,
    verdict: Verdict,
    *,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Written:
    """Record one gate verdict as a ledger row, for a gate run outside a `Loop`.

    Args:
        ledger: Where the run is recorded.
        kind: The run kind.
        verdict: The verdict; its `subject` is the row's subject.
        params: The knobs the gate used.
        metrics: What it measured; `VERDICT_KEY` is set from `verdict`.

    Returns:
        The ledger's answer. Never raises.
    """
    try:
        row = dict(metrics or {})
        row[VERDICT_KEY] = verdict.as_dict()
        return await ledger.append(kind, subject=verdict.subject, params=dict(params or {}), metrics=row)
    except Exception as exc:  # a ledger that broke its never-raise contract is still answered
        return Written(0, "unavailable", _error("ledger raised", exc))


def _policy(value: object) -> Unevaluated:
    """Return a loop's `unevaluated` policy; an unknown value is the loop's error, not a silent "fail"."""
    if value == "fail" or value == "insufficient":
        return value
    raise ValueError(f"unevaluated must be 'fail' or 'insufficient', got {value!r}")


async def run[E](loop: Loop[E], ledger: RunLedger, *, clock: Callable[[], float] = time.monotonic) -> Run:
    """Run `loop` once and record it in `ledger`.

    Args:
        loop: The loop. An `unevaluated` attribute, when present, is passed to `gate.decide`;
            a value other than "fail" or "insufficient" makes the run `could_not_run`.
        ledger: Where the run is recorded.
        clock: A monotonic clock in seconds, for `seconds`; injectable for tests.

    Returns:
        The run. Never raises for a stage's exception (recorded as `could_not_run`, or as
        `apply_failed` for `apply`), a loop attribute that raises or a fit whose `metrics` or
        `model` is not a mapping (`could_not_run`, stage `loop` or `fit`), or a ledger failure
        (`Run.recorded` carries it). A loop whose `kind` cannot be read is recorded under its
        class name.
    """
    start = clock()
    kind, subject, params = type(loop).__name__, "", {}
    fit: Fit | None = None
    fit_metrics: dict[str, Any] = {}
    model: dict[str, Any] = {}
    stage = "loop"
    try:
        kind = str(loop.kind)
        subject = str(loop.subject)
        params = dict(loop.params)
        policy = _policy(getattr(loop, "unevaluated", "fail"))
        stage = "evidence"
        evidence = await loop.evidence()
        stage = "fit"
        fit = loop.fit(evidence)
        fit_metrics, model = dict(fit.metrics), dict(fit.model)
        stage = "clauses"
        verdict = decide(kind, loop.clauses(fit, evidence), subject=subject, unevaluated=policy)
    except Exception as exc:  # a loop's own failure becomes a recorded could_not_run
        verdict = could_not_run(kind, _error(stage, exc), subject=subject)
        if stage != "clauses":
            fit, fit_metrics, model = None, {}, {}
    applied, apply_reason = "", ""
    if fit is not None and verdict.decision != "could_not_run":
        try:
            applied = str(await loop.apply(verdict, fit))
        except Exception as exc:  # the verdict stands; the failed action is on the record
            applied, apply_reason = "apply_failed", _error("apply", exc)
    seconds = clock() - start
    metrics: dict[str, Any] = {
        **fit_metrics,
        "learn_model": model,
        "learn_apply": applied,
        "learn_apply_reason": apply_reason,
        "learn_seconds": round(seconds, 3),
    }
    artifact_uri = getattr(fit, "artifact_uri", None)  # a duck-typed fit may not carry one
    if isinstance(artifact_uri, str):
        metrics["learn_artifact_uri"] = artifact_uri
    else:
        metrics.pop("learn_artifact_uri", None)  # reserved: a fit metric never poses as one
    recorded = await record(ledger, kind, verdict, params=params, metrics=metrics)
    return Run(kind, subject, verdict, fit, applied, recorded, seconds)


__all__ = ["RESERVED", "Run", "failed", "record", "run"]
