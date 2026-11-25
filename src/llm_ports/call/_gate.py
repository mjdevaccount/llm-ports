"""Acceptance gates: the graders a role names in `gates` decide whether a reply is returned.

A grader measures a reply; a gate refuses one. The graders in `Role.gates` run twice over the
same seam every caller already has:

- as part of the call's validator (`armed`), so a reply that fails a gate is re-asked with the
  failure named, while `Role.output_retries` remain;
- over the kept reply's scores (`refused`), so a reply that still fails is returned `invalid`
  with reason `gate: <grader>: <reason>` and no value.

The scores are written and handed to `Hooks.on_scores` either way, so a failure counted before a
role was gated is still counted after.

Invariants:
    - Pure, and never raises: a gate is a grader run by `graders.score_one`, which turns a crash
      into a failed score.
    - The caller's own validator runs first; a gate is asked only about a reply the caller accepts.
    - A reply that passes every gate is returned exactly as an ungated role's reply.

Rationale: ADR-0028.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from llm_ports.call.graders import CallGrader, CallScore, score_one
from llm_ports.call.roles import Role
from llm_ports.ports import Generated, RetryMessage, Validator, failure
from llm_ports.ports.outcomes import REASON_MAX

#: The stamp key naming the gates a refused reply failed: `res.stamp[GATES_FAILED]`, a tuple of
#: grader names, present exactly when a gate refused the reply.
GATES_FAILED = "gates_failed"


@dataclass(frozen=True)
class GateProblem:
    """One failed gate on one reply, as a validator problem.

    Attributes:
        grader: The gate's grader name.
        reason: The grader's reason, naming the violated property.
    """

    grader: str
    reason: str

    def __str__(self) -> str:
        return f"{self.grader}: {self.reason}"


def failed(role: Role, prompt: str, value: Any, table: Mapping[str, CallGrader] | None) -> list[CallScore]:
    """Return the failed scores of the role's gates over one reply, in the role's order."""
    scores = (score_one(name, prompt, value, role, table) for name in role.gates)
    return [s for s in scores if not s.passed]


def retry_text(problems: Sequence[Any]) -> str:
    """Return the re-ask message for a reply its gates refused."""
    named = "; ".join(str(p) for p in problems)
    return (
        f"Your reply was refused by a check and was not accepted: {named}. "
        "Write the reply again with that fixed. Keep everything the check did not name."
    )


def armed[T](
    role: Role,
    prompt: str,
    table: Mapping[str, CallGrader] | None,
    validate: Validator[T] | None,
    retry_message: RetryMessage | None,
) -> tuple[Validator[T] | None, RetryMessage | None]:
    """Return the call's validator and re-ask message with the role's gates added.

    A role with no gates gets its arguments back unchanged. Otherwise the caller's validator runs
    first and its problems win; a reply it accepts is scored by the gates, and each failure is a
    `GateProblem`. The re-ask message for gate problems is `retry_text`; any other problem list
    goes to the caller's `retry_message`.
    """
    if not role.gates:
        return validate, retry_message

    def check(value: T) -> Sequence[Any]:
        own = list(validate(value)) if validate is not None else []
        return own or [GateProblem(s.name, s.reason) for s in failed(role, prompt, value, table)]

    def message(problems: Sequence[Any]) -> str:
        if all(isinstance(p, GateProblem) for p in problems):
            return retry_text(problems)
        return retry_message(problems) if retry_message is not None else "; ".join(map(str, problems))

    return check, message


def refused[T](role: Role, res: Generated[T]) -> Generated[T]:
    """Return a graded `ok` reply as `invalid` when a gate's score failed, else unchanged.

    The refusal keeps the stamp (its `scores`, and `gates_failed` naming the gates) and drops the
    value: a caller that reads only `outcome` cannot store a refused reply.
    """
    if res.outcome != "ok" or not role.gates:
        return res
    scores: Sequence[CallScore] = res.stamp.get("scores") or ()
    bad = [s for s in scores if s.name in role.gates and not s.passed]
    if not bad:
        return res
    reason = ("gate: " + "; ".join(f"{s.name}: {s.reason}" for s in bad))[:REASON_MAX]
    return replace(
        res,
        value=None,
        outcome="invalid",
        reason=reason,
        failure=failure("invalid_output", reason=reason),
        stamp={**res.stamp, GATES_FAILED: tuple(s.name for s in bad)},
    )


__all__ = ["GATES_FAILED", "GateProblem"]
