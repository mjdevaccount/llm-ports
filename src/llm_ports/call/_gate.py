"""Acceptance gates: the graders a role names in `gates` decide whether a reply is returned.

A grader measures a reply; a gate refuses one. The graders in `Role.gates` run twice over the
same seam every caller already has:

- as part of the call's validator (`armed`), so a reply that fails a gate is re-asked with the
  failure named, while `Role.output_retries` remain;
- over the kept reply's scores (`refused`), so a reply that still fails is returned `invalid`
  with reason `gate: <grader>: <reason>` and no value.

The scores are written and handed to `Hooks.on_scores` either way, so a failure counted before a
role was gated is still counted after.

A gate is named by its KEY in the grader table, the word `Role.gates` carries. A score is matched
to its gate by position in `Role.graders`, never by `CallScore.name`: a table entry may bind a
built-in under another key (`{"complete_nullable": partial(schema_complete, ...)}`), and its score
still says `schema_complete`.

Invariants:
    - Never raises: a gate is a grader run by `graders.score_one`, which turns a crash into a
      failed score. Scoring is pure; building a re-ask message records the re-ask (`Armed.reasked`).
    - A gate whose grader does not read this call's inputs (images, a web reply) is not applied,
      and the result says so: `stamp[GATES_DROPPED]`.
    - The caller's own validator runs first; a gate is asked only about a reply the caller accepts.
    - A reply that passes every gate is returned exactly as an ungated role's reply.
    - A gate with no score refuses: a gate that was not run never reads as a pass.

Rationale: ADR-0028.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from llm_ports.call.graders import GRADERS, CallGrader, CallScore, score_one
from llm_ports.call.roles import Role
from llm_ports.ports import Generated, RetryMessage, Validator, failure
from llm_ports.ports.outcomes import REASON_MAX

#: The stamp key naming the gates a refused reply failed: `res.stamp[GATES_FAILED]`, a tuple of
#: grader table keys, present exactly when a gate refused the reply.
GATES_FAILED = "gates_failed"
#: The stamp key naming the gates a re-ask was sent for: `res.stamp[GATES_REASKED]`, a tuple of
#: grader table keys in first-seen order, present exactly when a gate caused a re-ask. On an `ok`
#: reply it is the only witness that an earlier reply of the same call was refused.
GATES_REASKED = "gates_reasked"
#: The stamp key naming the gates the call did NOT apply: `res.stamp[GATES_DROPPED]`, a tuple of
#: grader table keys, present exactly when a gated role was called with inputs its gate's grader
#: does not read (`graders.graders_for_inputs`: images, a web reply). The reply was not checked by
#: those gates, and this is the only witness of it.
GATES_DROPPED = "gates_dropped"


@dataclass(frozen=True)
class GateProblem:
    """One failed gate on one reply, as a validator problem.

    Attributes:
        grader: The gate's key in the grader table, as `Role.gates` names it.
        reason: The grader's reason, naming the violated property.
    """

    grader: str
    reason: str

    def __str__(self) -> str:
        return f"{self.grader}: {self.reason}"


@dataclass
class Armed[T]:
    """One call's validator seam with the role's gates added.

    Attributes:
        validate: The validator handed to the `Generator`; the caller's own for an ungated role.
        retry_message: The re-ask message builder handed to the `Generator`.
        reasked: The gate problems of each re-ask sent so far, in order. Filled when the
            `Generator` asks for a re-ask message over gate problems.
    """

    validate: Validator[T] | None
    retry_message: RetryMessage | None
    reasked: list[tuple[GateProblem, ...]] = field(default_factory=list)

    @property
    def gates_reasked(self) -> tuple[str, ...]:
        """The gates a re-ask was sent for, each once, in first-seen order."""
        return tuple(dict.fromkeys(p.grader for problems in self.reasked for p in problems))


def unknown(role: Role, table: Mapping[str, CallGrader] | None) -> list[str]:
    """Return the role's gates that name no grader in `table` (default `GRADERS`).

    Such a gate can only crash, so it would refuse every reply after spending the re-ask.
    """
    have = GRADERS if table is None else table
    return [g for g in role.gates if g not in have]


def failed(role: Role, prompt: str, value: Any, table: Mapping[str, CallGrader] | None) -> list[GateProblem]:
    """Return the role's failed gates over one reply, in the role's order, each named by its key."""
    scored = ((key, score_one(key, prompt, value, role, table)) for key in role.gates)
    return [GateProblem(key, s.reason) for key, s in scored if not s.passed]


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
    on_reask: Callable[[Sequence[GateProblem]], None] | None = None,
) -> Armed[T]:
    """Return the call's validator and re-ask message with the role's gates added.

    A role with no gates gets its arguments back unchanged. Otherwise the caller's validator runs
    first and its problems win; a reply it accepts is scored by the gates, and each failure is a
    `GateProblem`. The re-ask message for gate problems is `retry_text`; any other problem list
    goes to the caller's `retry_message`.

    A `Generator` asks for the re-ask message exactly when it sends a re-ask, so that is where a
    gate re-ask is recorded (`Armed.reasked`) and `on_reask` is called with its problems.
    """
    if not role.gates:
        return Armed(validate, retry_message)
    arm: Armed[T]

    def check(value: T) -> Sequence[Any]:
        own = list(validate(value)) if validate is not None else []
        return own or failed(role, prompt, value, table)

    def message(problems: Sequence[Any]) -> str:
        gates = [p for p in problems if isinstance(p, GateProblem)]
        if gates:  # recorded whenever a gate is among the reasons, whatever else the list holds
            arm.reasked.append(tuple(gates))
            if on_reask is not None:
                on_reask(gates)
        if gates and len(gates) == len(problems):
            return retry_text(gates)
        return retry_message(problems) if retry_message is not None else "; ".join(map(str, problems))

    arm = Armed(check, message)
    return arm


def refused[T](role: Role, res: Generated[T]) -> Generated[T]:
    """Return a graded `ok` reply as `invalid` when a gate's score failed, else unchanged.

    `stamp["scores"]` is read in the order of `role.graders` (`graders.score_all` writes it so), and
    a gate with no score there refuses. The refusal keeps the stamp (its `scores`, and
    `gates_failed` naming the gates by key) and drops the value: a caller that reads only `outcome`
    cannot store a refused reply.
    """
    if res.outcome != "ok" or not role.gates:
        return res
    scores: Sequence[CallScore] = res.stamp.get("scores") or ()
    by_key = dict(zip(role.graders, scores, strict=False))
    bad = [
        GateProblem(g, by_key[g].reason if g in by_key else "the gate was not scored")
        for g in role.gates
        if g not in by_key or not by_key[g].passed
    ]
    if not bad:
        return res
    reason = ("gate: " + "; ".join(map(str, bad)))[:REASON_MAX]
    return replace(
        res,
        value=None,
        outcome="invalid",
        reason=reason,
        failure=failure("invalid_output", reason=reason),
        stamp={**res.stamp, GATES_FAILED: tuple(p.grader for p in bad)},
    )


__all__ = ["GATES_DROPPED", "GATES_FAILED", "GATES_REASKED", "GateProblem"]
