"""`check_generator`: the `Generator` port's contract, driven through a script.

The factory takes a SCRIPT, one reply per model request (`Reply.ok(value, usage=...)` or
`Reply.failed(kind)`), and returns a `Generator` that answers those requests in order: the stub
takes the script as is; a real adapter's test turns it into its wire (a pydantic-ai
`FunctionModel` answering each request from the next reply). Every case builds a fresh generator.

Cases (each is the invariant name a `ContractViolation` carries):
    - `is_generator`: the object satisfies the runtime `Generator` check.
    - `ok_value`: an ok reply is `ok`, with the value, no failure, no problems, `retries == 0`.
    - `answered_requested`: `answered.requested` is the request's model.
    - `failure_never_raises`: a failed reply returns (never raises) with one `Failure` whose kind
      is the reply's and whose outcome is `OUTCOME_FOR[kind]`; the value is None.
    - `reask_counts_retries`: a value `validate` rejects is re-asked; `retries` counts the re-ask.
    - `usage_sums_reasks`: `usage_v2` is the sum over both requests.
    - `last_problems_kept`: problems left after the last re-ask keep the value as `ok` + problems.
    - `callback_internal`: a `validate` that raises is the kind `internal`, never a model outage.

Rationale: ADR-0001, ADR-0016.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from llm_ports.ports import OUTCOME_FOR, FailureKind, Generated, Generator, GenRequest, Usage
from llm_ports.testing._generator import Reply
from llm_ports.testing.conformance._core import call, expect, is_instance, present

#: The model id every case requests.
CHECK_MODEL = "conformance-model"
_GOOD = "good"
_KIND: FailureKind = "bad_request"


def _req(retries: int = 1) -> GenRequest:
    return GenRequest(model=CHECK_MODEL, prompt="conformance", output_retries=retries)


def _rejects_all_but_good(value: str) -> list[str]:
    return [] if value == _GOOD else [f"{value!r} is not {_GOOD!r}"]


async def _run(
    make: Callable[[Sequence[Any]], Generator],
    script: Sequence[Any],
    case: str,
    validate: Callable[[str], Sequence[Any]] | None = None,
) -> Generated[str]:
    gen = make(script)
    is_instance("is_generator", gen, Generator)
    return await call(case, "generate", gen.generate(_req(), str, validate=validate))


async def check_generator(make: Callable[[Sequence[Any]], Generator]) -> tuple[str, ...]:
    """Check a `Generator` against the port's contract; the module docstring lists the cases.

    Args:
        make: Builds a fresh generator that answers its requests from the script, in order.

    Returns:
        The invariants checked, in order.

    Raises:
        ContractViolation: The first invariant the generator breaks.
    """
    res = await _run(make, [Reply.ok("alpha", usage=Usage(input=3, output=2, requests=1))], "ok_value")
    expect(
        "ok_value",
        res.outcome == "ok" and res.value == "alpha" and res.failure is None and not res.problems,
        f"ok reply gave outcome={res.outcome} value={res.value!r} failure={res.failure} problems={res.problems}",
    )
    expect("ok_value", res.retries == 0, f"no re-ask, yet retries={res.retries}")
    who = present("answered_requested", res.answered, "an ok result carries no `answered`")
    expect("answered_requested", who.requested == CHECK_MODEL, f"answered.requested={who.requested!r}")

    res = await _run(make, [Reply.failed(_KIND)], "failure_never_raises")
    got = present("failure_never_raises", res.failure, f"a failed reply gave outcome={res.outcome} and no failure")
    expect(
        "failure_never_raises",
        got.kind == _KIND and res.outcome == OUTCOME_FOR[_KIND] and res.value is None,
        f"failed({_KIND!r}) gave kind={got.kind} outcome={res.outcome} value={res.value!r}",
    )

    script = [
        Reply.ok("draft", usage=Usage(input=3, output=2, requests=1)),
        Reply.ok(_GOOD, usage=Usage(input=4, output=1, requests=1)),
    ]
    res = await _run(make, script, "reask_counts_retries", _rejects_all_but_good)
    expect(
        "reask_counts_retries",
        res.outcome == "ok" and res.value == _GOOD and res.retries == 1 and not res.problems,
        f"after one re-ask: outcome={res.outcome} value={res.value!r} retries={res.retries} problems={res.problems}",
    )
    u = present("usage_sums_reasks", res.usage_v2, "two measured requests gave no usage_v2")
    expect("usage_sums_reasks", (u.input, u.output) == (7, 3), f"usage over both requests: {u}")

    res = await _run(make, [Reply.ok("draft"), Reply.ok("draft")], "last_problems_kept", _rejects_all_but_good)
    expect(
        "last_problems_kept",
        res.outcome == "ok" and res.value == "draft" and bool(res.problems) and res.retries == 1,
        f"problems after the last re-ask: outcome={res.outcome} value={res.value!r} problems={res.problems}",
    )

    def broken(_v: str) -> list[str]:
        raise RuntimeError("validator bug")

    res = await _run(make, [Reply.ok("alpha")], "callback_internal", broken)
    expect(
        "callback_internal",
        res.outcome != "ok" and res.failure is not None and res.failure.kind == "internal",
        f"a raising validator gave outcome={res.outcome} failure={res.failure}",
    )
    return (
        "is_generator",
        "ok_value",
        "answered_requested",
        "failure_never_raises",
        "reask_counts_retries",
        "usage_sums_reasks",
        "last_problems_kept",
        "callback_internal",
    )


__all__ = ["CHECK_MODEL", "check_generator"]
