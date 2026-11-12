"""The facade's deadline and fallback loop: spec routing tests 5-6, resilience tests 4-6 at the facade.

A role's tier answers or fails; a failure whose kind is in the tier's `fallback_on` moves the call
to the next tier of its `fallback` chain with what is left of ONE deadline. The generators here
are fakes keyed on the model id, so each test counts exactly which tier was asked what.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from llm_ports.call import Hooks, generate, roles
from llm_ports.call._generate import DEADLINE_GRACE_S
from llm_ports.ports import Answered, Attempt, Failure, Generated, Generator, GenRequest, WirePolicy, failure
from tests.call.test_table_v2 import V2, _with

LOCAL, CLOUD = "small-chat", "gpt-5-nano"


class Tiers:
    """A generator per model id: each answers from its script, after `delay` seconds."""

    def __init__(self, script: Mapping[str, Failure | str], delay: Mapping[str, float] | None = None) -> None:
        self.script = script
        self.delay = delay or {}
        self.requests: list[GenRequest] = []
        self.started: dict[str, float] = {}

    async def generate(self, req: GenRequest, output: type[Any], **_kw: Any) -> Generated[Any]:
        self.requests.append(req)
        self.started[req.model] = asyncio.get_running_loop().time()
        await asyncio.sleep(self.delay.get(req.model, 0.0))
        got = self.script[req.model]
        if isinstance(got, Failure):
            retried = Attempt(1, None, req.model, failure("rate_limited"), 0.5)
            return Generated(value=None, outcome=got.outcome, model=req.model, failure=got, attempts=(retried,))
        return Generated(value=got, outcome="ok", model=req.model, answered=Answered(requested=req.model))


def _role(table: roles.TierTable = V2, **kw: Any) -> roles.Role:
    class R(roles.Role):
        TIERS: ClassVar = table.tiers

    return R("t.fallback", tier="local", **kw)


def _events() -> tuple[list[tuple[str, dict[str, Any]]], Hooks]:
    seen: list[tuple[str, dict[str, Any]]] = []
    return seen, Hooks(on_event=lambda _lvl, name, fields: seen.append((name, dict(fields))))


# -- routing 5: fallback after the tier's own wire policy, only on a listed kind -----------------


async def test_5_an_overloaded_tier_falls_back_and_the_result_names_who_answered() -> None:
    gen = Tiers({LOCAL: failure("overloaded"), CLOUD: "from the cloud"})
    seen, hooks = _events()
    res = await generate(_role(), prompt="p", generator=gen, hooks=hooks)
    assert res.outcome == "ok" and res.value == "from the cloud"
    assert res.answered is not None and res.answered.tier == "cloud" and res.stamp["fallback_depth"] == 1
    assert [r.model for r in gen.requests] == [LOCAL, CLOUD]
    tiers = [(a.tier, a.failure.kind) for a in res.attempts]
    assert tiers == [("local", "rate_limited"), ("local", "overloaded")], "the tier's wire retry, then the tier"
    assert [name for name, _ in seen if name == "llm_fallback"] == ["llm_fallback"]
    assert dict(seen)["llm_fallback"] == {
        "role": "t.fallback",
        "from": "local",
        "to": "cloud",
        "kind": "overloaded",
        "depth": 1,
    }


async def test_5_quota_does_not_fall_back_and_listing_it_makes_it_fall_back() -> None:
    gen = Tiers({LOCAL: failure("quota"), CLOUD: "from the cloud"})
    res = await generate(_role(), prompt="p", generator=gen)
    assert res.failure is not None and res.failure.kind == "quota" and len(gen.requests) == 1
    assert res.answered is not None and res.answered.tier == "local" and res.stamp["fallback_depth"] == 0
    listed = roles.load_tiers(_with(lambda r: r["tiers"]["local"].update(fallback_on=["overloaded", "quota"])))
    gen = Tiers({LOCAL: failure("quota"), CLOUD: "from the cloud"})
    res = await generate(_role(listed), prompt="p", generator=gen)
    assert res.outcome == "ok" and res.answered is not None and res.answered.tier == "cloud"


async def test_5_the_last_tier_s_failure_is_the_call_s_with_every_earlier_one_in_attempts() -> None:
    gen = Tiers({LOCAL: failure("overloaded"), CLOUD: failure("overloaded")})
    res = await generate(_role(), prompt="p", generator=gen)
    assert res.failure is not None and res.failure.kind == "overloaded" and res.answered is not None
    assert res.answered.tier == "cloud" and [a.tier for a in res.attempts] == ["local", "local", "cloud"]


async def test_5_each_tier_gets_its_own_model_policy_and_generator() -> None:
    local = Tiers({"pinned-override": failure("overloaded")})
    cloud = Tiers({CLOUD: "hosted"})
    picked: dict[str, Generator] = {"local": local, "cloud": cloud}
    res = await generate(
        _role(model="pinned-override"), prompt="p", generator=local, hooks=Hooks(generator_for=picked.__getitem__)
    )
    assert res.outcome == "ok" and [r.model for r in local.requests] == ["pinned-override"]
    assert [r.model for r in cloud.requests] == [CLOUD], "a fallback tier drops the role's model override"
    assert local.requests[0].policy == V2.tiers["local"].wire_policy and cloud.requests[0].policy is None
    assert local.requests[0].concurrency_group == "gpu" and cloud.requests[0].concurrency_group == "cloud"


async def test_5_a_role_policy_beats_the_tier_s() -> None:
    gen = Tiers({LOCAL: "ok"})
    await generate(_role(policy=WirePolicy.none()), prompt="p", generator=gen)
    assert gen.requests[0].policy == WirePolicy.none()


# -- routing 6 / resilience 5: one deadline over every tier --------------------------------------


async def test_6_the_next_tier_runs_on_what_is_left_of_the_deadline() -> None:
    gen = Tiers({LOCAL: failure("overloaded"), CLOUD: "late but in time"}, delay={LOCAL: 0.2})
    res = await generate(_role(timeout_s=2.0), prompt="p", generator=gen)
    assert res.outcome == "ok"
    first, second = gen.requests
    assert first.deadline is second.deadline and first.deadline is not None and first.deadline.total_s == 2.0
    assert second.deadline.remaining(gen.started[CLOUD]) <= 2.0 - 0.2


async def test_6_a_tier_that_overruns_the_deadline_is_cut_off_as_timeout_total() -> None:
    gen = Tiers({LOCAL: failure("overloaded"), CLOUD: "never"}, delay={LOCAL: 0.05, CLOUD: 30.0})
    seen, hooks = _events()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    res = await generate(_role(timeout_s=0.3), prompt="p", generator=gen, hooks=hooks)
    assert loop.time() - t0 < 0.3 + DEADLINE_GRACE_S + 0.5
    assert res.failure is not None and (res.failure.kind, res.failure.phase) == ("timeout", "total")
    assert res.answered is not None and res.answered.tier == "cloud" and res.stamp["fallback_depth"] == 1
    assert [a.tier for a in res.attempts] == ["local", "local"]
    names = [name for name, _ in seen]
    assert "llm_deadline_expired" in names and "llm_call_failed" in names


async def test_6_no_tier_starts_once_the_deadline_is_spent() -> None:
    gen = Tiers({LOCAL: failure("overloaded"), CLOUD: "never"}, delay={LOCAL: 0.12})
    res = await generate(_role(timeout_s=0.08), prompt="p", generator=gen)
    assert [r.model for r in gen.requests] == [LOCAL], "the fallback tier never started"
    assert res.failure is not None and (res.failure.kind, res.failure.phase) == ("timeout", "total")


# -- resilience 6 at the facade: a cancel is not a timeout ---------------------------------------


async def test_cancelling_the_call_propagates_and_is_never_a_result() -> None:
    gen = Tiers({LOCAL: "never"}, delay={LOCAL: 30.0})
    seen, hooks = _events()
    task = asyncio.create_task(generate(_role(), prompt="p", generator=gen, hooks=hooks))
    while not gen.requests:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not [name for name, _ in seen if name in ("llm_deadline_expired", "llm_call_failed")]
