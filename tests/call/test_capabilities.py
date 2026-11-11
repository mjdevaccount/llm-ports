"""The capability pre-check: a request its model cannot serve is refused before the wire (spec test 3).

Also: every result carries the table's fingerprint (`stamp["tiers"]`), and the schema 1 and
model-override warnings fire once.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from llm_ports import call
from llm_ports.call import _generate, roles
from llm_ports.ports import Generated
from llm_ports.testing import StubGenerator
from tests.call.test_table_v2 import V2
from tests.conftest import TABLE, Role

PNG = (b"\x89PNG", "image/png")


class V2Role(roles.Role):
    TIERS: ClassVar = V2.tiers


class Answer(BaseModel):
    text: str


Events = list[tuple[int, str, dict[str, Any]]]


def _hooks() -> tuple[call.Hooks, Events]:
    events: Events = []
    return call.Hooks(on_event=lambda lvl, ev, f: events.append((lvl, ev, dict(f)))), events


async def test_3_images_to_a_model_without_vision_are_refused_with_zero_wire_requests() -> None:
    gen, (hooks, events) = StubGenerator("seen"), _hooks()
    res = await call.generate(
        V2Role("t.web", tier="local-web"), prompt="read it", generator=gen, hooks=hooks, images=[PNG]
    )
    assert res.outcome == "request_fault" and res.value is None
    assert res.reason == "unsupported: model 'web' (small-web) does not state vision"
    assert gen.requests == [], "nothing reached the wire"
    assert res.stamp["tiers"] == V2.fingerprint()
    assert ("llm_call_failed", "request_fault") in [(e, f.get("outcome")) for _, e, f in events]


async def test_3_images_to_a_model_with_vision_send_one_request() -> None:
    gen, (hooks, _events) = StubGenerator("seen"), _hooks()
    res = await call.generate(
        V2Role("t.chat", tier="local"), prompt="read it", generator=gen, hooks=hooks, images=[PNG]
    )
    assert res.outcome == "ok" and len(gen.requests) == 1
    assert res.stamp["tiers"] == V2.fingerprint()


@pytest.mark.parametrize(
    ("tier", "web", "output", "refused"),
    [
        ("local-web", True, Answer, False),  # json_object stands in for json_schema
        ("local", True, str, True),  # the chat model does not state web
        ("cloud", False, Answer, False),
    ],
)
async def test_web_and_structured_output_are_checked_too(
    tier: str, web: bool, output: type[Any], refused: bool
) -> None:
    gen = StubGenerator(Answer(text="x"))
    res = await call.generate(V2Role("t.x", tier=tier), prompt="p", output=output, generator=gen, web=web)
    assert (res.outcome == "request_fault") is refused
    assert len(gen.requests) == (0 if refused else 1)


async def test_a_refused_admit_is_stamped_too() -> None:
    hooks = call.Hooks(admit=lambda _role: "paused", on_event=lambda *_: None)
    res = await call.generate(V2Role("t.chat", tier="local"), prompt="p", generator=StubGenerator("x"), hooks=hooks)
    assert res.outcome == "gated" and res.stamp["tiers"] == V2.fingerprint()


async def test_a_schema_1_table_is_unchecked_and_says_so_once_per_process() -> None:
    _generate._WARNED.clear()
    gen, (hooks, events) = StubGenerator("seen"), _hooks()
    role = Role("t.v1", tier="local-web")
    for _ in range(2):
        res = await call.generate(role, prompt="read it", generator=gen, hooks=hooks, images=[PNG])
        assert res.outcome == "ok" and res.stamp["tiers"] == TABLE.fingerprint()
    warned = [(lvl, f) for lvl, e, f in events if e == "llm_call_warning"]
    assert len(warned) == 1 and len(gen.requests) == 2
    assert warned[0][0] == logging.WARNING and warned[0][1]["setting"] == "capabilities"


async def test_a_role_that_overrides_the_model_id_is_named_once() -> None:
    gen, (hooks, events) = StubGenerator("seen"), _hooks()
    role = V2Role("t.override", tier="local", model="other-chat")
    for _ in range(2):
        await call.generate(role, prompt="p", generator=gen, hooks=hooks)
    warned = [f for _, e, f in events if e == "llm_call_warning"]
    assert [w["setting"] for w in warned] == ["model"] and "t.override" in warned[0]["detail"]
    assert gen.requests[0].model == "other-chat"


async def test_a_hand_built_tier_has_no_fingerprint_to_stamp() -> None:
    from llm_ports.call.roles import Tier

    class HandRole(roles.Role):
        TIERS: ClassVar = {"t": Tier("t", "m", 1.0)}

    res = await call.generate(
        HandRole("t.hand", tier="t"), prompt="p", generator=StubGenerator(Generated(value="x", outcome="ok"))
    )
    assert "tiers" not in res.stamp
