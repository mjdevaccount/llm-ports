"""Cached control reads: TTL, last-known on a failed read (deferred one TTL), and the gate's verdicts."""

from __future__ import annotations

import logging
from typing import Any

from llm_ports import call
from llm_ports.market import CATEGORY_CAPPED, CATEGORY_DISABLED, GATE_UNREAD, Gate, GateCache, LastKnown
from llm_ports.testing import StubGenerator
from tests.conftest import Role
from tests.market._events import Events


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class Store:
    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.reads = 0

    async def __call__(self) -> Any:
        self.reads += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


async def test_last_known_reads_once_per_ttl_and_serves_the_last_value_on_a_failure() -> None:
    clock, events = Clock(), Events()
    store = Store(False, RuntimeError("db down"), True)
    flag = LastKnown(store, True, name="work_enabled", ttl_s=10.0, clock=clock, on_event=events)
    assert flag.value is True and not flag.fresh and not flag.known
    assert await flag.get() is False and flag.known
    clock.t += 5
    assert await flag.get() is False and store.reads == 1, "fresh: no read"
    clock.t += 6
    assert await flag.get() is False, "a failed read keeps the last known value"
    assert events.seen == [
        (
            logging.WARNING,
            "market_read_failed",
            {"value": "work_enabled", "reason": "RuntimeError: db down", "serving": "last_known"},
        )
    ]
    clock.t += 5
    assert await flag.get() is False and store.reads == 2, "the failed read is fresh too: no hot retry"
    clock.t += 6
    assert await flag.get() is True and store.reads == 3
    flag.invalidate()
    assert not flag.fresh
    flag.prime(False)
    assert flag.fresh and await flag.get() is False and store.reads == 3


async def test_the_gate_judges_the_cached_rows() -> None:
    clock = Clock()
    rows = {
        "off": Gate(enabled=False),
        "capped": Gate(call_cap=3, used=3),
        "under": Gate(call_cap=3, used=2),
        "unknown_use": Gate(call_cap=3, used=None),
    }
    gates = GateCache(Store(rows), window="daily", clock=clock)
    assert await gates.refresh() == rows
    assert gates.gates == rows
    assert gates.check("off") == f"{CATEGORY_DISABLED}: off is switched off"
    assert gates.check("capped") == f"{CATEGORY_CAPPED}: capped hit its daily call cap (3)"
    assert gates.check("under") is None
    assert gates.check("unknown_use") is None
    assert gates.check("unseeded") is None
    assert gates.check(None) is None and gates.check("") is None
    gates.invalidate()
    gates.prime({"capped": Gate(call_cap=3, used=3)})
    windowless = GateCache(Store(), clock=clock)
    windowless.prime({"c": Gate(call_cap=1, used=1)})
    assert windowless.check("c") == f"{CATEGORY_CAPPED}: c hit its call cap (1)"


async def test_admit_is_a_call_hook_that_gates_before_a_trace_opens() -> None:
    gates = GateCache(Store())
    gates.prime({"demo.off": Gate(enabled=False)})
    role = Role("demo.off", tier="local")
    res = await call.generate(role, prompt="p", generator=StubGenerator(["x"]), hooks=call.Hooks(admit=gates.admit))
    assert (res.outcome, res.reason) == ("gated", "category_disabled: demo.off is switched off")
    ok = Role("demo.on", tier="local")
    res = await call.generate(ok, prompt="p", generator=StubGenerator(["x"]), hooks=call.Hooks(admit=gates.admit))
    assert res.outcome == "ok"


async def test_a_gate_never_read_refuses_and_fail_open_admits_with_an_event() -> None:
    """Spec routing test 10: a never-refreshed gate gates with gate_unread; refreshed and enabled admits."""
    events = Events()
    closed = GateCache(Store(RuntimeError("db down"), {"demo.on": Gate()}), on_event=events, clock=Clock())
    reason = closed.check("demo.on")
    assert reason is not None and reason.startswith(f"{GATE_UNREAD}: ")
    role = Role("demo.on", tier="local")
    res = await call.generate(role, prompt="p", generator=StubGenerator(["x"]), hooks=call.Hooks(admit=closed.admit))
    assert res.outcome == "gated" and (res.reason or "").startswith(GATE_UNREAD)
    await closed.refresh()
    assert closed.check("demo.on") is not None, "a failed first read is still unread"
    closed.invalidate()
    await closed.refresh()
    assert closed.check("demo.on") is None, "refreshed and enabled: admitted"
    res = await call.generate(role, prompt="p", generator=StubGenerator(["x"]), hooks=call.Hooks(admit=closed.admit))
    assert res.outcome == "ok"
    assert closed.check("") is None and closed.check(None) is None

    opened = GateCache(Store(), fail_open=True, on_event=events)
    events.seen.clear()
    assert opened.check("demo.on") is None and opened.check("demo.on") is None
    assert events.seen == [(logging.WARNING, "market_gate_unread", {"category": "demo.on", "serving": "open"})] * 2
