"""The action registry: the declare-time refusals (both directions), the duplicate guard, include."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pytest

from llm_ports.market import SILENCE_CLOCKS, ActionRegistry, ActionSpec, DuplicateAction, parse_interval, refusal
from tests.market._events import CACHE, Events


async def _handler(_session: object, key: str) -> dict[str, Any]:
    return {"key": key}


async def _stale(_session: object, ttl_s: float) -> list[str]:
    return ["a", "b"]


def test_a_spec_with_no_freshness_rule_is_refused_at_decoration_and_not_registered() -> None:
    events = Events()
    reg = ActionRegistry("t", on_event=events)
    with pytest.raises(ValueError, match=r"zz\.probe.*max_silence_h"):
        reg.action(feature="zz", action="probe", cache_table="out.zz")(_handler)
    assert reg.find("zz", "probe") is None
    assert events.seen[0][:2] == (logging.WARNING, "market_action_refused")
    assert events.seen[0][2]["action"] == "zz.probe"


@pytest.mark.parametrize("kw", [{"desired_ttl": "6h", **CACHE}, {"cadence": "24h"}, {"max_silence_h": 48}])
def test_each_freshness_rule_alone_is_admitted(kw: dict[str, Any]) -> None:
    reg = ActionRegistry()
    reg.action(feature="zz", action="probe", **kw)(_handler)
    spec = reg.find("zz", "probe")
    assert spec is not None and spec.fn is _handler
    assert spec.handler_path == f"{__name__}._handler"


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"silence_clock": "rth"}, "silence_clock"),
        ({"desired_ttl": "18 hours"}, "desired_ttl: bad interval"),
        ({"cadence": "1w"}, "cadence: bad interval"),
        ({"max_silence_h": 0}, "max_silence_h must be > 0"),
        ({"lane": -1}, "lane must be"),
        ({"lane": True}, "lane must be"),
        ({"value_weight": 0.0}, "value_weight must be > 0"),
        ({"est_cost_s": 0}, "est_cost_s must be > 0"),
        ({"feature": ""}, "non-empty"),
    ],
)
def test_a_malformed_field_is_refused(kw: dict[str, Any], match: str) -> None:
    args: dict[str, Any] = {"feature": "zz", "action": "probe", "desired_ttl": "6h", **CACHE, **kw}
    reg = ActionRegistry()
    with pytest.raises(ValueError, match=match):
        reg.action(**args)(_handler)
    assert len(reg) == 0


def test_a_registry_can_accept_its_own_clocks() -> None:
    reg = ActionRegistry(clocks=(*SILENCE_CLOCKS, "rth"))
    reg.action(feature="zz", action="probe", desired_ttl="6h", **CACHE, silence_clock="rth")(_handler)
    assert reg.clocks == ("wall", "session", "rth")
    assert reg.find("zz", "probe") is not None


def test_spec_values_parse_and_name_themselves() -> None:
    reg = ActionRegistry()
    reg.action(feature="f", action="a", desired_ttl="90m", **CACHE, cadence="2d", source="src", stale_fn=_stale)(_handler)
    spec = reg[("f", "a")]
    assert (spec.key, spec.name, spec.source) == (("f", "a"), "f.a", ("src",))
    assert spec.ttl == timedelta(minutes=90)
    assert spec.cadence_interval == timedelta(days=2)
    assert spec.limit_h == 1.5
    only_silence = ActionSpec(feature="f", action="b", handler_path="m.h", fn=_handler, max_silence_h=12)
    assert (only_silence.ttl, only_silence.cadence_interval, only_silence.limit_h) == (None, None, None)
    assert refusal(only_silence) is None
    assert parse_interval(" 45s ") == timedelta(seconds=45)
    reg.action(feature="f", action="c", cadence="7d", source=["x", "y"])(_handler)
    assert reg[("f", "c")].limit_h == 168.0 and reg[("f", "c")].source == ("x", "y")


def test_a_redeclaration_raises_even_with_equal_metadata_and_the_identical_object_is_a_no_op() -> None:
    events = Events()
    reg = ActionRegistry("t", on_event=events)
    reg.action(feature="f", action="a", desired_ttl="6h", **CACHE)(_handler)

    async def other(_session: object, key: str) -> None: ...

    other.__qualname__ = _handler.__qualname__
    with pytest.raises(DuplicateAction, match=r"f\.a declared by both '[^']+_handler' and '[^']+_handler'"):
        reg.action(feature="f", action="a", desired_ttl="6h", **CACHE)(other)
    assert reg[("f", "a")].fn is _handler, "a different fn never swaps in"
    assert events.seen[-1][1] == "market_action_refused"
    with pytest.raises(DuplicateAction):
        reg.action(feature="f", action="a", desired_ttl="12h", **CACHE)(_handler)
    held = reg[("f", "a")]
    assert reg.add(held) is held and len(reg) == 1, "the identical object re-adds as a no-op"
    assert reg.origin("f", "a") == held.handler_path and reg.origin("f", "z") is None


def test_a_ttl_needs_its_output_columns_and_a_source_must_exist() -> None:
    reg = ActionRegistry("t", sources=("chain", "news"))
    with pytest.raises(ValueError, match="desired_ttl needs cache_table and age_col"):
        reg.action(feature="f", action="a", desired_ttl="6h", cache_table="out.items")(_handler)
    with pytest.raises(ValueError, match=r"source \['tape'\] not among the known sources"):
        reg.action(feature="f", action="a", cadence="1h", source=("chain", "tape"))(_handler)
    reg.action(feature="f", action="a", desired_ttl="6h", **CACHE, source=("chain", "news"))(_handler)
    assert len(reg) == 1 and reg.empty().find("f", "a") is None
    spec = reg[("f", "a")]
    assert refusal(spec, SILENCE_CLOCKS, ("chain",)) is not None and refusal(spec, SILENCE_CLOCKS, None) is None


def test_a_frozen_registry_refuses_every_write_and_keeps_raw_teardown() -> None:
    reg = ActionRegistry("t")
    reg.action(feature="f", action="a", cadence="1h")(_handler)
    assert not reg.frozen
    reg.freeze()
    assert reg.frozen
    with pytest.raises(RuntimeError, match="frozen"):
        reg.action(feature="f", action="b", cadence="1h")(_handler)
    pack = ActionRegistry("pack")
    pack.action(feature="g", action="x", cadence="1h")(_handler)
    with pytest.raises(RuntimeError, match="frozen"):
        reg.include(pack)
    with pytest.raises(RuntimeError, match="frozen"):
        reg[("g", "x")] = pack[("g", "x")]
    del reg[("f", "a")]
    assert len(reg) == 0 and not reg.empty().frozen


def test_the_mapping_surface_keeps_the_guard_on_writes_and_stays_raw_on_deletes() -> None:
    reg = ActionRegistry("t")
    reg.action(feature="f", action="a", desired_ttl="6h", **CACHE)(_handler)
    reg.action(feature="f", action="b", desired_ttl="6h", **CACHE)(_handler)
    assert list(reg) == [("f", "a"), ("f", "b")]
    assert [s.name for s in reg.registered()] == ["f.a", "f.b"]
    spec = reg.pop(("f", "a"))
    assert reg.get(("f", "a")) is None
    reg[spec.key] = spec
    with pytest.raises(ValueError, match="is not the spec's own key"):
        reg[("x", "y")] = spec
    del reg[("f", "b")]
    assert repr(reg) == "ActionRegistry('t', n=1)"
    fresh = reg.empty()
    assert (fresh.name, len(fresh), fresh.clocks) == ("t", 0, reg.clocks)


def test_include_is_all_or_nothing() -> None:
    events = Events()
    main = ActionRegistry("main", on_event=events)
    main.action(feature="f", action="a", desired_ttl="6h", **CACHE)(_handler)
    pack = ActionRegistry("pack")
    pack.action(feature="g", action="x", desired_ttl="6h", **CACHE)(_handler)
    pack.action(feature="f", action="a", desired_ttl="12h", **CACHE)(_handler)
    with pytest.raises(DuplicateAction, match=r"pack 'pack' cannot join 'main': market action f\.a"):
        main.include(pack)
    assert main.find("g", "x") is None, "no action of a refused pack joins"
    assert events.seen[-1][1] == "market_action_refused"

    clocked = ActionRegistry("clocked", clocks=("wall", "rth"))
    clocked.action(feature="h", action="y", desired_ttl="6h", **CACHE, silence_clock="rth")(_handler)
    with pytest.raises(ValueError, match="silence_clock 'rth'"):
        main.include(clocked)

    equal = ActionRegistry("equal")
    equal.action(feature="f", action="a", desired_ttl="6h", **CACHE)(_handler)  # equal to main's, not the same
    with pytest.raises(DuplicateAction, match=r"cannot join 'main': market action f\.a declared by both"):
        main.include(equal, origin="equal-pack")
    ok = ActionRegistry("ok")
    ok.action(feature="g", action="x", desired_ttl="6h", **CACHE)(_handler)
    ok[("f", "a")] = main[("f", "a")]  # the identical object: no conflict
    assert [s.name for s in main.include(ok)] == ["g.x", "f.a"]
    assert len(main) == 2 and main.origin("g", "x") == "ok"


def test_the_default_event_sink_is_the_stdlib_logger(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # import-linter's in-process run (tests/contract) disables loggers that already exist.
    monkeypatch.setattr(logging.getLogger("llm_ports.market"), "disabled", False)
    reg = ActionRegistry()
    with caplog.at_level(logging.WARNING, logger="llm_ports.market"), pytest.raises(ValueError):
        reg.action(feature="f", action="a")(_handler)
    assert "market_action_refused" in caplog.text
