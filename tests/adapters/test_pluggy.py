"""`adapters.pluggy`: strict loading raises naming the stage; `strict=False` records each pack's end."""

from __future__ import annotations

import importlib.metadata
import logging
from importlib.metadata import EntryPoint
from typing import Any

import pytest

from llm_ports.adapters.pluggy import (
    PACKS_PIN_ENV,
    PROVIDERS_PIN_ENV,
    PackLoad,
    PackLoadError,
    PluginLoadError,
    builtin,
    failed,
    load_packs,
    load_providers,
    packs_loaded,
)
from llm_ports.call.providers import ENTRY_POINT_GROUP as PROVIDERS_GROUP
from llm_ports.call.providers import Loaded
from llm_ports.market import ENTRY_POINT_GROUP, ActionRegistry
from tests.adapters.provider_plugins import kinds
from tests.market._events import Events

PACKS = "tests.adapters.market_packs"
KINDS = "tests.adapters.provider_plugins.kinds"


def _ep(name: str, value: str, group: str = ENTRY_POINT_GROUP) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=group)


class Dist:
    name = "reports-dist"


class VersionedDist:
    name = "reports-dist"
    version = "2.1.0"


class DistEntryPoint:
    """An installed entry point: the same surface `importlib.metadata` hands back, with a dist."""

    name = "reports"
    dist: object = Dist()

    def load(self) -> Any:
        return _ep("reports", f"{PACKS}.good").load()


class VersionedEntryPoint(DistEntryPoint):
    dist = VersionedDist()


def test_a_good_pack_joins_and_names_its_distribution() -> None:
    reg, events = ActionRegistry("main"), Events()
    loads = load_packs(reg, entry_points=[DistEntryPoint()], on_event=events)  # type: ignore[list-item]  # a duck-typed entry point
    assert loads == (
        PackLoad("reports", "reports-dist", "ok", actions=("report.weekly", "report.digest"), origin="reports-dist"),
    )
    assert [s.name for s in reg.registered()] == ["report.weekly", "report.digest"]
    assert reg[("report", "weekly")].handler_path == f"{PACKS}.good.weekly"
    assert reg.origin("report", "weekly") == "reports"
    assert events.seen == [
        (
            logging.INFO,
            "market_pack_loaded",
            {"pack": "reports", "dist": "reports-dist", "actions": ["report.weekly", "report.digest"]},
        ),
        (
            logging.INFO,
            "plugins_loaded",
            {"group": ENTRY_POINT_GROUP, "names": ["reports"], "origin": {"reports": "reports-dist"}},
        ),
    ]
    assert failed(loads) == ()
    assert packs_loaded(loads) == Loaded(ENTRY_POINT_GROUP, ("reports",), {"reports": "reports-dist"})


def test_the_origin_carries_the_distribution_version_when_it_has_one() -> None:
    loads = load_packs(ActionRegistry("main"), entry_points=[VersionedEntryPoint()], on_event=Events())  # type: ignore[list-item]  # a duck-typed entry point
    assert loads[0].origin == "reports-dist==2.1.0"


def test_strict_raises_the_first_failure_naming_group_plugin_and_stage() -> None:
    reg, events = ActionRegistry("main"), Events()
    eps = [_ep("reports", f"{PACKS}.good"), _ep("raises", f"{PACKS}.broken:raises")]
    with pytest.raises(PackLoadError) as caught:
        load_packs(reg, entry_points=eps, on_event=events)
    err = caught.value
    assert (err.group, err.plugin, err.stage) == (ENTRY_POINT_GROUP, "raises", "hook")
    assert isinstance(err, PluginLoadError) and isinstance(err, ImportError)
    assert isinstance(err.__cause__, RuntimeError)
    assert str(err).startswith(f"{ENTRY_POINT_GROUP}: plugin 'raises' failed at hook: RuntimeError")
    assert events.seen[-1][1] == "market_pack_failed", "the failure is emitted before it raises"


def test_not_strict_names_every_stage_and_the_rest_still_load() -> None:
    reg, events = ActionRegistry("main"), Events()
    eps = [
        _ep("missing", "tests.adapters.market_packs.nope"),
        _ep("no_hook", f"{PACKS}.broken:no_hook"),
        _ep("extra_argument", f"{PACKS}.broken:extra_argument"),
        _ep("raises", f"{PACKS}.broken:raises"),
        _ep("not_a_registry", f"{PACKS}.broken:not_a_registry"),
        _ep("reports", f"{PACKS}.good"),
        _ep("zz_conflicts", f"{PACKS}.broken:conflicts"),
        _ep("zz_again", f"{PACKS}.good"),
        _ep("misspelled", f"{PACKS}.broken:misspelled"),
    ]
    loads = load_packs(reg, entry_points=eps, on_event=events, strict=False)
    by_name = {load.name: load for load in loads}
    stages = {load.name: (load.outcome, load.stage) for load in loads}
    assert stages == {
        "extra_argument": ("failed", "validate"),
        "misspelled": ("failed", "validate"),
        "missing": ("failed", "import"),
        "no_hook": ("failed", "validate"),
        "not_a_registry": ("failed", "pack"),
        "raises": ("failed", "pack"),
        "reports": ("ok", None),
        "zz_again": ("failed", "validate"),
        "zz_conflicts": ("failed", "join"),
    }
    assert [load.name for load in loads] == sorted(by_name), "entry points load sorted by name"
    assert by_name["missing"].reason is not None and by_name["missing"].reason.startswith("ModuleNotFoundError")
    assert by_name["no_hook"].reason == "implements no market_actions hook"
    assert "registry" in (by_name["extra_argument"].reason or "")
    assert "market_action" in (by_name["misspelled"].reason or "")
    assert by_name["raises"].reason == "RuntimeError: pack build failed"
    assert by_name["not_a_registry"].reason == "market_actions returned dict, not an ActionRegistry"
    conflict = by_name["zz_conflicts"].reason or ""
    assert "report.weekly" in conflict and "'reports'" in conflict and "'zz_conflicts'" in conflict
    assert len(reg) == 2, "only the good pack joined; the conflicting one added nothing"
    assert [load.name for load in failed(loads)] == [n for n, (o, _) in stages.items() if o == "failed"]
    warned = [e for e in events.seen if e[1] == "market_pack_failed"]
    assert len(warned) == 8 and all(level == logging.WARNING for level, _, _ in warned)
    assert {w[2]["pack"]: w[2]["stage"] for w in warned} == {n: st for n, (o, st) in stages.items() if o == "failed"}, (
        "the event's stage is the PackLoad's stage"
    )
    assert {w[2]["plugin_stage"] for w in warned} == {"import", "register", "pending", "hook", "duplicate"}
    assert not reg.frozen, "a diagnostic load leaves the registry open"
    assert events.seen[-1][2]["names"] == ["reports"]


def test_a_pack_refused_by_the_registry_is_stage_register() -> None:
    reg = ActionRegistry("main", clocks=("trading",))  # the good pack's actions use the wall clock
    loads = load_packs(reg, entry_points=[_ep("reports", f"{PACKS}.good")], on_event=Events(), strict=False)
    assert (loads[0].outcome, loads[0].stage) == ("failed", "join")
    with pytest.raises(PackLoadError) as caught:
        load_packs(reg, entry_points=[_ep("reports", f"{PACKS}.good")], on_event=Events())
    assert caught.value.stage == "register"


def test_selection_problems_are_recorded_when_not_strict() -> None:
    builtins = [("core", object()), ("core", object())]
    eps = [_ep("core", f"{PACKS}.good"), _ep("a", f"{PACKS}.good"), _ep("a", f"{PACKS}.other")]
    loads = load_packs(
        ActionRegistry("main"), entry_points=eps, builtins=builtins, pin="a,ghost", strict=False, on_event=Events()
    )
    problems = [(load.name, load.stage) for load in loads if load.outcome == "failed"]
    assert ("core", "join") in problems and ("a", "join") in problems and ("ghost", "pin") in problems


def test_a_blocked_pack_is_skipped_unloaded() -> None:
    reg, events = ActionRegistry("main"), Events()
    loads = load_packs(reg, block=["reports"], entry_points=[_ep("reports", f"{PACKS}.good")], on_event=events)
    assert loads == (PackLoad("reports", None, "blocked", origin="unknown"),)
    assert len(reg) == 0 and failed(loads) == ()
    assert events.seen[0] == (logging.INFO, "market_pack_blocked", {"pack": "reports", "dist": None})
    assert events.seen[1][2]["names"] == []


def test_the_installed_group_is_read_when_no_list_is_given(caplog: Any) -> None:
    reg = ActionRegistry("main")
    assert load_packs(reg, group="llm_ports.market.tests_no_such_group") == ()
    assert reg.frozen, "a strict load that finishes freezes the registry"
    with pytest.raises(RuntimeError, match="frozen"):
        load_packs(reg, entry_points=[_ep("reports", f"{PACKS}.good")])
    with caplog.at_level(logging.INFO, logger="llm_ports.market"):
        load_packs(ActionRegistry("main"), entry_points=[_ep("reports", f"{PACKS}.good")])
    assert "market_pack_loaded" in caplog.text and "plugins_loaded" in caplog.text


def test_providers_load_frozen_with_their_origins_and_an_event(caplog: Any) -> None:
    events = Events()
    eps = [_ep("b", f"{KINDS}:beta", PROVIDERS_GROUP), _ep("a", f"{KINDS}:alpha", PROVIDERS_GROUP)]
    store = load_providers(builtins=[builtin("core", kinds.Kind("core"))], entry_points=eps, on_event=events)
    assert list(store) == ["core", "alpha", "beta"] and store.frozen
    assert store.loaded == Loaded(
        PROVIDERS_GROUP, ("core", "a", "b"), {"core": "builtin", "a": "unknown", "b": "unknown"}
    )
    assert store.plugin_of("alpha") == "a"
    assert events.seen == [
        (
            logging.INFO,
            "plugins_loaded",
            {"group": PROVIDERS_GROUP, "names": ["core", "a", "b"], "origin": dict(store.loaded.origin)},
        )
    ]
    with caplog.at_level(logging.INFO, logger="llm_ports.call"):
        assert len(load_providers(entry_points=lambda: (), pin="")) == 0
    assert "plugins_loaded" in caplog.text


def test_load_providers_reads_the_installed_group_when_no_list_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def installed(*, group: str) -> list[EntryPoint]:
        asked.append(group)
        return [_ep("a", f"{KINDS}:alpha", PROVIDERS_GROUP)]

    monkeypatch.setattr(importlib.metadata, "entry_points", installed)
    store = load_providers(on_event=Events())
    assert asked == [PROVIDERS_GROUP] and list(store) == ["alpha"]
    assert store.loaded.plugins == ("a",)


def test_a_provider_plugin_that_registers_a_non_factory_or_no_hook_fails_loudly() -> None:
    with pytest.raises(PluginLoadError) as caught:
        load_providers(entry_points=[_ep("odd", f"{KINDS}:not_a_factory", PROVIDERS_GROUP)], on_event=Events())
    assert (caught.value.plugin, caught.value.stage) == ("odd", "register")
    with pytest.raises(PluginLoadError) as caught:
        load_providers(entry_points=[_ep("none", f"{KINDS}:no_hook", PROVIDERS_GROUP)], on_event=Events())
    assert (caught.value.plugin, caught.value.stage) == ("none", "pending")


def test_the_pin_setting_names_are_constants_the_library_never_reads() -> None:
    assert (PACKS_PIN_ENV, PROVIDERS_PIN_ENV) == ("LLM_PORTS_PLUGINS", "LLM_PORTS_PROVIDERS")
