"""`adapters.pluggy.load_packs`: each entry point ends ok, failed (with its stage) or blocked; never raises."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint
from typing import Any

from llm_ports.adapters.pluggy import PackLoad, failed, load_packs
from llm_ports.market import ENTRY_POINT_GROUP, ActionRegistry
from tests.market._events import Events

PACKS = "tests.adapters.market_packs"


def _ep(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)


class Dist:
    name = "reports-dist"


class DistEntryPoint:
    """An installed entry point: the same surface `importlib.metadata` hands back, with a dist."""

    name = "reports"
    dist = Dist()

    def load(self) -> Any:
        return _ep("reports", f"{PACKS}.good").load()


def test_a_good_pack_joins_and_names_its_distribution() -> None:
    reg, events = ActionRegistry("main"), Events()
    loads = load_packs(reg, entry_points=[DistEntryPoint()], on_event=events)  # type: ignore[list-item]  # a duck-typed entry point
    assert loads == (PackLoad("reports", "reports-dist", "ok", actions=("report.weekly", "report.digest")),)
    assert [s.name for s in reg.registered()] == ["report.weekly", "report.digest"]
    assert reg[("report", "weekly")].handler_path == f"{PACKS}.good.weekly"
    assert events.seen == [
        (
            logging.INFO,
            "market_pack_loaded",
            {"pack": "reports", "dist": "reports-dist", "actions": ["report.weekly", "report.digest"]},
        )
    ]
    assert failed(loads) == ()


def test_every_failure_stage_is_named_and_the_rest_still_load() -> None:
    reg, events = ActionRegistry("main"), Events()
    eps = [
        _ep("missing", "tests.adapters.market_packs.nope"),
        _ep("no_hook", f"{PACKS}.broken:no_hook"),
        _ep("extra_argument", f"{PACKS}.broken:extra_argument"),
        _ep("raises", f"{PACKS}.broken:raises"),
        _ep("not_a_registry", f"{PACKS}.broken:not_a_registry"),
        _ep("reports", f"{PACKS}.good"),
        _ep("conflicts", f"{PACKS}.broken:conflicts"),
        _ep("again", f"{PACKS}.good"),
    ]
    loads = load_packs(reg, entry_points=eps, on_event=events)
    by_name = {load.name: load for load in loads}
    stages = {load.name: (load.outcome, load.stage) for load in loads}
    assert stages == {
        "missing": ("failed", "import"),
        "no_hook": ("failed", "validate"),
        "extra_argument": ("failed", "validate"),
        "raises": ("failed", "pack"),
        "not_a_registry": ("failed", "pack"),
        "reports": ("ok", None),
        "conflicts": ("failed", "join"),
        "again": ("failed", "validate"),
    }
    assert by_name["missing"].reason is not None and by_name["missing"].reason.startswith("ModuleNotFoundError")
    assert by_name["no_hook"].reason == "implements no market_actions hook"
    assert "registry" in (by_name["extra_argument"].reason or "")
    assert by_name["raises"].reason == "RuntimeError: pack build failed"
    assert by_name["not_a_registry"].reason == "market_actions returned dict, not an ActionRegistry"
    assert "report.weekly" in (by_name["conflicts"].reason or "")
    assert len(reg) == 2, "only the good pack joined; the conflicting one added nothing"
    assert [load.name for load in failed(loads)] == [n for n, (o, _) in stages.items() if o == "failed"]
    warned = [e for e in events.seen if e[1] == "market_pack_failed"]
    assert len(warned) == 7 and all(level == logging.WARNING for level, _, _ in warned)
    assert warned[0][2]["stage"] == "import"


def test_a_blocked_pack_is_skipped_unloaded() -> None:
    reg, events = ActionRegistry("main"), Events()
    loads = load_packs(reg, block=["reports"], entry_points=[_ep("reports", f"{PACKS}.good")], on_event=events)
    assert loads == (PackLoad("reports", None, "blocked"),)
    assert len(reg) == 0 and failed(loads) == ()
    assert events.seen == [(logging.INFO, "market_pack_blocked", {"pack": "reports", "dist": None})]


def test_the_installed_group_is_read_when_no_list_is_given(caplog: Any) -> None:
    reg = ActionRegistry("main")
    assert load_packs(reg, group="llm_ports.market.tests_no_such_group") == ()
    with caplog.at_level(logging.INFO, logger="llm_ports.market"):
        load_packs(reg, entry_points=[_ep("reports", f"{PACKS}.good")])
    assert "market_pack_loaded" in caplog.text
