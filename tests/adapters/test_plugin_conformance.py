"""The plugin-registry contract (R1-R10), the same ten tests against both entry-point groups.

`llm_ports.market` (action packs, `load_packs`) and `llm_ports.providers` (provider kinds,
`load_providers`) share one loader; each test runs once per group.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint

import pluggy
import pytest

from llm_ports.adapters.pluggy import (
    PackLoadError,
    PluginLoadError,
    builtin,
    hookimpl,
    load_packs,
    load_providers,
    packs_loaded,
)
from llm_ports.call.providers import Loaded
from llm_ports.market import ENTRY_POINT_GROUP, ActionRegistry, ActionSpec
from tests.adapters.provider_plugins import kinds
from tests.market._events import Events

PACKS = "tests.adapters.market_packs"
KINDS = "tests.adapters.provider_plugins.kinds"

EPs = Iterable[EntryPoint] | Callable[[], Iterable[EntryPoint]]


class _BuiltinPack:
    """A pack that ships with the caller: registered by name, no distribution metadata."""

    @hookimpl
    def market_actions(self) -> ActionRegistry:
        pack = ActionRegistry("builtin")

        async def tick(_session: object, work_key: str) -> None: ...

        pack.action(feature="core", action="tick", max_silence_h=1)(tick)
        return pack


@dataclass(frozen=True)
class Group:
    """One entry-point group, as the conformance tests drive it."""

    id: str
    group: str
    error: type[PluginLoadError]
    good_a: str
    good_b: str
    keys_a: frozenset[str]
    keys_b: frozenset[str]
    builtin_keys: frozenset[str]
    missing_module: str
    hook_raises: str
    not_conforming: str
    duplicates_a: str
    misspelled: str
    extra_argument: str
    builtins: Callable[[], Sequence[tuple[str, object]]]
    load: Callable[[EPs, Sequence[tuple[str, object]], str | None], tuple[Loaded, frozenset[str]]]
    register_late: Callable[[], None]

    def ep(self, name: str, value: str) -> EntryPoint:
        return EntryPoint(name=name, value=value, group=self.group)


def _load_packs(eps: EPs, builtins: Sequence[tuple[str, object]], pin: str | None) -> tuple[Loaded, frozenset[str]]:
    reg = ActionRegistry("main")
    loads = load_packs(reg, entry_points=eps, builtins=builtins, pin=pin, on_event=Events())
    reg.freeze()
    return packs_loaded(loads), frozenset(s.name for s in reg.registered())


def _packs_register_late() -> None:
    reg = ActionRegistry("main")
    load_packs(reg, entry_points=(), on_event=Events())
    reg.freeze()

    async def late(_session: object, work_key: str) -> None: ...

    reg.add(ActionSpec(feature="late", action="one", handler_path="late", fn=late, cadence="1h"))


def _load_providers(eps: EPs, builtins: Sequence[tuple[str, object]], pin: str | None) -> tuple[Loaded, frozenset[str]]:
    store = load_providers(entry_points=eps, builtins=builtins, pin=pin, on_event=Events())
    return store.loaded, frozenset(store)


def _providers_register_late() -> None:
    load_providers(builtins=[("keeps", kinds.keeps)], entry_points=(), on_event=Events())
    assert kinds.keeps.register is not None
    kinds.keeps.register(kinds.Kind("late"))


PACK_GROUP = Group(
    id="market",
    group=ENTRY_POINT_GROUP,
    error=PackLoadError,
    good_a=f"{PACKS}.good",
    good_b=f"{PACKS}.other",
    keys_a=frozenset({"report.weekly", "report.digest"}),
    keys_b=frozenset({"audit.nightly"}),
    builtin_keys=frozenset({"core.tick"}),
    missing_module=f"{PACKS}.nope",
    hook_raises=f"{PACKS}.broken:raises",
    not_conforming=f"{PACKS}.broken:not_a_registry",
    duplicates_a=f"{PACKS}.broken:conflicts",
    misspelled=f"{PACKS}.broken:misspelled",
    extra_argument=f"{PACKS}.broken:extra_argument",
    builtins=lambda: [("core", _BuiltinPack())],
    load=_load_packs,
    register_late=_packs_register_late,
)
PROVIDER_GROUP = Group(
    id="providers",
    group="llm_ports.providers",
    error=PluginLoadError,
    good_a=f"{KINDS}:alpha",
    good_b=f"{KINDS}:beta",
    keys_a=frozenset({"alpha"}),
    keys_b=frozenset({"beta"}),
    builtin_keys=frozenset({"core"}),
    missing_module="tests.adapters.provider_plugins.nope",
    hook_raises=f"{KINDS}:raises",
    not_conforming=f"{KINDS}:not_a_factory",
    duplicates_a=f"{KINDS}:alpha_again",
    misspelled=f"{KINDS}:misspelled",
    extra_argument=f"{KINDS}:extra_argument",
    builtins=lambda: [builtin("core", kinds.Kind("core"))],
    load=_load_providers,
    register_late=_providers_register_late,
)
GROUPS = pytest.mark.parametrize("g", [PACK_GROUP, PROVIDER_GROUP], ids=lambda g: g.id)


def _fails(g: Group, eps: EPs, *, builtins: Sequence[tuple[str, object]] = (), pin: str | None = None) -> PluginLoadError:
    with pytest.raises(g.error) as caught:
        g.load(eps, builtins, pin)
    err = caught.value
    assert err.group == g.group
    return err


@GROUPS
def test_r1_a_hook_with_an_argument_the_spec_lacks_is_refused_at_register(g: Group) -> None:
    err = _fails(g, [g.ep("extra", g.extra_argument)])
    assert (err.plugin, err.stage) == ("extra", "register")
    assert isinstance(err.__cause__, pluggy.PluginValidationError)


@GROUPS
def test_r3_r10_load_order_is_sorted_by_name_whatever_order_the_entry_points_arrive_in(g: Group) -> None:
    a, b = g.ep("a", g.good_a), g.ep("b", g.good_b)
    forward, _ = g.load([a, b], (), None)
    backward, keys = g.load([b, a], (), None)
    assert forward.plugins == backward.plugins == ("a", "b")
    assert keys == g.keys_a | g.keys_b


@GROUPS
def test_r4_an_import_failure_raises_naming_the_plugin_and_must_fire_both_ways(g: Group) -> None:
    good = [g.ep("a", g.good_a)]
    err = _fails(g, [*good, g.ep("missing", g.missing_module)])
    assert (err.plugin, err.stage) == ("missing", "import")
    assert "missing" in str(err) and "ModuleNotFoundError" in str(err)
    loaded, keys = g.load(good, (), None)
    assert loaded.plugins == ("a",) and keys == g.keys_a


@GROUPS
def test_r4_a_raising_hook_is_stage_hook(g: Group) -> None:
    err = _fails(g, [g.ep("raises", g.hook_raises)])
    assert (err.plugin, err.stage) == ("raises", "hook")
    assert "RuntimeError" in str(err)


@GROUPS
def test_r4_a_non_conforming_contribution_is_stage_register(g: Group) -> None:
    err = _fails(g, [g.ep("odd", g.not_conforming)])
    assert (err.plugin, err.stage) == ("odd", "register")


@GROUPS
def test_r5_a_duplicate_key_raises_and_names_both_plugins(g: Group) -> None:
    err = _fails(g, [g.ep("first", g.good_a), g.ep("second", g.duplicates_a)])
    assert (err.plugin, err.stage) == ("second", "duplicate")
    assert "'first'" in str(err) and "'second'" in str(err)


@GROUPS
def test_r6_the_pin_selects_exactly_and_a_missing_pinned_name_raises(g: Group) -> None:
    eps = [g.ep("a", g.good_a), g.ep("b", g.good_b)]
    only_builtins, keys = g.load(eps, g.builtins(), "")
    assert only_builtins.plugins == ("core",) and keys == g.builtin_keys
    assert only_builtins.origin == {"core": "builtin"}
    pinned, keys = g.load(eps, g.builtins(), " b ,core")
    assert pinned.plugins == ("core", "b") and keys == g.builtin_keys | g.keys_b
    err = _fails(g, eps, builtins=g.builtins(), pin="a,ghost")
    assert (err.plugin, err.stage) == ("ghost", "pin")
    every, keys = g.load(eps, g.builtins(), None)
    assert every.plugins == ("core", "a", "b") and keys == g.builtin_keys | g.keys_a | g.keys_b


@GROUPS
def test_r7_a_misspelled_hook_raises_at_check_pending(g: Group) -> None:
    err = _fails(g, [g.ep("typo", g.misspelled)])
    assert (err.plugin, err.stage) == ("typo", "pending")
    assert isinstance(err.__cause__, pluggy.PluginValidationError)


@GROUPS
def test_r2_builtins_load_with_no_distribution_metadata(g: Group) -> None:
    loaded, keys = g.load(lambda: (), g.builtins(), None)
    assert loaded.plugins == ("core",) and loaded.origin == {"core": "builtin"}
    assert keys == g.builtin_keys


@GROUPS
def test_r10_freeze_a_registration_after_load_raises(g: Group) -> None:
    with pytest.raises((RuntimeError, PluginLoadError), match="frozen"):
        g.register_late()
