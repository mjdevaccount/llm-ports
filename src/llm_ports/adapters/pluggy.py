"""Plugins found by entry point and loaded with pluggy (extra `[market]`): action packs and provider kinds.

Two entry-point groups share one loader:

- `llm_ports.market` (`market.ENTRY_POINT_GROUP`): action packs. A module declares its actions on
  its own `ActionRegistry` and returns it from a `market_actions` hook; `load_packs` adds each pack
  to the scheduler's registry.
- `llm_ports.providers` (`call.providers.ENTRY_POINT_GROUP`): provider kinds. A module calls
  `register(factory)` from a `register_providers` hook; `load_providers` returns the kinds.

    PACK = ActionRegistry("reports")

    @PACK.action(feature="report", action="weekly", cadence="7d", lane=3)
    async def weekly(session, work_key): ...

    @hookimpl
    def market_actions() -> ActionRegistry:
        return PACK

    # pyproject.toml
    [project.entry-points."llm_ports.market"]
    reports = "reports_pkg.market"

The process calls each loader once, at its composition root, passing the value of its pin setting
(`PACKS_PIN_ENV`, `PROVIDERS_PIN_ENV` by convention; the library reads no environment).

Invariants:
    - The hook specs are added before any plugin registers, so pluggy refuses a hook implementation
      whose arguments the spec does not declare.
    - Built-ins register first, by explicit name, needing no distribution metadata; entry points
      follow, sorted by name. No order derives from a set or from install order.
    - The pin: None loads the built-ins and every installed plugin; `""` the built-ins only;
      `"a,b"` the built-ins and exactly those, and a pinned name that is not installed raises.
    - Strict loading (the default) raises `PluginLoadError` (`PackLoadError` for packs) on any
      failure, naming the group, the plugin and the stage: `import`, `register` (pluggy refused it,
      or it contributed a wrong object), `hook` (its hook raised), `duplicate` (a key or a name
      another plugin holds; both are named), `pin`, `pending` (an unknown hook name, or no hook).
      `load_packs(strict=False)` records each failure as a `PackLoad` instead, for a diagnostic.
    - A registry is frozen after load: a strict `load_packs` that finishes freezes the
      `ActionRegistry` it loaded into, and `load_providers` returns its `ProviderRegistry` frozen,
      so a registration after the composition root raises.
    - Every load emits `plugins_loaded` (group, names, origins) once it finishes.
    - `market_pack_failed` carries `stage` in the `PackLoad.stage` vocabulary (`PackStage`) and
      `plugin_stage` in the `PluginLoadError.stage` vocabulary (`PluginStage`).

Rationale: ADR-0018 (packs), ADR-0022 (providers).
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from llm_ports._extras import extra
from llm_ports.call.providers import ENTRY_POINT_GROUP as PROVIDERS_GROUP
from llm_ports.call.providers import DuplicateProvider, Loaded, ProviderFactory, ProviderRegistry
from llm_ports.market import ENTRY_POINT_GROUP, ActionRegistry, DuplicateAction
from llm_ports.observability import EventHook, log_events

with extra("market"):
    import pluggy

#: pluggy's project name for packs: hook markers match plugins by it.
PROJECT: str = ENTRY_POINT_GROUP
#: pluggy's project name for provider kinds.
PROVIDERS_PROJECT: str = PROVIDERS_GROUP
#: Marks a pack module's `market_actions` implementation.
hookimpl: pluggy.HookimplMarker = pluggy.HookimplMarker(PROJECT)
#: Marks a provider plugin's `register_providers` implementation.
provider_hookimpl: pluggy.HookimplMarker = pluggy.HookimplMarker(PROVIDERS_PROJECT)
#: The conventional name of the setting that pins the pack set (the caller reads it).
PACKS_PIN_ENV = "LLM_PORTS_PLUGINS"
#: The conventional name of the setting that pins the provider-kind set (the caller reads it).
PROVIDERS_PIN_ENV = "LLM_PORTS_PROVIDERS"
_hookspec = pluggy.HookspecMarker(PROJECT)
_provider_hookspec = pluggy.HookspecMarker(PROVIDERS_PROJECT)
_log_market = log_events("llm_ports.market")
_log_call = log_events("llm_ports.call")

#: How one entry point ended.
PackOutcome = Literal["ok", "failed", "blocked"]
#: Where a failed pack stopped, as a `PackLoad` records it: importing its module, validating its
#: hooks, building its pack, joining the registry, or a pinned name that is not installed.
PackStage = Literal["import", "validate", "pack", "join", "pin"]
#: Where a plugin failed, as `PluginLoadError` names it.
PluginStage = Literal["import", "register", "hook", "duplicate", "pin", "pending"]

EntryPoints = Iterable[importlib.metadata.EntryPoint] | Callable[[], Iterable[importlib.metadata.EntryPoint]]


class PluginLoadError(ImportError):
    """A plugin that did not load; raised at the composition root, so the process does not start.

    Attributes:
        group: The entry-point group.
        plugin: The plugin's name (comma-joined for several pinned-but-missing names).
        stage: Where it failed.
    """

    def __init__(self, message: str, *, group: str, plugin: str, stage: PluginStage) -> None:
        super().__init__(f"{group}: plugin {plugin!r} failed at {stage}: {message}")
        self.group: str = group
        self.plugin: str = plugin
        self.stage: PluginStage = stage


class PackLoadError(PluginLoadError):
    """An action pack that did not load (`load_packs`, strict)."""


class PackSpec:
    """The hooks a pack module may implement."""

    @_hookspec
    def market_actions(self) -> ActionRegistry:  # type: ignore[empty-body]  # a hook spec has no body
        """Return the pack's actions."""


class ProviderHooks:
    """The hooks a provider plugin may implement."""

    @_provider_hookspec
    def register_providers(self, register: Callable[[ProviderFactory], None]) -> None:
        """Call `register(factory)` once per provider kind the plugin contributes."""


@dataclass(frozen=True)
class PackLoad:
    """One pack, loaded or not.

    Attributes:
        name: The entry point's name (the pack's name, and its pluggy plugin name).
        dist: The distribution that declares it, or None when unknown (and for a built-in).
        outcome: `ok`, `failed` or `blocked`.
        actions: The `feature.action` names the pack added; empty unless `ok`.
        stage: Where a failed pack stopped; None unless `failed`.
        reason: Why it failed; None unless `failed`.
        origin: `builtin`, `<dist>==<version>`, or None when unknown.
    """

    name: str
    dist: str | None
    outcome: PackOutcome
    actions: tuple[str, ...] = ()
    stage: PackStage | None = None
    reason: str | None = None
    origin: str | None = None


class _Fail(Exception):
    """One plugin's failure inside the loader, before it is raised or recorded."""

    def __init__(self, stage: PluginStage, pack_stage: PackStage, message: str) -> None:
        super().__init__(message)
        self.stage: PluginStage = stage
        self.pack_stage: PackStage = pack_stage


@dataclass(frozen=True)
class _Candidate:
    name: str
    dist: str | None
    origin: str
    load: Callable[[], object]


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _origin(ep: importlib.metadata.EntryPoint) -> tuple[str | None, str]:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None, "unknown"
    version = getattr(dist, "version", None)
    return dist.name, f"{dist.name}=={version}" if version else dist.name


def _given(obj: object) -> Callable[[], object]:
    """A loader for a built-in: it is already an object."""
    return lambda: obj


def _select(
    builtins: Sequence[tuple[str, object]], eps: EntryPoints | None, group: str, pin: str | None
) -> tuple[list[_Candidate], list[tuple[str, _Fail]]]:
    """The plugins to load, in load order, and the selection failures (pin, shadow, duplicate name)."""
    problems: list[tuple[str, _Fail]] = []
    out: list[_Candidate] = []
    for name, obj in builtins:
        if any(c.name == name for c in out):
            problems.append((name, _Fail("duplicate", "join", "two built-ins share the name")))
            continue
        out.append(_Candidate(name, None, "builtin", _given(obj)))
    found = importlib.metadata.entry_points(group=group) if eps is None else (eps() if callable(eps) else eps)
    available: dict[str, importlib.metadata.EntryPoint] = {}
    for ep in found:
        if ep.name in {c.name for c in out}:
            problems.append((ep.name, _Fail("duplicate", "join", "an entry point shadows the built-in of that name")))
        elif ep.name in available:
            dists = sorted(str(_origin(e)[1]) for e in (available[ep.name], ep))
            problems.append((ep.name, _Fail("duplicate", "join", f"declared by both {dists[0]!r} and {dists[1]!r}")))
        else:
            available[ep.name] = ep
    wanted = None if pin is None else {p.strip() for p in pin.split(",") if p.strip()}
    if wanted is not None:
        builtin_names = {n for n, _ in builtins}
        for name in sorted(wanted - set(available) - builtin_names):
            problems.append((name, _Fail("pin", "pin", f"pinned but not installed (installed: {sorted(available)})")))
    for name in sorted(available):
        if wanted is None or name in wanted:
            dist, origin = _origin(available[name])
            out.append(_Candidate(name, dist, origin, available[name].load))
    return out, problems


def _admit(pm: pluggy.PluginManager, cand: _Candidate) -> None:
    """Import and register one plugin, then prove pluggy knows every hook it implements."""
    try:
        plugin = cand.load()
    except Exception as exc:
        raise _Fail("import", "import", _reason(exc)) from exc
    try:
        pm.register(plugin, name=cand.name)
    except pluggy.PluginValidationError as exc:
        raise _Fail("register", "validate", _reason(exc)) from exc
    except ValueError as exc:  # pluggy: this plugin object, or this name, is already registered
        raise _Fail("duplicate", "validate", _reason(exc)) from exc
    try:
        pm.check_pending()
    except pluggy.PluginValidationError as exc:
        pm.unregister(name=cand.name)
        raise _Fail("pending", "validate", _reason(exc)) from exc


def _impl(pm: pluggy.PluginManager, hook: str, name: str) -> Callable[..., object]:
    impls = [i for i in getattr(pm.hook, hook).get_hookimpls() if i.plugin_name == name]
    if not impls:
        pm.unregister(name=name)
        raise _Fail("pending", "validate", f"implements no {hook} hook")
    fn: Callable[..., object] = impls[0].function
    return fn


def _join(pm: pluggy.PluginManager, cand: _Candidate, registry: ActionRegistry) -> tuple[str, ...]:
    """Build one pack and add it to `registry`, all or nothing."""
    build = _impl(pm, "market_actions", cand.name)
    try:
        pack = build()
    except Exception as exc:
        pm.unregister(name=cand.name)
        raise _Fail("hook", "pack", _reason(exc)) from exc
    if not isinstance(pack, ActionRegistry):
        pm.unregister(name=cand.name)
        raise _Fail("register", "pack", f"market_actions returned {type(pack).__name__}, not an ActionRegistry")
    try:
        added = registry.include(pack, origin=cand.name)
    except DuplicateAction as exc:
        pm.unregister(name=cand.name)
        raise _Fail("duplicate", "join", str(exc)) from exc
    except ValueError as exc:  # a spec this registry refuses (a clock or a source it does not accept)
        pm.unregister(name=cand.name)
        raise _Fail("register", "join", str(exc)) from exc
    return tuple(spec.name for spec in added)


def load_packs(
    registry: ActionRegistry,
    *,
    group: str = ENTRY_POINT_GROUP,
    block: Iterable[str] = (),
    entry_points: EntryPoints | None = None,
    on_event: EventHook | None = None,
    builtins: Sequence[tuple[str, object]] = (),
    pin: str | None = None,
    strict: bool = True,
) -> tuple[PackLoad, ...]:
    """Load the built-in packs, then every pack listed under `group`, into `registry`.

    Args:
        registry: The scheduler's registry; each pack joins it through `include`. Under
            `strict`, a load that finishes freezes it (`ActionRegistry.freeze`), so a declaration
            after the composition root raises; declare the caller's own actions first, or pass
            them as `builtins`. A diagnostic load (`strict=False`) leaves it open.
        group: The entry-point group to read.
        block: Pack names to skip (pluggy's blocked set); for tests. Production pins instead.
        entry_points: The entry points to load instead of the installed ones (or a callable that
            returns them).
        on_event: Receives `market_pack_loaded`, `market_pack_failed` (`stage` as `PackLoad.stage`,
            `plugin_stage` as `PluginLoadError.stage`), `market_pack_blocked` and `plugins_loaded`;
            without one, the stdlib logger `llm_ports.market`.
        builtins: `(name, module)` packs that ship with the caller, registered first, in order.
        pin: The pin setting's value; None when it is unset (see the module docstring).
        strict: Raise on the first failure. False records each failure and loads the rest.

    Returns:
        One `PackLoad` per pack, in load order.

    Raises:
        PackLoadError: Under `strict`, the first pack that failed, naming its stage.
        RuntimeError: `registry` is already frozen; nothing is loaded.
    """
    if registry.frozen:
        raise RuntimeError(f"action registry {registry.name!r} is frozen; load packs into it once, before freezing")
    emit = on_event or _log_market
    pm = pluggy.PluginManager(PROJECT)
    pm.add_hookspecs(PackSpec)
    for name in block:
        pm.set_blocked(name)
    candidates, problems = _select(builtins, entry_points, group, pin)
    loads: list[PackLoad] = []

    def record(name: str, dist: str | None, origin: str | None, fail: _Fail) -> None:
        load = PackLoad(name, dist, "failed", stage=fail.pack_stage, reason=str(fail)[:300], origin=origin)
        emit(
            logging.WARNING,
            "market_pack_failed",
            {"pack": name, "dist": dist, "stage": load.stage, "plugin_stage": fail.stage, "reason": load.reason},
        )
        if strict:
            raise PackLoadError(str(fail), group=group, plugin=name, stage=fail.stage) from fail.__cause__
        loads.append(load)

    for name, fail in problems:
        record(name, None, None, fail)
    for cand in candidates:
        if pm.is_blocked(cand.name):
            loads.append(PackLoad(cand.name, cand.dist, "blocked", origin=cand.origin))
            emit(logging.INFO, "market_pack_blocked", {"pack": cand.name, "dist": cand.dist})
            continue
        try:
            _admit(pm, cand)
            actions = _join(pm, cand, registry)
        except _Fail as fail:
            record(cand.name, cand.dist, cand.origin, fail)
            continue
        loads.append(PackLoad(cand.name, cand.dist, "ok", actions=actions, origin=cand.origin))
        emit(logging.INFO, "market_pack_loaded", {"pack": cand.name, "dist": cand.dist, "actions": list(actions)})
    if strict:
        registry.freeze()
    _emit_loaded(emit, packs_loaded(loads, group=group))
    return tuple(loads)


def packs_loaded(loads: Iterable[PackLoad], *, group: str = ENTRY_POINT_GROUP) -> Loaded:
    """The `Loaded` record of the packs that joined: names in load order, with their origins."""
    ok = [load for load in loads if load.outcome == "ok"]
    return Loaded(group, tuple(load.name for load in ok), {load.name: load.origin or "unknown" for load in ok})


def failed(loads: Iterable[PackLoad]) -> tuple[PackLoad, ...]:
    """The loads that failed (a blocked pack is a decision, not a failure)."""
    return tuple(load for load in loads if load.outcome == "failed")


def builtin(name: str, *factories: ProviderFactory) -> tuple[str, object]:
    """Return a built-in provider plugin that registers `factories`, for `load_providers(builtins=...)`.

    Args:
        name: The plugin's name.
        factories: The kinds it contributes.
    """

    class _Builtin:
        @provider_hookimpl
        def register_providers(self, register: Callable[[ProviderFactory], None]) -> None:
            for factory in factories:
                register(factory)

    return name, _Builtin()


def load_providers(
    *,
    builtins: Sequence[tuple[str, object]] = (),
    pin: str | None = None,
    entry_points: EntryPoints | None = None,
    on_event: EventHook | None = None,
) -> ProviderRegistry:
    """Load the built-in provider plugins, then every one listed under `llm_ports.providers`.

    Args:
        builtins: `(name, plugin)` pairs registered first, in order (`builtin(...)` makes one).
        pin: The pin setting's value; None when it is unset (see the module docstring).
        entry_points: The entry points to load instead of the installed ones (or a callable that
            returns them).
        on_event: Receives `plugins_loaded`; without one, the stdlib logger `llm_ports.call`.

    Returns:
        The kinds, frozen, with `loaded` naming each plugin and its origin.

    Raises:
        PluginLoadError: The first plugin that failed, naming its stage; a kind registered by two
            plugins is `duplicate` and names both.
    """
    group = PROVIDERS_GROUP
    pm = pluggy.PluginManager(PROVIDERS_PROJECT)
    pm.add_hookspecs(ProviderHooks)
    store = ProviderRegistry()
    candidates, problems = _select(builtins, entry_points, group, pin)
    for name, fail in problems:
        raise PluginLoadError(str(fail), group=group, plugin=name, stage=fail.stage)
    for cand in candidates:
        try:
            _admit(pm, cand)
            hook = _impl(pm, "register_providers", cand.name)
        except _Fail as fail:
            raise PluginLoadError(str(fail), group=group, plugin=cand.name, stage=fail.stage) from fail.__cause__

        def register(factory: ProviderFactory, _plugin: str = cand.name) -> None:
            try:
                store.add(factory, plugin=_plugin)
            except DuplicateProvider as exc:
                raise PluginLoadError(str(exc), group=group, plugin=_plugin, stage="duplicate") from exc
            except (TypeError, RuntimeError) as exc:
                raise PluginLoadError(str(exc), group=group, plugin=_plugin, stage="register") from exc

        try:
            hook(register=register)
        except PluginLoadError:
            raise
        except Exception as exc:
            raise PluginLoadError(_reason(exc), group=group, plugin=cand.name, stage="hook") from exc
    loaded = Loaded(group, tuple(c.name for c in candidates), {c.name: c.origin for c in candidates})
    store.freeze(loaded)
    _emit_loaded(on_event or _log_call, loaded)
    return store


def _emit_loaded(emit: EventHook, loaded: Loaded) -> None:
    emit(
        logging.INFO,
        "plugins_loaded",
        {"group": loaded.group, "names": list(loaded.plugins), "origin": dict(loaded.origin)},
    )


__all__ = [
    "PACKS_PIN_ENV",
    "PROJECT",
    "PROVIDERS_PIN_ENV",
    "PROVIDERS_PROJECT",
    "Loaded",
    "PackLoad",
    "PackLoadError",
    "PackOutcome",
    "PackSpec",
    "PackStage",
    "PluginLoadError",
    "PluginStage",
    "ProviderHooks",
    "builtin",
    "failed",
    "hookimpl",
    "load_packs",
    "load_providers",
    "packs_loaded",
    "provider_hookimpl",
]
