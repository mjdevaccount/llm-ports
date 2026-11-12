"""Providers by name: the factory a kind registers, and `build`, which turns a table into backends.

A schema 2 tier table names, for every model, the provider that serves it; a provider names a
registered KIND (`openai_compatible`) and, by name only, the URL key and the secret it needs.
`build` resolves those names through the caller's `urls` and `secrets`, hands each provider to its
kind's `ProviderFactory`, and returns `Backends`, which a consumer's facade asks for the
`Generator` of a tier:

    providers = adapters.pluggy.load_providers(builtins=[builtin("openai_compatible", factory)], pin=pin)
    backends = build(TABLE, urls=topology, secrets=vault.get, deps=BuildDeps(), factories=providers)
    res = await call.generate(ROLE, prompt=text, generator=backends.generator_for(ROLE.tier))

Kinds arrive through the provider registry (`adapters.pluggy.load_providers`, entry-point group
`ENTRY_POINT_GROUP`), or as a plain mapping in a test. Only the core types live here: no vendor
library and no entry-point reading.

Invariants:
    - Everything that can be wrong with a table and its environment is raised at `build`, at boot:
      an unknown kind, a URL key the caller cannot resolve, a secret that resolves to nothing or
      to a value no header can carry (whitespace, a control or a non-ASCII character).
      Nothing waits for the first call to find a missing key.
    - A secret value is never logged, stamped or put in an error; only its name is.
    - A kind's key is flat and global: two plugins registering one kind raise, naming both.
    - A registry is frozen once loaded: a later `add` raises.

Rationale: ADR-0022.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from llm_ports._headers import HEADER_VALUE_RULE, header_value_ok
from llm_ports.call.models import ModelInfo, ProviderSpec, TableInvalid
from llm_ports.call.roles import TierTable
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import Embedder, Generator, Limiter, Pricer, Probe, Reranker, WirePolicy

#: The entry-point group a distribution lists its provider kinds under.
ENTRY_POINT_GROUP = "llm_ports.providers"
#: Where events go when `BuildDeps.on_event` is None.
_log_event = log_events("llm_ports.call")


@dataclass(frozen=True, slots=True)
class Loaded:
    """Which plugins a process loaded into one registry, and from where.

    Attributes:
        group: The entry-point group.
        plugins: Plugin names in load order: built-ins as given, then entry points sorted by name.
        origin: Plugin name to `builtin`, `caller`, or `<distribution>==<version>`.
    """

    group: str
    plugins: tuple[str, ...]
    origin: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BuildDeps:
    """What every factory may share: the call's planes, passed once to `build`.

    Attributes:
        policy: The wire policy every built client uses when a request carries none; None takes
            each adapter's own default.
        limiter: The concurrency limiter every built generator takes its slots from; None for no
            in-process limit.
        pricer: Prices a reply's usage the gateway did not price.
        transport: An HTTP transport every client is built on (a test's mock, a counting proxy).
        on_event: Receives the build's events and every built client's; without one, the stdlib
            logger `llm_ports.call`.
    """

    policy: WirePolicy | None = None
    limiter: Limiter | None = None
    pricer: Pricer | None = None
    transport: Any | None = None
    on_event: EventHook | None = None


@dataclass(frozen=True, slots=True)
class Built:
    """What one factory built for one provider. A kind builds the ports it serves; the rest are None.

    Attributes:
        generator: The `Generator` for every model of the provider (the model id is on the request).
        chat: The chat port; None until the chat row of the work list defines it.
        embedder_for: Model id to an `Embedder`.
        reranker_for: Model id to a `Reranker`.
        probe: The provider's health probe.
    """

    generator: Generator | None = None
    chat: object | None = None
    embedder_for: Callable[[str], Embedder] | None = None
    reranker_for: Callable[[str], Reranker] | None = None
    probe: Probe | None = None


@runtime_checkable
class ProviderFactory(Protocol):
    """Builds the clients of one provider kind."""

    @property
    def kind(self) -> str:
        """The kind's registered name, the value of a provider's `kind` key."""
        ...

    def build(
        self, spec: ProviderSpec, *, url: str | None, secret: str | None, models: Sequence[ModelInfo], deps: BuildDeps
    ) -> Built:
        """Build the provider's clients.

        Args:
            spec: The provider's section; the factory validates `spec.options`.
            url: The resolved URL of `spec.base_url`; None when the provider names none.
            secret: The resolved secret of `spec.secret`; None for a keyless provider.
            models: The table's models this provider serves, in table order.
            deps: The shared planes.

        Raises:
            TableInvalid: The provider's section asks for something the kind cannot do.
        """
        ...


class DuplicateProvider(ValueError):
    """Two plugins registered one provider kind; the message names both."""


class ProviderRegistry(Mapping[str, ProviderFactory]):
    """Provider kind to its factory, with the registry rules: a duplicate raises, a frozen one is closed.

    Args:
        factories: Factories to add now, as plugin `caller`.
    """

    __slots__ = ("_factories", "_loaded", "_owner")

    def __init__(self, factories: Sequence[ProviderFactory] = ()) -> None:
        self._factories: dict[str, ProviderFactory] = {}
        self._owner: dict[str, str] = {}
        self._loaded: Loaded | None = None
        for factory in factories:
            self.add(factory)

    def add(self, factory: ProviderFactory, *, plugin: str = "caller") -> None:
        """Register `factory` under its kind, contributed by `plugin`.

        Raises:
            TypeError: `factory` is not a `ProviderFactory`.
            DuplicateProvider: Another plugin registered the kind.
            RuntimeError: The registry is frozen.
        """
        if self._loaded is not None:
            raise RuntimeError(f"provider registry is frozen; plugin {plugin!r} registered after load")
        if not isinstance(factory, ProviderFactory) or not isinstance(factory.kind, str) or not factory.kind:
            raise TypeError(f"plugin {plugin!r} registered {factory!r}, not a ProviderFactory")
        held = self._owner.get(factory.kind)
        if held is not None:
            raise DuplicateProvider(f"provider kind {factory.kind!r} registered by both {held!r} and {plugin!r}")
        self._factories[factory.kind] = factory
        self._owner[factory.kind] = plugin

    def freeze(self, loaded: Loaded | None = None) -> None:
        """Close the registry, recording what loaded it; a later `add` raises."""
        self._loaded = loaded or Loaded(
            ENTRY_POINT_GROUP, tuple(dict.fromkeys(self._owner.values())), {p: "caller" for p in self._owner.values()}
        )

    @property
    def frozen(self) -> bool:
        """Whether `freeze` has run."""
        return self._loaded is not None

    @property
    def loaded(self) -> Loaded:
        """What loaded the registry; derived from the plugins so far while it is open."""
        if self._loaded is not None:
            return self._loaded
        plugins = tuple(dict.fromkeys(self._owner.values()))
        return Loaded(ENTRY_POINT_GROUP, plugins, {p: "caller" for p in plugins})

    def plugin_of(self, kind: str) -> str:
        """The plugin that registered `kind`."""
        return self._owner[kind]

    def __getitem__(self, kind: str) -> ProviderFactory:
        return self._factories[kind]

    def __iter__(self) -> Iterator[str]:
        return iter(self._factories)

    def __len__(self) -> int:
        return len(self._factories)

    def __repr__(self) -> str:
        return f"ProviderRegistry({sorted(self._factories)}, frozen={self.frozen})"


class Backends:
    """The clients `build` made, looked up by tier, model key or provider.

    Attributes:
        table: The table the backends were built from.
        loaded: Which provider kinds were registered, by which plugins, from where.
    """

    __slots__ = ("_built", "_closed", "loaded", "table")

    def __init__(self, table: TierTable, built: Mapping[str, Built], loaded: Loaded) -> None:
        self.table: TierTable = table
        self.loaded: Loaded = loaded
        self._built = MappingProxyType(dict(built))
        self._closed = False

    @property
    def providers(self) -> Mapping[str, Built]:
        """Provider name to what was built for it."""
        return self._built

    def _for_tier(self, tier: str) -> tuple[str, Built]:
        if tier not in self.table.tiers:
            raise TableInvalid(f"tiers.{tier}", f"unknown tier (have {sorted(self.table.tiers)})")
        info = self.table.tiers[tier].info
        assert info is not None  # build refuses a schema 1 table
        return info.provider, self._built[info.provider]

    def generator_for(self, tier: str) -> Generator:
        """The `Generator` that serves `tier`'s model.

        Raises:
            TableInvalid: The tier is unknown.
            LookupError: The provider's kind builds no `Generator`.
        """
        provider, built = self._for_tier(tier)
        if built.generator is None:
            raise LookupError(f"provider {provider!r} builds no Generator")
        return built.generator

    def chat_for(self, tier: str) -> object:
        """The chat port that serves `tier`'s model.

        Raises:
            TableInvalid: The tier is unknown.
            LookupError: The provider's kind builds no chat port (none does until the chat row of
                the work list).
        """
        provider, built = self._for_tier(tier)
        if built.chat is None:
            raise LookupError(f"provider {provider!r} builds no chat port")
        return built.chat

    def embedder_for(self, model_key: str) -> Embedder:
        """The `Embedder` for the model under `model_key`.

        Raises:
            TableInvalid: The key is unknown, or the model does not state `embed`.
            LookupError: The provider's kind builds no `Embedder`.
        """
        info = self._model(model_key, "embed")
        build_for = self._built[info.provider].embedder_for
        if build_for is None:
            raise LookupError(f"provider {info.provider!r} builds no Embedder")
        return build_for(info.id)

    def reranker_for(self, model_key: str) -> Reranker:
        """The `Reranker` for the model under `model_key`.

        Raises:
            TableInvalid: The key is unknown, or the model does not state `rerank`.
            LookupError: The provider's kind builds no `Reranker`.
        """
        info = self._model(model_key, "rerank")
        build_for = self._built[info.provider].reranker_for
        if build_for is None:
            raise LookupError(f"provider {info.provider!r} builds no Reranker")
        return build_for(info.id)

    def _model(self, key: str, capability: str) -> ModelInfo:
        info = self.table.model_info.get(key)
        if info is None:
            raise TableInvalid(f"models.{key}", f"unknown model key (have {sorted(self.table.model_info)})")
        if capability not in info.supports:
            raise TableInvalid(f"models.{key}.supports", f"does not state {capability!r}")
        return info

    def probe_for(self, provider: str) -> Probe:
        """The health probe of `provider`.

        Raises:
            TableInvalid: The provider was not built (unknown, or no model uses it).
            LookupError: Its kind builds no probe.
        """
        built = self._built.get(provider)
        if built is None:
            raise TableInvalid(f"providers.{provider}", f"not built (built: {sorted(self._built)})")
        if built.probe is None:
            raise LookupError(f"provider {provider!r} builds no Probe")
        return built.probe

    async def aclose(self) -> None:
        """Close every built client that holds a resource, each once (a kind may share one). Idempotent."""
        if self._closed:
            return
        self._closed = True
        seen: set[int] = set()
        for built in self._built.values():
            for client in (built.generator, built.chat, built.probe):
                close = getattr(client, "aclose", None)
                if callable(close) and id(client) not in seen:
                    seen.add(id(client))
                    await close()


def build(
    table: TierTable,
    *,
    urls: Mapping[str, str],
    secrets: Callable[[str], str | None],
    deps: BuildDeps,
    factories: Mapping[str, ProviderFactory] | None = None,
) -> Backends:
    """Build the clients of every provider the table's models use.

    Args:
        table: A schema 2 tier table.
        urls: Topology key to URL, for each provider's `base_url`.
        secrets: Secret name to its value, or None when the caller has none by that name. Called
            once per provider that names a secret; the value goes to the factory and nowhere else.
        deps: The shared planes every factory gets.
        factories: Kind to factory: a `ProviderRegistry` from `adapters.pluggy.load_providers`, or
            a plain mapping. None registers no kind, so any provider is refused.

    Returns:
        The `Backends`, after emitting `llm_backends_built` (the table's fingerprint, each
        provider's kind, and which plugins registered the kinds).

    Raises:
        TableInvalid: The table is schema 1, a provider names an unregistered kind, a URL key the
            caller does not resolve, or a secret that resolves to nothing or to a value holding
            whitespace, a control or a non-ASCII character (the error names the secret, never its
            value); or a factory refuses its section.
    """
    if table.schema_version < 2:
        raise TableInvalid("schema_version", "build needs a schema_version 2 table: a schema 1 table names no provider")
    registry: Mapping[str, ProviderFactory] = factories if factories is not None else {}
    loaded = registry.loaded if isinstance(registry, ProviderRegistry) else _caller_loaded(registry)
    used = list(dict.fromkeys(info.provider for info in table.model_info.values()))
    built: dict[str, Built] = {}
    for name in used:
        spec = table.providers[name]
        path = f"providers.{name}"
        factory = registry.get(spec.kind)
        if factory is None:
            raise TableInvalid(f"{path}.kind", f"unknown provider kind {spec.kind!r} (registered: {sorted(registry)})")
        url = None
        if spec.base_url is not None:
            url = urls.get(spec.base_url)
            if url is None:
                raise TableInvalid(f"{path}.base_url", f"no URL for key {spec.base_url!r} (have {sorted(urls)})")
        secret = None
        if spec.secret is not None:
            secret = secrets(spec.secret)
            if not secret:
                raise TableInvalid(f"{path}.secret", f"secret {spec.secret!r} resolved to nothing")
            if not header_value_ok(secret):
                raise TableInvalid(
                    f"{path}.secret",
                    f"secret {spec.secret!r} {HEADER_VALUE_RULE} (a trailing newline from a secret file is the usual cause)",
                )
        models = tuple(info for info in table.model_info.values() if info.provider == name)
        built[name] = factory.build(spec, url=url, secret=secret, models=models, deps=deps)
    (deps.on_event or _log_event)(
        logging.INFO,
        "llm_backends_built",
        {
            "fingerprint": table.fingerprint(),
            "providers": {n: table.providers[n].kind for n in built},
            "group": loaded.group,
            "plugins": list(loaded.plugins),
            "origin": dict(loaded.origin),
        },
    )
    return Backends(table, built, loaded)


def _caller_loaded(registry: Mapping[str, ProviderFactory]) -> Loaded:
    kinds = tuple(sorted(registry))
    return Loaded(ENTRY_POINT_GROUP, kinds, dict.fromkeys(kinds, "caller"))


__all__ = [
    "ENTRY_POINT_GROUP",
    "Backends",
    "BuildDeps",
    "Built",
    "DuplicateProvider",
    "Loaded",
    "ProviderFactory",
    "ProviderRegistry",
    "build",
]
