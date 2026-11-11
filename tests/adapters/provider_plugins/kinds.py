"""Provider plugins for the loader tests: two good ones, and one per failure stage."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from llm_ports.adapters.pluggy import provider_hookimpl
from llm_ports.call.models import ModelInfo, ProviderSpec
from llm_ports.call.providers import BuildDeps, Built, ProviderFactory


class Kind:
    """A provider kind that builds nothing; it records what `build` was given."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls: list[dict[str, Any]] = []

    def build(
        self, spec: ProviderSpec, *, url: str | None, secret: str | None, models: Sequence[ModelInfo], deps: BuildDeps
    ) -> Built:
        self.calls.append({"spec": spec, "url": url, "secret": secret, "models": tuple(models), "deps": deps})
        return Built()


Register = Callable[[ProviderFactory], None]


class Alpha:
    @provider_hookimpl
    def register_providers(self, register: Register) -> None:
        register(Kind("alpha"))


class Beta:
    @provider_hookimpl
    def register_providers(self, register: Register) -> None:
        register(Kind("beta"))


class ExtraArgument:
    @provider_hookimpl
    def register_providers(self, register: Register, registry: object) -> None:
        register(Kind("extra"))


class Raises:
    @provider_hookimpl
    def register_providers(self, register: Register) -> None:
        raise RuntimeError("provider plugin failed")


class NotAFactory:
    @provider_hookimpl
    def register_providers(self, register: Register) -> None:
        register(object())  # type: ignore[arg-type]  # the non-conforming object under test


class AlphaAgain:
    @provider_hookimpl
    def register_providers(self, register: Register) -> None:
        register(Kind("alpha"))


class Misspelled:
    @provider_hookimpl
    def register_provider(self, register: Register) -> None:  # the spec's hook is register_providers
        register(Kind("misspelled"))


class NoHook:
    """Implements nothing."""


class Keeps:
    """Keeps its `register` to call after the load (a plugin that registers late)."""

    def __init__(self) -> None:
        self.register: Register | None = None

    @provider_hookimpl
    def register_providers(self, register: Register) -> None:
        self.register = register


alpha = Alpha()
beta = Beta()
extra_argument = ExtraArgument()
raises = Raises()
not_a_factory = NotAFactory()
alpha_again = AlphaAgain()
misspelled = Misspelled()
no_hook = NoHook()
keeps = Keeps()

__all__ = [
    "Kind",
    "alpha",
    "alpha_again",
    "beta",
    "extra_argument",
    "keeps",
    "misspelled",
    "no_hook",
    "not_a_factory",
    "raises",
]
