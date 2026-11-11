"""`call.providers`: the registry rules, and `build`, which raises at boot for anything missing (spec test 8)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest

from llm_ports.call import providers
from llm_ports.call.models import ModelInfo, ProviderSpec, TableInvalid
from llm_ports.call.providers import Backends, BuildDeps, Built, DuplicateProvider, Loaded, ProviderRegistry, build
from llm_ports.ports import Health
from llm_ports.testing import StubGenerator
from tests.adapters.provider_plugins.kinds import Kind
from tests.call.test_table_v2 import V2, _raw
from tests.conftest import TABLE
from tests.market._events import Events

URLS = {"gateway": "http://gw.test/v1", "local": "http://box.test/v1"}
SECRETS = {"gateway_key": "sk-test-value"}


class Probed:
    async def probe(self) -> Health:
        return Health("up", models=("small-chat",))


class Closing:
    """A client with an `aclose`, counting the closes."""

    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class Full(Kind):
    """A kind that builds every port, so each lookup has something to find."""

    def __init__(self, kind: str = "openai_compatible") -> None:
        super().__init__(kind)
        self.generator = StubGenerator("x")
        self.chat = Closing()

    def build(
        self, spec: ProviderSpec, *, url: str | None, secret: str | None, models: Sequence[ModelInfo], deps: BuildDeps
    ) -> Built:
        super().build(spec, url=url, secret=secret, models=models, deps=deps)
        return Built(
            generator=self.generator,
            chat=self.chat,
            embedder_for=lambda model_id: f"embedder:{model_id}",  # type: ignore[arg-type,return-value]
            reranker_for=lambda model_id: f"reranker:{model_id}",  # type: ignore[arg-type,return-value]
            probe=Probed(),
        )


def _build(
    factories: Any, *, secrets: dict[str, str] | None = None, urls: dict[str, str] | None = None, **kw: Any
) -> Backends:
    vault = SECRETS if secrets is None else secrets
    return build(V2, urls=URLS if urls is None else urls, secrets=vault.get, deps=BuildDeps(**kw), factories=factories)


# -- spec test 8 --------------------------------------------------------------------------------
def test_8_a_missing_secret_raises_at_build_naming_the_name_not_a_value() -> None:
    with pytest.raises(TableInvalid) as caught:
        _build({"openai_compatible": Kind("openai_compatible")}, secrets={})
    assert caught.value.path == "providers.gateway.secret"
    assert "'gateway_key' resolved to nothing" in str(caught.value)
    with pytest.raises(TableInvalid):
        _build({"openai_compatible": Kind("openai_compatible")}, secrets={"gateway_key": ""})


def test_8_a_keyless_provider_builds_and_each_factory_gets_its_own_models() -> None:
    kind, events = Kind("openai_compatible"), Events()
    backends = _build({"openai_compatible": kind}, on_event=events)
    by_provider = {c["spec"].name: c for c in kind.calls}
    assert list(by_provider) == ["local", "gateway"], "built in the order the models name them"
    assert (by_provider["local"]["url"], by_provider["local"]["secret"]) == ("http://box.test/v1", None)
    assert (by_provider["gateway"]["url"], by_provider["gateway"]["secret"]) == ("http://gw.test/v1", "sk-test-value")
    assert [m.key for m in by_provider["local"]["models"]] == ["chat", "web", "embed"]
    assert set(backends.providers) == {"local", "gateway"}
    (level, name, fields) = events.seen[0]
    assert (level, name) == (logging.INFO, "llm_backends_built")
    assert fields["fingerprint"] == V2.fingerprint()
    assert fields["providers"] == {"local": "openai_compatible", "gateway": "openai_compatible"}
    assert fields["plugins"] == ["openai_compatible"] and fields["origin"] == {"openai_compatible": "caller"}
    assert "sk-test-value" not in repr(events.seen), "the secret's value is never in an event"


@pytest.mark.parametrize(
    ("factories", "urls", "path", "match"),
    [
        ({}, URLS, "providers.local.kind", "unknown provider kind 'openai_compatible' \\(registered: \\[\\]\\)"),
        (None, URLS, "providers.local.kind", "registered: \\[\\]"),
        (
            {"openai_compatible": Kind("openai_compatible")},
            {"gateway": "x"},
            "providers.local.base_url",
            "no URL for key 'local'",
        ),
    ],
)
def test_build_raises_for_an_unknown_kind_or_url_key(
    factories: Any, urls: dict[str, str], path: str, match: str
) -> None:
    with pytest.raises(TableInvalid, match=match) as caught:
        _build(factories, urls=urls)
    assert caught.value.path == path


def test_build_refuses_a_schema_1_table() -> None:
    with pytest.raises(TableInvalid, match="schema_version 2") as caught:
        build(TABLE, urls={}, secrets=lambda _n: None, deps=BuildDeps())
    assert caught.value.path == "schema_version"


def test_a_provider_no_model_uses_is_not_built() -> None:
    from llm_ports.call import roles

    raw = _raw()
    raw["providers"]["spare"] = {"kind": "nobody_registered_this"}
    backends = build(
        roles.load_tiers(raw),
        urls=URLS,
        secrets=SECRETS.get,
        deps=BuildDeps(),
        factories={"openai_compatible": Kind("x")},
    )
    assert "spare" not in backends.providers
    with pytest.raises(TableInvalid, match="not built") as caught:
        backends.probe_for("spare")
    assert caught.value.path == "providers.spare"


# -- Backends lookups ---------------------------------------------------------------------------
async def test_backends_find_each_port_by_tier_model_key_or_provider() -> None:
    full = Full()
    backends = _build({"openai_compatible": full})
    assert backends.generator_for("local") is full.generator and backends.generator_for("cloud") is full.generator
    assert backends.chat_for("local") is full.chat
    assert backends.embedder_for("embed") == "embedder:small-embed"  # type: ignore[comparison-overlap]
    assert (await backends.probe_for("local").probe()).state == "up"
    with pytest.raises(TableInvalid, match="does not state 'embed'") as caught:
        backends.embedder_for("chat")
    assert caught.value.path == "models.chat.supports"
    with pytest.raises(TableInvalid, match="unknown model key") as caught:
        backends.reranker_for("nope")
    assert caught.value.path == "models.nope"
    with pytest.raises(TableInvalid, match="unknown tier") as caught:
        backends.generator_for("nope")
    assert caught.value.path == "tiers.nope"
    await backends.aclose()
    await backends.aclose()
    assert full.chat.closed == 1, "aclose is idempotent"


def test_a_kind_that_builds_nothing_raises_lookup_error_per_port() -> None:
    from llm_ports.call import roles

    raw = _raw()
    raw["models"]["embed"]["supports"] = ["embed", "rerank"]
    backends = build(
        roles.load_tiers(raw),
        urls=URLS,
        secrets=SECRETS.get,
        deps=BuildDeps(),
        factories={"openai_compatible": Kind("x")},
    )
    for lookup, arg in (
        (backends.generator_for, "local"),
        (backends.chat_for, "local"),
        (backends.embedder_for, "embed"),
        (backends.reranker_for, "embed"),
        (backends.probe_for, "local"),
    ):
        with pytest.raises(LookupError, match="builds no"):
            lookup(arg)


def test_reranker_lookup_reaches_a_kind_that_builds_one() -> None:
    from llm_ports.call import roles

    raw = _raw()
    raw["models"]["embed"]["supports"] = ["rerank"]
    backends = build(
        roles.load_tiers(raw), urls=URLS, secrets=SECRETS.get, deps=BuildDeps(), factories={"openai_compatible": Full()}
    )
    assert backends.reranker_for("embed") == "reranker:small-embed"  # type: ignore[comparison-overlap]


# -- the registry ---------------------------------------------------------------------------------
def test_the_registry_refuses_a_duplicate_a_non_factory_and_a_late_add() -> None:
    reg = ProviderRegistry([Kind("a")])
    assert reg.plugin_of("a") == "caller" and not reg.frozen
    assert reg.loaded == Loaded(providers.ENTRY_POINT_GROUP, ("caller",), {"caller": "caller"})
    with pytest.raises(DuplicateProvider, match="'a' registered by both 'caller' and 'other'"):
        reg.add(Kind("a"), plugin="other")
    with pytest.raises(TypeError, match="not a ProviderFactory"):
        reg.add(object(), plugin="odd")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reg.add(Kind(""), plugin="blank")
    reg.freeze()
    assert reg.frozen and reg.loaded.plugins == ("caller",)
    with pytest.raises(RuntimeError, match="frozen"):
        reg.add(Kind("b"))
    assert list(reg) == ["a"] and len(reg) == 1 and "frozen=True" in repr(reg)


def test_build_reports_the_registry_that_loaded_the_kinds() -> None:
    reg = ProviderRegistry()
    reg.add(Kind("openai_compatible"), plugin="core")
    reg.freeze(Loaded(providers.ENTRY_POINT_GROUP, ("core",), {"core": "builtin"}))
    events = Events()
    backends = _build(reg, on_event=events)
    assert backends.loaded.origin == {"core": "builtin"}
    assert events.seen[0][2]["origin"] == {"core": "builtin"}
