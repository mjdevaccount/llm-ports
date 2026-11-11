"""The model table: schema 1 loads as 1.22.0 did; schema 2 states providers, models, policies and fallbacks.

Spec routing tests 1 (schema 1 snapshot) and 2 (every `TableInvalid`, naming its path), plus the
fingerprint, `require`, and the provider-neutral thinking switch.
"""

from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import Mapping
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pytest

from llm_ports.call import models, roles
from llm_ports.call.models import DEFAULT_FALLBACK_ON, ModelInfo, TableInvalid
from tests.conftest import FIXTURES, TABLE

V2_PATH = FIXTURES / "llm_tiers_v2.toml"
V2 = roles.load_tiers(V2_PATH)
SNAPSHOT = FIXTURES / "snapshots" / "llm_tiers_v1_1.22.0.json"
V1_FIELDS = ("name", "model", "timeout_s", "self_hosted")


def _raw() -> dict[str, Any]:
    return tomllib.loads(V2_PATH.read_text(encoding="utf-8"))


def _invalid(raw: dict[str, Any], path: str, match: str | None = None) -> TableInvalid:
    with pytest.raises(TableInvalid, match=match) as caught:
        roles.load_tiers(raw)
    assert caught.value.path == path
    assert str(caught.value).startswith(f"{path}: ")
    return caught.value


# -- spec test 1: a schema 1 table loads to the 1.22.0 snapshot ---------------------------------
def test_1_a_schema_1_table_loads_to_the_1_22_0_snapshot() -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    got = {
        "ladder": dict(TABLE.ladder),
        "models": dict(TABLE.models),
        "tiers": {n: {f: getattr(t, f) for f in V1_FIELDS} for n, t in TABLE.tiers.items()},
    }
    assert got == expected
    assert TABLE.schema_version == 1 and not TABLE.model_info and not TABLE.providers and not TABLE.policies
    for tier in TABLE.tiers.values():
        assert (tier.info, tier.policy, tier.fallback, tier.fallback_on) == (None, None, (), DEFAULT_FALLBACK_ON)
        assert tier.table_fingerprint == TABLE.fingerprint()
    assert TABLE.tiers["local"] == roles.Tier("local", "small-chat", 120.0, True), "a hand-built tier still equals it"


def test_a_schema_1_table_with_a_schema_2_section_loads_unchanged_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    v1 = tomllib.loads((FIXTURES / "llm_tiers.toml").read_text(encoding="utf-8"))
    extra = {"providers": {"gw": {"kind": "openai_compatible"}}, "policies": {"box": {"attempts": 3}}}
    with caplog.at_level(logging.WARNING, logger="llm_ports.call"):
        table = roles.load_tiers({**v1, **extra})
    assert table == TABLE and table.fingerprint() == TABLE.fingerprint(), "1.22 ignored the sections; so does 1.x"
    assert not table.providers and not table.policies and table.schema_version == 1
    assert "llm_table_ignored" in caplog.text and "'providers', 'policies'" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm_ports.call"):
        roles.load_tiers(v1)
    assert "llm_table_ignored" not in caplog.text, "must fire both ways: a plain schema 1 table is silent"


def test_an_unknown_schema_version_is_refused() -> None:
    v1 = tomllib.loads((FIXTURES / "llm_tiers.toml").read_text(encoding="utf-8"))
    _invalid({**v1, "schema_version": 3}, "schema_version", "unknown version 3")
    _invalid({**v1, "schema_version": True}, "schema_version")
    assert roles.load_tiers({**v1, "schema_version": 1}).fingerprint() == TABLE.fingerprint()


# -- schema 2 ---------------------------------------------------------------------------------
def test_a_schema_2_table_states_its_models_providers_policies_and_fallbacks() -> None:
    assert V2.schema_version == 2
    assert dict(V2.models) == {"chat": "small-chat", "web": "small-web", "embed": "small-embed", "cloud": "gpt-5-nano"}
    chat = V2.model_info["chat"]
    assert chat == ModelInfo(
        key="chat",
        id="small-chat",
        provider="local",
        supports=frozenset({"json_schema", "vision", "thinking", "tools"}),
        context_window=65536,
        max_output=8192,
        thinking="chat_template",
        slots=1,
        group="gpu",
        self_hosted=True,
        usd_per_request=Decimal("0"),
    )
    web = V2.model_info["web"]
    assert (web.context_window, web.max_output, web.usd_per_request, web.thinking) == (None, None, None, "none")
    gateway = V2.providers["gateway"]
    assert (gateway.kind, gateway.base_url, gateway.secret, gateway.gateway, dict(gateway.options)) == (
        "openai_compatible",
        "gateway",
        "gateway_key",
        None,
        {},
    )
    assert V2.providers["local"].secret is None
    local = V2.tiers["local"]
    assert (local.model, local.timeout_s, local.self_hosted, local.info) == ("small-chat", 120.0, True, chat)
    assert local.fallback == ("cloud",) and local.fallback_on == DEFAULT_FALLBACK_ON
    assert local.policy == {"rate_limited": {"attempts": 45, "wait_initial_s": 2.0}}
    with pytest.raises(TypeError):
        local.policy["x"] = 1  # type: ignore[index]  # the declared policy is read-only
    cloud = V2.tiers["cloud"]
    assert cloud.self_hosted is False and cloud.fallback_on == frozenset({"overloaded", "quota"})
    assert V2.tiers["local-web"].policy is None


def test_a_tier_stays_hashable_with_a_policy_as_in_1_22() -> None:
    local = V2.tiers["local"]
    assert local.policy is not None
    assert hash(local) == hash(roles.load_tiers(V2_PATH).tiers["local"]), "equal tiers hash equal"
    assert len({local, V2.tiers["cloud"], local}) == 2
    raw = _with(lambda r: r["policies"]["local"].update(codes=[429, 503], tags={"a": ["x"]}))
    tier = roles.load_tiers(raw).tiers["local"]
    hash(tier)
    assert isinstance(tier.policy, Mapping) and tier.policy["codes"] == (429, 503), "an array reads as a tuple"


def _with(edit: Any) -> dict[str, Any]:
    raw = _raw()
    edit(raw)
    return raw


# -- spec test 2: every TableInvalid names its path ----------------------------------------------
@pytest.mark.parametrize(
    ("edit", "path", "match"),
    [
        (lambda r: r["providers"]["gateway"].pop("kind"), "providers.gateway.kind", "missing"),
        (lambda r: r["providers"]["gateway"].update(kind=""), "providers.gateway.kind", "non-empty string"),
        (
            lambda r: r["providers"]["gateway"].update(base_url="http://x:4000"),
            "providers.gateway.base_url",
            "holds a URL",
        ),
        (lambda r: r["providers"].update(bad=[1]), "providers.bad", "must be a table"),
        (lambda r: r["models"]["chat"].update(colour="red"), "models.chat.colour", "unknown key"),
        (lambda r: r["models"]["chat"].pop("id"), "models.chat.id", "missing"),
        (lambda r: r["models"]["chat"].pop("provider"), "models.chat.provider", "missing"),
        (lambda r: r["models"]["chat"].update(provider="nope"), "models.chat.provider", "unknown provider 'nope'"),
        (
            lambda r: r["models"]["chat"].update(supports=["vision", "telepathy"]),
            "models.chat.supports[1]",
            "unknown capability",
        ),
        (lambda r: r["models"]["chat"].update(supports="vision"), "models.chat.supports", "list of capability words"),
        (lambda r: r["models"]["chat"].update(thinking="loud"), "models.chat.thinking", "unknown style"),
        (lambda r: r["models"]["chat"].update(self_hosted="yes"), "models.chat.self_hosted", "true or false"),
        (lambda r: r["models"]["chat"].update(context_window=0), "models.chat.context_window", "integer >= 1"),
        (lambda r: r["models"]["chat"].update(slots=True), "models.chat.slots", "integer >= 1"),
        (lambda r: r["models"]["chat"].update(usd_per_request="free"), "models.chat.usd_per_request", "decimal string"),
        (lambda r: r["models"]["chat"].update(usd_per_request=[1]), "models.chat.usd_per_request", "decimal string"),
        (lambda r: r["models"]["chat"].update(usd_per_request="-1"), "models.chat.usd_per_request", ">= 0"),
        (lambda r: r["models"]["chat"].update(usd_per_request="NaN"), "models.chat.usd_per_request", ">= 0"),
        (lambda r: r["models"]["chat"].update(group=""), "models.chat.group", "non-empty string"),
        (lambda r: r["models"].update(bad="small"), "models.bad", "must be a table"),
        (lambda r: r["policies"].update(bad=1), "policies.bad", "must be a table"),
        (lambda r: r["tiers"].update(bad="x"), "tiers.bad", "must be a table"),
        (lambda r: r["tiers"]["local"].update(model="nope"), "tiers.local.model", "unknown model key 'nope'"),
        (lambda r: r["tiers"]["local"].update(timeout="nope"), "tiers.local.timeout", "unknown rung 'nope'"),
        (lambda r: r["tiers"]["local"].update(policy="nope"), "tiers.local.policy", "unknown policy 'nope'"),
        (lambda r: r["tiers"]["local"].update(fallback=["nope"]), "tiers.local.fallback[0]", "unknown tier 'nope'"),
        (lambda r: r["tiers"]["local"].update(fallback=["local"]), "tiers.local.fallback[0]", "itself"),
        (lambda r: r["tiers"]["local"].update(fallback="cloud"), "tiers.local.fallback", "list of names"),
        (
            lambda r: r["tiers"]["local"].update(fallback_on=["meltdown"]),
            "tiers.local.fallback_on[0]",
            "unknown failure kind",
        ),
        (lambda r: r["tiers"]["local"].update(self_hosted=True), "tiers.local.self_hosted", "set on \\[models"),
        (lambda r: r["tiers"]["local"].update(box=True), "tiers.local.box", "set on \\[models"),
        (lambda r: r["tiers"]["local"].update(extra=1), "tiers.local.extra", "unknown key"),
        (
            lambda r: r["tiers"]["cloud"].update(fallback=["local"]),
            "tiers.cloud.fallback",
            "fallback cycle: local -> cloud -> local",
        ),
    ],
)
def test_2_every_table_invalid_names_its_path(edit: Any, path: str, match: str) -> None:
    _invalid(_with(edit), path, match)


def test_a_three_tier_cycle_names_the_loop() -> None:
    def three(r: dict[str, Any]) -> None:
        r["tiers"]["local-web"]["fallback"] = ["local"]
        r["tiers"]["cloud"]["fallback"] = ["local-web"]

    err = _invalid(_with(three), "tiers.local-web.fallback")
    assert "local -> cloud -> local-web -> local" in str(err)


def test_table_invalid_is_a_value_error_so_1_22_callers_still_catch_it() -> None:
    assert issubclass(TableInvalid, ValueError)


# -- the fingerprint --------------------------------------------------------------------------
def test_the_fingerprint_is_stable_order_free_and_moves_with_any_value() -> None:
    fp = V2.fingerprint()
    assert len(fp) == 16 and int(fp, 16) >= 0
    assert all(t.table_fingerprint == fp for t in V2.tiers.values())
    raw = _raw()
    reordered = {k: raw[k] for k in reversed(list(raw))}
    reordered["models"] = {k: raw["models"][k] for k in reversed(list(raw["models"]))}
    assert roles.load_tiers(reordered).fingerprint() == fp, "key order is not a change"
    assert roles.load_tiers(_with(lambda r: r["models"]["chat"].update(slots=2))).fingerprint() != fp
    assert roles.load_tiers(_with(lambda r: r["tiers"]["local"].update(fallback_on=["server"]))).fingerprint() != fp
    assert TABLE.fingerprint() != fp


def test_a_policy_value_with_no_canonical_form_is_refused_by_the_fingerprint() -> None:
    raw = _with(lambda r: r["policies"]["local"].update(odd=object()))
    with pytest.raises(TypeError, match="no canonical JSON form for object"):
        roles.load_tiers(raw)


# -- require: a role's capabilities, on its tier and every fallback ------------------------------
class V2Role(roles.Role):
    TIERS: ClassVar = V2.tiers


def test_require_walks_the_tier_and_its_fallbacks() -> None:
    V2.require(V2Role("t.vision", tier="local"), "vision")
    with pytest.raises(TableInvalid, match=r"role 't\.thinks' needs \['thinking'\]") as caught:
        V2.require(V2Role("t.thinks", tier="local"), "thinking")
    assert caught.value.path == "tiers.local.fallback (cloud)", "the chat model thinks; its fallback does not"
    with pytest.raises(TableInvalid) as caught:
        V2.require(V2Role("t.web", tier="local-web"), "vision")
    assert caught.value.path == "tiers.local-web"
    other = roles.load_tiers(_with(lambda r: r["tiers"].pop("local-web")))

    class OtherRole(roles.Role):
        TIERS: ClassVar = V2.tiers

    with pytest.raises(TableInvalid, match="does not have"):
        other.require(OtherRole("t.web", tier="local-web"))


def test_require_on_a_schema_1_table_checks_only_the_tier() -> None:
    from tests.conftest import Role

    TABLE.require(Role("t.any", tier="local"), "vision", "web")


def test_the_fallback_chain_visits_each_tier_once() -> None:
    def diamond(r: dict[str, Any]) -> None:
        r["tiers"]["local"]["fallback"] = ["local-web", "cloud"]
        r["tiers"]["local-web"]["fallback"] = ["cloud"]

    table = roles.load_tiers(_with(diamond))
    assert roles._fallback_chain(table.tiers, "local") == ["local-web", "cloud"]


# -- the thinking switch is provider-neutral on schema 2 ----------------------------------------
def test_a_schema_2_role_sends_thinking_as_a_bool_and_no_extra_body() -> None:
    settings = V2Role("t.chat", tier="local", thinking=True).model_settings()
    assert settings["thinking"] is True and "extra_body" not in settings
    assert V2Role("t.cloud", tier="cloud").model_settings()["thinking"] is False


def test_the_capability_words_are_closed_and_json_object_stands_in_for_json_schema() -> None:
    assert models.needs(structured=True, images=True, web=True) == frozenset({"json_schema", "vision", "web"})
    assert models.needs(structured=False, images=False, web=False) == frozenset()
    web = V2.model_info["web"]
    assert models.missing(web, frozenset({"json_schema", "web", "vision"})) == ("vision",)
    assert models.missing(V2.model_info["embed"], frozenset({"json_schema"})) == ("json_schema",)
    assert set(models.CAPABILITIES) >= {"tools", "vision", "embed", "rerank"} and len(models.FAILURE_KINDS) == 16
    assert models.DEFAULT_FALLBACK_ON <= set(models.FAILURE_KINDS)


def test_the_provider_section_keeps_extra_keys_as_options() -> None:
    parsed = models.parse_providers({"p": {"kind": "k", "gateway": "litellm", "verify": False}})
    assert parsed["p"].options == {"verify": False} and parsed["p"].gateway == "litellm"
    assert parsed["p"].base_url is None and parsed["p"].secret is None


def test_a_model_info_serialises_with_its_fields() -> None:
    assert asdict(V2.model_info["embed"])["supports"] == frozenset({"embed"})


def test_a_table_loads_from_the_path_as_from_its_mapping() -> None:
    assert roles.load_tiers(_raw()).fingerprint() == V2.fingerprint()
    assert isinstance(V2_PATH, Path)
