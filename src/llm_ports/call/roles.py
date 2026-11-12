"""Roles: the one place a model call's settings are declared.

A `Role` names one generation surface (`feature.action`, the string a category switchboard, a
trace and a work row group by) and resolves it to a `Tier`: a model group, whether that group is
self-hosted, and a client timeout rung. The role adds the per-role budget and sampling. Feature
code declares a role and calls `llm_ports.call.generate`; it carries no model literal, no timeout
literal and no framework import.

The tier table is data, and it belongs to the consumer: a TOML file (`[ladder]` rungs in seconds,
`[models]` groups, `[tiers.<name>]` naming a model key, a rung and `self_hosted`), parsed and
checked by `load_tiers`. The same table can arrive as a mapping (parsed TOML, a dict in a test).
A consumer binds its table to its roles once, by subclass, so two consumers never share a
process-global table.

A `schema_version = 2` table adds providers by name, what each model can do, wire policies and
fallback order (`call.models`); a table without `schema_version` is version 1 and loads as it
always did.

Example:
    TABLE = load_tiers(Path(__file__).with_name("llm_tiers.toml"))

    class Role(roles.Role):
        TIERS = TABLE.tiers

    CATEGORIZE = Role("categorize.batch", tier="gateway", temperature=0.0)

Invariants:
    - A `Role` whose tier is not in its class's table raises at construction, at import time,
      never as a call that resolved to nothing.
    - A tier's `box` key reads as `self_hosted`, with a `DeprecationWarning`, through 1.x.
    - A schema 2 table that names an unknown model, rung, provider, policy, tier, capability or
      failure kind, or whose fallbacks form a cycle, is `TableInvalid` naming the path.
    - A model id is written once: a schema 2 tier names a model key, and `[models.<key>] id` is
      the only place the wire string lives.

Rationale: ADR-0022.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

from llm_ports.call._frozen import freeze
from llm_ports.call.context import ContextMode, TrimPolicy
from llm_ports.call.models import (
    DEFAULT_FALLBACK_ON,
    FAILURE_KINDS,
    Capability,
    ModelInfo,
    ProviderSpec,
    TableInvalid,
    parse_models,
    parse_providers,
)
from llm_ports.observability import log_events
from llm_ports.ports import WirePolicy

#: The table versions `load_tiers` reads.
SCHEMA_VERSIONS: tuple[int, ...] = (1, 2)
#: Where table events go (`load_tiers` takes no hook: its signature is 1.x's).
_log_event = log_events("llm_ports.call")


@dataclass(frozen=True)
class Tier:
    """One resolved tier: a model, a client timeout, and whether the model is self-hosted.

    Attributes:
        name: The tier's key in the table.
        model: The model id its `[models]` key names.
        timeout_s: The client timeout its `[ladder]` rung names, in seconds.
        self_hosted: True when the model group is self-hosted.
        info: What the model is and can do; None for a schema 1 table, where nothing is checked.
        policy: The tier's wire policy as declared (`[policies.<name>]`, read-only and hashable:
            nested tables read as mappings, arrays as tuples); None for none. `wire_policy` is
            the same policy as a `WirePolicy`.
        fallback: Tier names tried in order when this tier fails with a kind in `fallback_on`.
        fallback_on: The failure kinds that move a call to the next fallback tier.
        table_fingerprint: The fingerprint of the table the tier was loaded from; None for a tier
            built by hand. Not compared.
    """

    name: str
    model: str
    timeout_s: float
    self_hosted: bool = True
    info: ModelInfo | None = None
    policy: object | None = None
    fallback: tuple[str, ...] = ()
    fallback_on: frozenset[str] = DEFAULT_FALLBACK_ON
    table_fingerprint: str | None = field(default=None, compare=False)

    @property
    def wire_policy(self) -> WirePolicy | None:
        """The declared `policy` as a `WirePolicy`; None when the tier declares none.

        Raises:
            ValueError: The declared policy is not a valid `WirePolicy` (`load_tiers` refuses such
                a table at load, so only a hand-built tier can raise here).
        """
        return None if self.policy is None else WirePolicy.from_mapping(_mapping(self.policy))


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"a policy is a table of rules, not {type(value).__name__}")
    return value


@dataclass(frozen=True)
class TierTable:
    """A parsed tier table.

    Attributes:
        ladder: Rung name to timeout, in seconds.
        models: Model key to model id.
        tiers: Tier name to resolved `Tier`.
        model_info: Model key to `ModelInfo`; empty for a schema 1 table.
        providers: Provider name to `ProviderSpec`; empty for a schema 1 table.
        policies: Policy name to its declared section, read-only; empty for a schema 1 table.
        schema_version: 1 or 2.
    """

    ladder: Mapping[str, float]
    models: Mapping[str, str]
    tiers: Mapping[str, Tier]
    model_info: Mapping[str, ModelInfo] = field(default_factory=dict)
    providers: Mapping[str, ProviderSpec] = field(default_factory=dict)
    policies: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    schema_version: int = 1

    def fingerprint(self) -> str:
        """Return the first 16 hex digits of the sha256 of the parsed table's canonical JSON.

        Two tables that parse to the same values have the same fingerprint, whatever their key
        order or source format.

        Raises:
            TypeError: A policy holds a value with no canonical JSON form.
        """
        body = {
            "schema_version": self.schema_version,
            "ladder": self.ladder,
            "models": self.model_info or self.models,
            "providers": self.providers,
            "policies": self.policies,
            "tiers": {
                n: {
                    "model": t.model,
                    "timeout_s": t.timeout_s,
                    "self_hosted": t.self_hosted,
                    "policy": t.policy,
                    "fallback": t.fallback,
                    "fallback_on": t.fallback_on if t.info is not None else None,
                }
                for n, t in self.tiers.items()
            },
        }
        text = json.dumps(_canon(body), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]

    def require(self, role: Role, *caps: Capability) -> None:
        """Prove that `role`'s tier, and every tier it can fall back to, has each of `caps`.

        Called at import time, once per role that needs a capability (`vision` for a role sent
        images), so a table edit that removes it fails the boot rather than a call. A schema 1
        table states no capability and is not checked.

        Raises:
            TableInvalid: The role's tier is not in this table, or a tier on its fallback path
                lacks a capability; the path names the tier.
        """
        if role.tier not in self.tiers:
            raise TableInvalid(f"tiers.{role.tier}", f"role {role.name!r} names a tier this table does not have")
        if self.schema_version < 2:
            return
        path = f"tiers.{role.tier}"
        for name in (role.tier, *_fallback_chain(self.tiers, role.tier)):
            info = self.tiers[name].info
            assert info is not None  # a schema 2 tier always carries its model
            lacks = [c for c in caps if c not in info.supports]
            if lacks:
                where = path if name == role.tier else f"{path}.fallback ({name})"
                raise TableInvalid(
                    where, f"role {role.name!r} needs {lacks}; model {info.key!r} supports {sorted(info.supports)}"
                )


def _canon(value: Any) -> Any:
    """Return `value` as plain JSON types, refusing a type with no canonical form."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, ModelInfo | ProviderSpec):
        return {f: _canon(getattr(value, f)) for f in value.__slots__}
    if isinstance(value, Mapping):
        return {str(k): _canon(v) for k, v in value.items()}
    if isinstance(value, frozenset | set):
        return sorted(_canon(v) for v in value)
    if isinstance(value, list | tuple):
        return [_canon(v) for v in value]
    raise TypeError(f"no canonical JSON form for {type(value).__name__}: {value!r}")


def _fallback_chain(tiers: Mapping[str, Tier], start: str) -> list[str]:
    """Every tier reachable from `start` through `fallback`, depth first, each once."""
    seen: list[str] = []
    stack = list(reversed(tiers[start].fallback))
    while stack:
        name = stack.pop()
        if name not in seen and name != start:
            seen.append(name)
            stack.extend(reversed(tiers[name].fallback))
    return seen


LadderCheck = Callable[[Mapping[str, float], Path], None]

#: What `check_ladder` and error messages name as the source of a table passed as a mapping.
MAPPING_SOURCE = Path("<mapping>")
#: The tier key `self_hosted` replaced; read as an alias through 1.x.
_DEPRECATED_SELF_HOSTED_KEY = "box"
#: Sections only a schema 2 table reads; a schema 1 table ignores them.
_V2_SECTIONS = ("providers", "policies")
_V2_TIER_KEYS = frozenset({"model", "timeout", "policy", "fallback", "fallback_on"})


def load_tiers(source: Path | Mapping[str, Any], *, check_ladder: LadderCheck | None = None) -> TierTable:
    """Parse and check a tier table.

    Args:
        source: The TOML file, or the table itself as a mapping with the same keys.
        check_ladder: The consumer's own ordering rule over the rungs (e.g. a client timeout that
            must clear a server deadline). Called as `check_ladder(ladder, path)` before the tiers
            resolve, with `MAPPING_SOURCE` as the path of a mapping; it raises `ValueError` to
            refuse the table.

    Raises:
        ValueError: A tier names a `[models]` key or a `[ladder]` rung the table does not have, or
            sets both `self_hosted` and its deprecated alias `box`. Every refusal of a schema 2
            table is the subclass `call.models.TableInvalid`, naming the path.
    """
    raw: Mapping[str, Any]
    if isinstance(source, Path):
        path, raw = source, tomllib.loads(source.read_text(encoding="utf-8"))
    else:
        path, raw = MAPPING_SOURCE, source
    version = raw.get("schema_version", 1)
    if version not in SCHEMA_VERSIONS or isinstance(version, bool):
        raise TableInvalid(
            "schema_version", f"{path.name}: unknown version {version!r} (known: {list(SCHEMA_VERSIONS)})"
        )
    ladder = {k: float(v) for k, v in raw["ladder"].items()}
    if check_ladder is not None:
        check_ladder(ladder, path)
    table = _load_v2(raw, ladder, path) if version == 2 else _load_v1(raw, ladder, path)
    fp = table.fingerprint()
    tiers = {n: replace(t, table_fingerprint=fp) for n, t in table.tiers.items()}
    return TierTable(
        ladder=table.ladder,
        models=table.models,
        tiers=tiers,
        model_info=table.model_info,
        providers=table.providers,
        policies=table.policies,
        schema_version=table.schema_version,
    )


def _load_v1(raw: Mapping[str, Any], ladder: dict[str, float], path: Path) -> TierTable:
    """The 1.x table: `[models]` maps a key to an id string; nothing about the model is stated.

    A schema 2 section in it is ignored, as 1.22 ignored it, and named once per load as
    `llm_table_ignored` (WARNING): the table loads to the same values it always did.
    """
    ignored = [section for section in _V2_SECTIONS if section in raw]
    if ignored:
        _log_event(
            logging.WARNING,
            "llm_table_ignored",
            {
                "source": path.name,
                "sections": ignored,
                "detail": "a schema 1 table does not read these; set schema_version = 2 to use them",
            },
        )
    models = {k: str(v) for k, v in raw["models"].items()}
    tiers: dict[str, Tier] = {}
    for name, t in raw["tiers"].items():
        if t.get("model") not in models or t.get("timeout") not in ladder:
            raise ValueError(
                f"{path.name}: tier {name!r} names model {t.get('model')!r} / rung "
                f"{t.get('timeout')!r}; have models {sorted(models)}, rungs {sorted(ladder)}"
            )
        tiers[name] = Tier(name, models[t["model"]], ladder[t["timeout"]], _self_hosted(t, name, path))
    return TierTable(ladder=ladder, models=models, tiers=tiers)


def _load_v2(raw: Mapping[str, Any], ladder: dict[str, float], path: Path) -> TierTable:
    """The schema 2 table: providers, model facts, policies and fallbacks, each checked by path."""
    providers = parse_providers(raw.get("providers", {}))
    info = parse_models(raw["models"], providers)
    policies: dict[str, Mapping[str, Any]] = {}
    for name, section in raw.get("policies", {}).items():
        if not isinstance(section, Mapping):
            raise TableInvalid(f"policies.{name}", f"must be a table, not {type(section).__name__}")
        try:
            WirePolicy.from_mapping(section, path=f"policies.{name}")
        except ValueError as exc:
            where, _, why = str(exc).partition(": ")
            raise TableInvalid(where, why) from None
        policies[name] = freeze(section)
    tiers = {name: _tier_v2(name, t, info, ladder, policies) for name, t in raw["tiers"].items()}
    for name, tier in tiers.items():
        for i, target in enumerate(tier.fallback):
            if target not in tiers:
                raise TableInvalid(f"tiers.{name}.fallback[{i}]", f"unknown tier {target!r} (have {sorted(tiers)})")
    _refuse_cycles(tiers)
    return TierTable(
        ladder=ladder,
        models={k: m.id for k, m in info.items()},
        tiers=tiers,
        model_info=info,
        providers=providers,
        policies=policies,
        schema_version=2,
    )


def _tier_v2(
    name: str,
    t: Any,
    info: Mapping[str, ModelInfo],
    ladder: Mapping[str, float],
    policies: Mapping[str, Mapping[str, Any]],
) -> Tier:
    path = f"tiers.{name}"
    if not isinstance(t, Mapping):
        raise TableInvalid(path, f"must be a table, not {type(t).__name__}")
    extra = sorted(set(t) - _V2_TIER_KEYS)
    if extra:
        hint = "; self_hosted is set on [models.<key>] in schema 2" if extra[0] in ("self_hosted", "box") else ""
        raise TableInvalid(f"{path}.{extra[0]}", f"unknown key (allowed: {sorted(_V2_TIER_KEYS)}){hint}")
    model = t.get("model")
    if not isinstance(model, str) or model not in info:
        raise TableInvalid(f"{path}.model", f"unknown model key {model!r} (have {sorted(info)})")
    rung = t.get("timeout")
    if not isinstance(rung, str) or rung not in ladder:
        raise TableInvalid(f"{path}.timeout", f"unknown rung {rung!r} (have {sorted(ladder)})")
    policy = t.get("policy")
    if policy is not None and policy not in policies:
        raise TableInvalid(f"{path}.policy", f"unknown policy {policy!r} (have {sorted(policies)})")
    fallback = _names(t.get("fallback", []), f"{path}.fallback")
    if name in fallback:
        raise TableInvalid(f"{path}.fallback[{fallback.index(name)}]", "a tier cannot fall back to itself")
    kinds = DEFAULT_FALLBACK_ON
    if "fallback_on" in t:
        listed = _names(t["fallback_on"], f"{path}.fallback_on")
        for i, kind in enumerate(listed):
            if kind not in FAILURE_KINDS:
                raise TableInvalid(
                    f"{path}.fallback_on[{i}]", f"unknown failure kind {kind!r} (known: {list(FAILURE_KINDS)})"
                )
        kinds = frozenset(listed)
    m = info[model]
    return Tier(
        name,
        m.id,
        ladder[rung],
        m.self_hosted,
        info=m,
        policy=None if policy is None else policies[policy],
        fallback=fallback,
        fallback_on=kinds,
    )


def _names(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(isinstance(v, str) and v for v in value):
        raise TableInvalid(path, f"must be a list of names, not {value!r}")
    return tuple(value)


def _refuse_cycles(tiers: Mapping[str, Tier]) -> None:
    """Raise `TableInvalid` when following `fallback` from any tier can return to it."""
    done: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if name in trail:
            loop = " -> ".join((*trail[trail.index(name) :], name))
            raise TableInvalid(f"tiers.{trail[-1]}.fallback", f"fallback cycle: {loop}")
        if name in done:
            return
        for target in tiers[name].fallback:
            visit(target, (*trail, name))
        done.add(name)

    for name in tiers:
        visit(name, ())


def _self_hosted(tier: Mapping[str, Any], name: str, path: Path) -> bool:
    """Read a tier's `self_hosted` flag (default True), accepting the deprecated key `box`."""
    old = _DEPRECATED_SELF_HOSTED_KEY
    if old not in tier:
        return bool(tier.get("self_hosted", True))
    if "self_hosted" in tier:
        raise ValueError(f"{path.name}: tier {name!r} sets both self_hosted and its old name {old!r}")
    warnings.warn(
        f"{path.name}: tier {name!r} key {old!r} is deprecated; rename it to 'self_hosted'",
        DeprecationWarning,
        stacklevel=4,
    )
    return bool(tier[old])


@dataclass(frozen=True)
class Role:
    """One generation surface and the settings its calls use.

    Attributes:
        TIERS: The class's tier table. Empty on the base class, so the base class cannot be
            instantiated; a consumer subclasses it with its own table (see the module docstring).
        name: `feature.action`.
        tier: A key of `TIERS`.
        category: The category a switchboard and a trace group by; defaults to `name`.
        model: Overrides the tier's model.
        pack: The prompt pack id.
        max_tokens: The token ceiling for the reply.
        temperature: The sampling temperature.
        thinking: The thinking switch sent to a self-hosted tier; off by default. On a hybrid
            thinking model the grammar binds only the content, and reasoning otherwise spends the
            token budget.
        output_retries: Re-asks allowed when the validator finds problems.
        timeout_s: Overrides the tier's timeout, in seconds.
        graders: Names of `call.graders` entries that `call.generate` scores every `ok` reply with.
        settings: Per-role sampling the tier does not decide (seed, top_p, temperature).
        accept_truncated: A reply the token ceiling cut is `ok` (with a `CallWarning`) instead of
            `truncated`: the 1.x behaviour, for a role whose consumer reads a partial reply.
        policy: How the role's wire requests are retried; None takes the tier's declared policy,
            else the generator's default.
        stream_internally: The adapter requests a stream and assembles it, so a cancelled or
            timed-out call closes a connection the server can stop on. Off by default.
        max_steps: The most model requests in one `call.chat`, tool rounds included; None takes
            `ChatRequest`'s default (8).
        tool_output_chars: The one tool-output budget of the role's chat calls, in characters; None
            for no cut. A `Tool.max_output_chars` overrides it for that tool.
        context: Before a call is sent, `off` (the default, costs nothing), `check` (count it and
            refuse an overflow as `context_overflow` with `provider_code="preflight"`), or `trim`
            (cut the history by `trim` to fit, then refuse what still does not). ADR-0024.
        trim: The trimming policy for `context="trim"`; None is `TrimPolicy()`.
        cache_prefix: Ask for a prompt-cache breakpoint after the stable prefix: the system text
            (and, in a chat, the tool definitions). Dropped with a warning on a model that does not
            state `cache`.

    Raises:
        ValueError: `tier` is not in `TIERS`.
    """

    TIERS: ClassVar[Mapping[str, Tier]] = {}

    name: str
    tier: str
    category: str = ""
    model: str | None = None
    pack: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: bool = False
    output_retries: int = 1
    timeout_s: float | None = None
    graders: tuple[str, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)
    accept_truncated: bool = False
    policy: WirePolicy | None = None
    stream_internally: bool = False
    max_steps: int | None = None
    tool_output_chars: int | None = None
    context: ContextMode = "off"
    trim: TrimPolicy | None = None
    cache_prefix: bool = False

    def __post_init__(self) -> None:
        if self.tier not in self.TIERS:
            raise ValueError(f"role {self.name!r}: unknown tier {self.tier!r} (have {sorted(self.TIERS)})")
        if not self.category:
            object.__setattr__(self, "category", self.name)

    @property
    def resolved_tier(self) -> Tier:
        """The `Tier` this role names."""
        return self.TIERS[self.tier]

    @property
    def resolved_model(self) -> str:
        """The role's model override, else the tier's model."""
        return self.model or self.resolved_tier.model

    @property
    def resolved_timeout_s(self) -> float:
        """The role's timeout override, else the tier's timeout, in seconds."""
        return self.timeout_s if self.timeout_s is not None else self.resolved_tier.timeout_s

    def model_settings(self) -> dict[str, Any]:
        """Return the settings this role sends.

        On a schema 2 tier the thinking switch is provider-neutral, `thinking: bool`; the
        provider's factory spells it the way the model's `ModelInfo.thinking` says. On a schema 1
        tier a self-hosted model gets `extra_body.chat_template_kwargs.enable_thinking`, which
        llama.cpp and vLLM read, as in 1.22.
        """
        out: dict[str, Any] = dict(self.settings)
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            out["temperature"] = self.temperature
        tier = self.resolved_tier
        if tier.info is not None:
            out["thinking"] = self.thinking
        elif tier.self_hosted:
            extra = dict(out.get("extra_body") or {})
            extra["chat_template_kwargs"] = {"enable_thinking": self.thinking}
            out["extra_body"] = extra
        return out


__all__ = ["MAPPING_SOURCE", "SCHEMA_VERSIONS", "LadderCheck", "Role", "Tier", "TierTable", "load_tiers"]
