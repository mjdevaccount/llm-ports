"""The model table's value types: what a model can do, who serves it, and the closed vocabularies.

A `schema_version = 2` tier table (`call.roles.load_tiers`) adds `[providers.<name>]` and
`[models.<key>]` sections. A model key names one wire id, the provider that serves it, and what it
can do, stated by the consumer in a closed vocabulary (`CAPABILITIES`): nothing is inferred from a
model's name and nothing comes from a bundled database. A provider names a registered kind (the
factory that builds its client) and, by NAME only, the URL key and the secret it needs: the table
is reviewed and fingerprinted, so it never holds a URL or a key.

Example:
    schema_version = 2

    [providers.gateway]
    kind = "openai_compatible"
    base_url = "gateway"            # a key the caller resolves to a URL
    secret = "gateway_key"          # a name the caller resolves to a key

    [models.chat]
    id = "qwen3.5-9b"
    provider = "gateway"
    supports = ["json_schema", "vision"]
    thinking = "chat_template"

Invariants:
    - A capability word, thinking style or failure kind outside its vocabulary is `TableInvalid`
      at load, naming the path (`models.chat.supports[1]`).
    - An unknown limit is None, never a guessed number.
    - Pure: no I/O, no vendor import.

Rationale: ADR-0022.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, get_args

#: What a model can be asked to do. The consumer's table states it per model; a request that needs a
#: capability its model lacks is refused before the wire.
Capability = Literal[
    "tools",
    "json_schema",
    "json_object",
    "vision",
    "thinking",
    "cache",
    "web",
    "stream",
    "count_tokens",
    "embed",
    "rerank",
]
#: Every capability word, in declaration order.
CAPABILITIES: tuple[Capability, ...] = get_args(Capability)
#: How a model's thinking switch is spelled: `none` (it has none), `unified` (the framework's own
#: `thinking` setting), `chat_template` (`extra_body.chat_template_kwargs.enable_thinking`, which
#: llama.cpp and vLLM read).
ThinkingStyle = Literal["none", "unified", "chat_template"]
#: Every thinking style.
THINKING_STYLES: tuple[ThinkingStyle, ...] = get_args(ThinkingStyle)
#: The failure kinds a tier's `fallback_on` may name. The failures row of the work list types these
#: as `ports.FailureKind`; the words are the same.
FAILURE_KINDS: tuple[str, ...] = (
    "gated",
    "rate_limited",
    "quota",
    "overloaded",
    "server",
    "timeout",
    "transport",
    "auth",
    "not_found",
    "bad_request",
    "context_overflow",
    "unsupported",
    "internal",
    "truncated",
    "invalid_output",
    "refused",
)
#: The kinds a tier fails over on when its `fallback_on` is not set: the backend is busy or broken.
#: `quota`, `auth` and every request-side kind never fail over by default.
DEFAULT_FALLBACK_ON: frozenset[str] = frozenset({"overloaded", "server", "transport", "timeout"})

_PROVIDER_KEYS = frozenset({"kind", "base_url", "secret", "gateway"})
_MODEL_KEYS = frozenset(
    {
        "id",
        "provider",
        "context_window",
        "max_output",
        "supports",
        "thinking",
        "slots",
        "group",
        "self_hosted",
        "usd_per_request",
    }
)


class TableInvalid(ValueError):
    """A tier table that cannot be used, raised at load or at `build`, naming where.

    Attributes:
        path: The dotted path of the offending value (`tiers.box.fallback[0]`).
    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path: str = path


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One model of a schema 2 table: its wire id, its provider, and what it can do.

    Attributes:
        key: The model's key in `[models]`.
        id: The id sent on the wire (a gateway route is fine). The only place it is written.
        provider: The key of the `[providers]` entry that serves it.
        supports: The capabilities the consumer states for it.
        context_window: Tokens in its context window; None when unknown.
        max_output: The most output tokens it returns; None when unknown.
        thinking: How its thinking switch is spelled.
        slots: Calls it takes at once in-process; None when not limited here.
        group: Models sharing one backend resource; None when it shares none.
        self_hosted: True when the consumer runs the model itself.
        usd_per_request: A flat price per request; None when the cost is unknown.
    """

    key: str
    id: str
    provider: str
    supports: frozenset[Capability]
    context_window: int | None = None
    max_output: int | None = None
    thinking: ThinkingStyle = "none"
    slots: int | None = None
    group: str | None = None
    self_hosted: bool = True
    usd_per_request: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One provider of a schema 2 table: a registered kind, and the names of what it needs.

    Attributes:
        name: The provider's key in `[providers]`.
        kind: The registered provider kind that builds its client (`openai_compatible`).
        base_url: A topology key the caller resolves to a URL; None when the kind needs none.
        secret: A secret name the caller resolves to a key; None for a keyless endpoint.
        gateway: Which gateway's response headers the provider reads, when one stands in front.
        options: Every other key of the section, validated by the kind's factory.
    """

    name: str
    kind: str
    base_url: str | None
    secret: str | None
    gateway: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TableInvalid(path, f"must be a non-empty string, not {value!r}")
    return value


def _opt_str(section: Mapping[str, Any], key: str, path: str) -> str | None:
    return None if key not in section else _str(section[key], f"{path}.{key}")


def _opt_int(section: Mapping[str, Any], key: str, path: str) -> int | None:
    if key not in section:
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TableInvalid(f"{path}.{key}", f"must be an integer >= 1, not {value!r}")
    return value


def _section(raw: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TableInvalid(path, f"must be a table, not {type(raw).__name__}")
    return raw


def _unknown_keys(section: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    extra = sorted(set(section) - allowed)
    if extra:
        raise TableInvalid(f"{path}.{extra[0]}", f"unknown key (allowed: {sorted(allowed)})")


def parse_providers(raw: Mapping[str, Any]) -> dict[str, ProviderSpec]:
    """Parse a `[providers]` section.

    Args:
        raw: Provider name to its section.

    Returns:
        Provider name to `ProviderSpec`, in the section's order.

    Raises:
        TableInvalid: A section is not a table, `kind` is missing, a value has the wrong type, or
            `base_url` holds a URL rather than a key.
    """
    out: dict[str, ProviderSpec] = {}
    for name, value in raw.items():
        path = f"providers.{name}"
        section = _section(value, path)
        if "kind" not in section:
            raise TableInvalid(f"{path}.kind", "missing: name a registered provider kind")
        base_url = _opt_str(section, "base_url", path)
        if base_url is not None and "://" in base_url:
            raise TableInvalid(f"{path}.base_url", "holds a URL; the table names a key the caller resolves")
        out[name] = ProviderSpec(
            name=name,
            kind=_str(section["kind"], f"{path}.kind"),
            base_url=base_url,
            secret=_opt_str(section, "secret", path),
            gateway=_opt_str(section, "gateway", path),
            options={k: v for k, v in section.items() if k not in _PROVIDER_KEYS},
        )
    return out


def _supports(value: Any, path: str) -> frozenset[Capability]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TableInvalid(path, f"must be a list of capability words, not {value!r}")
    words: set[Capability] = set()
    for i, word in enumerate(value):
        if word not in CAPABILITIES:
            raise TableInvalid(f"{path}[{i}]", f"unknown capability {word!r} (known: {list(CAPABILITIES)})")
        words.add(word)
    return frozenset(words)


def _price(section: Mapping[str, Any], path: str) -> Decimal | None:
    if "usd_per_request" not in section:
        return None
    value = section["usd_per_request"]
    try:
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise InvalidOperation
        price = Decimal(str(value))
    except InvalidOperation:
        raise TableInvalid(f"{path}.usd_per_request", f"must be a decimal string, not {value!r}") from None
    if not price.is_finite() or price < 0:
        raise TableInvalid(f"{path}.usd_per_request", f"must be a finite amount >= 0, not {value!r}")
    return price


def parse_models(raw: Mapping[str, Any], providers: Mapping[str, ProviderSpec]) -> dict[str, ModelInfo]:
    """Parse a schema 2 `[models]` section against its providers.

    Args:
        raw: Model key to its section.
        providers: The parsed `[providers]` section.

    Returns:
        Model key to `ModelInfo`, in the section's order.

    Raises:
        TableInvalid: An unknown key, a missing `id` or `provider`, a provider the table does not
            declare, a word outside its vocabulary, or a value of the wrong type.
    """
    out: dict[str, ModelInfo] = {}
    for key, value in raw.items():
        path = f"models.{key}"
        section = _section(value, path)
        _unknown_keys(section, _MODEL_KEYS, path)
        for required in ("id", "provider"):
            if required not in section:
                raise TableInvalid(f"{path}.{required}", "missing")
        provider = _str(section["provider"], f"{path}.provider")
        if provider not in providers:
            raise TableInvalid(f"{path}.provider", f"unknown provider {provider!r} (have {sorted(providers)})")
        thinking = section.get("thinking", "none")
        if thinking not in THINKING_STYLES:
            raise TableInvalid(f"{path}.thinking", f"unknown style {thinking!r} (known: {list(THINKING_STYLES)})")
        self_hosted = section.get("self_hosted", True)
        if not isinstance(self_hosted, bool):
            raise TableInvalid(f"{path}.self_hosted", f"must be true or false, not {self_hosted!r}")
        out[key] = ModelInfo(
            key=key,
            id=_str(section["id"], f"{path}.id"),
            provider=provider,
            supports=_supports(section.get("supports", []), f"{path}.supports"),
            context_window=_opt_int(section, "context_window", path),
            max_output=_opt_int(section, "max_output", path),
            thinking=thinking,
            slots=_opt_int(section, "slots", path),
            group=_opt_str(section, "group", path),
            self_hosted=self_hosted,
            usd_per_request=_price(section, path),
        )
    return out


def needs(*, structured: bool, images: bool, web: bool) -> frozenset[Capability]:
    """Return the capabilities one request needs. Pure.

    Args:
        structured: The output is a schema, not prose.
        images: The request carries images.
        web: The model is asked to fetch pages.

    Returns:
        `vision` for images, `web` for a page fetch, and `json_schema` for structured output
        (`missing` accepts `json_object` in its place).
    """
    out: set[Capability] = set()
    if images:
        out.add("vision")
    if web:
        out.add("web")
    if structured:
        out.add("json_schema")
    return frozenset(out)


def missing(info: ModelInfo, wanted: frozenset[Capability]) -> tuple[Capability, ...]:
    """Return the capabilities in `wanted` that `info` lacks, in vocabulary order. Pure.

    `json_object` stands in for `json_schema`: either lets a model return structured output.
    """
    have = set(info.supports)
    if "json_object" in have:
        have.add("json_schema")
    return tuple(c for c in CAPABILITIES if c in wanted and c not in have)


__all__ = [
    "CAPABILITIES",
    "DEFAULT_FALLBACK_ON",
    "FAILURE_KINDS",
    "THINKING_STYLES",
    "Capability",
    "ModelInfo",
    "ProviderSpec",
    "TableInvalid",
    "ThinkingStyle",
    "missing",
    "needs",
    "parse_models",
    "parse_providers",
]
