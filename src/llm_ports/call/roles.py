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

Example:
    TABLE = load_tiers(Path(__file__).with_name("llm_tiers.toml"))

    class Role(roles.Role):
        TIERS = TABLE.tiers

    CATEGORIZE = Role("categorize.batch", tier="gateway", temperature=0.0)

Invariants:
    - A `Role` whose tier is not in its class's table raises at construction, at import time,
      never as a call that resolved to nothing.
    - A tier's `box` key reads as `self_hosted`, with a `DeprecationWarning`, through 1.x.
"""

from __future__ import annotations

import tomllib
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class Tier:
    """One resolved tier: a model, a client timeout, and whether the model is self-hosted.

    Attributes:
        name: The tier's key in the table.
        model: The model id its `[models]` key names.
        timeout_s: The client timeout its `[ladder]` rung names, in seconds.
        self_hosted: True when the model group is self-hosted.
    """

    name: str
    model: str
    timeout_s: float
    self_hosted: bool = True


@dataclass(frozen=True)
class TierTable:
    """A parsed tier table.

    Attributes:
        ladder: Rung name to timeout, in seconds.
        models: Model key to model id.
        tiers: Tier name to resolved `Tier`.
    """

    ladder: Mapping[str, float]
    models: Mapping[str, str]
    tiers: Mapping[str, Tier]


LadderCheck = Callable[[Mapping[str, float], Path], None]

#: What `check_ladder` and error messages name as the source of a table passed as a mapping.
MAPPING_SOURCE = Path("<mapping>")
#: The tier key `self_hosted` replaced; read as an alias through 1.x.
_DEPRECATED_SELF_HOSTED_KEY = "box"


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
            sets both `self_hosted` and its deprecated alias `box`.
    """
    raw: Mapping[str, Any]
    if isinstance(source, Path):
        path, raw = source, tomllib.loads(source.read_text(encoding="utf-8"))
    else:
        path, raw = MAPPING_SOURCE, source
    ladder = {k: float(v) for k, v in raw["ladder"].items()}
    models = {k: str(v) for k, v in raw["models"].items()}
    if check_ladder is not None:
        check_ladder(ladder, path)
    tiers: dict[str, Tier] = {}
    for name, t in raw["tiers"].items():
        if t.get("model") not in models or t.get("timeout") not in ladder:
            raise ValueError(
                f"{path.name}: tier {name!r} names model {t.get('model')!r} / rung "
                f"{t.get('timeout')!r}; have models {sorted(models)}, rungs {sorted(ladder)}"
            )
        tiers[name] = Tier(name, models[t["model"]], ladder[t["timeout"]], _self_hosted(t, name, path))
    return TierTable(ladder=ladder, models=models, tiers=tiers)


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
        stacklevel=3,
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
        """Return the pydantic-ai `ModelSettings` this role sends.

        A self-hosted tier (`Tier.self_hosted`) also carries the thinking switch,
        `extra_body.chat_template_kwargs.enable_thinking`, which llama.cpp and vLLM read.
        """
        out: dict[str, Any] = dict(self.settings)
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.resolved_tier.self_hosted:
            extra = dict(out.get("extra_body") or {})
            extra["chat_template_kwargs"] = {"enable_thinking": self.thinking}
            out["extra_body"] = extra
        return out


__all__ = ["MAPPING_SOURCE", "LadderCheck", "Role", "Tier", "TierTable", "load_tiers"]
