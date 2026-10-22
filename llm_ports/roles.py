"""roles -- the ONE place a model call's knobs are declared.

A `Role` names one generation surface (`feature.action`, the same string a category switchboard,
a trace and a work row group by) and resolves it to a `Tier` -- model group, self-hosted-or-cloud,
and the client timeout rung -- plus the per-role budget and sampling. Feature code declares a role
and calls `llm_ports.call.generate`; it never carries a model literal, a timeout literal or a
framework import.

THE TIER TABLE IS DATA, and it is the CONSUMER's: a TOML file (`[ladder]` rungs in seconds,
`[models]` groups, `[tiers.<name>]` = a model key + a rung + `box`), parsed and CHECKED by
`load_tiers`. A consumer binds its table to its roles ONCE, by subclass, so two consumers never
share a process-global table:

    TABLE = load_tiers(Path(__file__).with_name("llm_tiers.toml"))

    class Role(roles.Role):
        TIERS = TABLE.tiers

    CATEGORIZE = Role("categorize.batch", tier="gateway", temperature=0.0)

A `Role` whose tier is not in its class's table raises at construction -- at import, never as a
call that resolved to nothing.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    timeout_s: float
    box: bool = True


@dataclass(frozen=True)
class TierTable:
    """A parsed tier table: the ladder rungs (seconds), the model groups, the tiers."""

    ladder: Mapping[str, float]
    models: Mapping[str, str]
    tiers: Mapping[str, Tier]


LadderCheck = Callable[[Mapping[str, float], Path], None]


def load_tiers(path: Path, *, check_ladder: LadderCheck | None = None) -> TierTable:
    """Parse and CHECK a tier table. A tier names a `[models]` key and a `[ladder]` rung; an unknown
    one raises HERE. `check_ladder(ladder, path)` is the consumer's own ordering law over the rungs
    (e.g. a client timeout that must clear a server deadline); it runs before the tiers resolve and
    raises ValueError to refuse the table."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    ladder = {k: float(v) for k, v in raw["ladder"].items()}
    models = {k: str(v) for k, v in raw["models"].items()}
    if check_ladder is not None:
        check_ladder(ladder, path)
    tiers: dict[str, Tier] = {}
    for name, t in raw["tiers"].items():
        if t.get("model") not in models or t.get("timeout") not in ladder:
            raise ValueError(f"{path.name}: tier {name!r} names model {t.get('model')!r} / rung "
                             f"{t.get('timeout')!r}; have models {sorted(models)}, rungs {sorted(ladder)}")
        tiers[name] = Tier(name, models[t["model"]], ladder[t["timeout"]], bool(t.get("box", True)))
    return TierTable(ladder=ladder, models=models, tiers=tiers)


@dataclass(frozen=True)
class Role:
    """`name` = `feature.action`. `category` defaults to `name`. `thinking=False` is the self-hosted
    default: on a hybrid thinker the grammar binds only `content` and reasoning otherwise eats the
    budget. `settings` carries per-role sampling the tier does not decide (seed, top_p, temperature).
    `graders` names `call_graders` entries `call.generate` scores every `ok` reply with.

    `TIERS` is the CLASS's tier table -- empty here, so the base class cannot be instantiated; a
    consumer subclasses with its own (module docstring)."""

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

    def __post_init__(self) -> None:
        if self.tier not in self.TIERS:
            raise ValueError(f"role {self.name!r}: unknown tier {self.tier!r} (have {sorted(self.TIERS)})")
        if not self.category:
            object.__setattr__(self, "category", self.name)

    @property
    def resolved_tier(self) -> Tier:
        return self.TIERS[self.tier]

    @property
    def resolved_model(self) -> str:
        return self.model or self.resolved_tier.model

    @property
    def resolved_timeout_s(self) -> float:
        return self.timeout_s if self.timeout_s is not None else self.resolved_tier.timeout_s

    def model_settings(self) -> dict[str, Any]:
        """The pydantic-ai ModelSettings this role sends. Self-hosted (`box`) tiers carry the
        thinking switch (`chat_template_kwargs.enable_thinking`, llama.cpp / vLLM)."""
        out: dict[str, Any] = dict(self.settings)
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.resolved_tier.box:
            extra = dict(out.get("extra_body") or {})
            extra["chat_template_kwargs"] = {"enable_thinking": self.thinking}
            out["extra_body"] = extra
        return out


__all__ = ["LadderCheck", "Role", "Tier", "TierTable", "load_tiers"]
