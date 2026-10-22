"""A consumer, as the tests see one: its own tier table, bound to its own Role subclass."""

from __future__ import annotations

from pathlib import Path

from llm_ports import roles

HERE = Path(__file__).resolve().parent
PACKS = HERE / "packs"
TABLE = roles.load_tiers(HERE / "llm_tiers.toml")


class Role(roles.Role):
    TIERS = TABLE.tiers
