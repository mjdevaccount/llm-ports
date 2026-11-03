"""A consumer, as the tests see one: its own tier table, bound to its own Role subclass."""

from __future__ import annotations

from pathlib import Path

from llm_ports.call import roles

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PACKS = FIXTURES / "packs"
TABLE = roles.load_tiers(FIXTURES / "llm_tiers.toml")


class Role(roles.Role):
    TIERS = TABLE.tiers
