"""A consumer, as the tests see one: its own tier table, bound to its own Role subclass."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_ports.call import _generate, roles

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PACKS = FIXTURES / "packs"
TABLE = roles.load_tiers(FIXTURES / "llm_tiers.toml")


class Role(roles.Role):
    TIERS = TABLE.tiers


@pytest.fixture(autouse=True)
def _schema_1_warning_already_seen() -> Iterator[None]:
    """The once-per-process schema 1 `llm_call_warning` would land on whichever test calls first.

    Mark it seen, so every test's event list is the same whatever the order; the test of the
    warning itself clears the set.
    """
    _generate._WARNED.add("capabilities")
    yield
    _generate._WARNED.clear()
