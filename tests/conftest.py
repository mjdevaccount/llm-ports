"""A consumer, as the tests see one: its own tier table, bound to its own Role subclass."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_ports.call import _generate, roles

try:
    from llm_ports.adapters import _retry
except ImportError:  # a core-only install has no wire-retry loop to put on a fake clock
    _retry = None  # type: ignore[assignment]

# `pytester` runs the library's own pytest plugin in an inner session (tests/testing/test_plugin.py).
# The plugin itself loads here, after coverage starts (pyproject's `-p no:llm_ports` says why).
pytest_plugins = ["pytester", "llm_ports.testing.pytest_plugin"]

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


@pytest.fixture(autouse=True)
def _backoff_does_not_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The wire-retry loop's backoff waits are recorded, not slept, so a 30 s policy runs at once.

    A test that needs real waits (the deadline tests) patches `_retry._sleep` itself.
    """
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    if _retry is not None:
        monkeypatch.setattr(_retry, "_sleep", record)
    return waits
