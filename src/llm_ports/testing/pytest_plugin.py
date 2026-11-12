"""The `llm_ports` pytest plugin, registered through the `pytest11` entry point.

Installing `llm-ports` in a test environment turns it on; `-p no:llm_ports` turns it off.

- **Live model requests are blocked in every test** (`testing._live.block_live_requests`): a
  request that reaches a real network transport raises `LiveRequestBlocked` naming the host, and
  pydantic-ai's `ALLOW_MODEL_REQUESTS` is False. A test opts in with `@pytest.mark.llm_live`; it
  then runs only when `--llm-live` is passed, and is skipped otherwise.
- **Fixtures:** `stub_generator` (a factory; every script it built must be spent by teardown),
  `recording_tracer`, `fake_clock`, and `llm_replay` (a cassette by path: replayed, or recorded
  with `--llm-record`; a partly played cassette fails at teardown).

The plugin is test-only: it is loaded by pytest, never by `import llm_ports`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_ports.testing._clock import FakeClock
from llm_ports.testing._generator import StubGenerator
from llm_ports.testing._live import block_live_requests
from llm_ports.testing._stubs import RecordingTracer
from llm_ports.testing._violation import ContractViolation

LIVE_MARK = "llm_live"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add `--llm-live` and `--llm-record`."""
    group = parser.getgroup("llm_ports")
    group.addoption("--llm-live", action="store_true", help="run tests marked llm_live against live models")
    group.addoption("--llm-record", action="store_true", help="llm_replay records cassettes instead of replaying")


def pytest_configure(config: pytest.Config) -> None:
    """Register the `llm_live` marker."""
    config.addinivalue_line("markers", f"{LIVE_MARK}: the test may call a live model (needs --llm-live)")


@pytest.fixture(autouse=True)
def _llm_ports_block_live(request: pytest.FixtureRequest) -> Iterator[None]:
    """Block live requests unless the test is marked `llm_live` and the run passed `--llm-live`."""
    if request.node.get_closest_marker(LIVE_MARK) is not None:
        if not request.config.getoption("--llm-live"):
            pytest.skip("llm_live test: pass --llm-live to run it")
        yield
        return
    release = block_live_requests()
    try:
        yield
    finally:
        release()


@pytest.fixture
def stub_generator() -> Iterator[Callable[..., StubGenerator]]:
    """Build `StubGenerator`s; a scripted one with replies left over fails the test at teardown."""
    made: list[StubGenerator] = []

    def make(reply: Any, **kw: Any) -> StubGenerator:
        gen = StubGenerator(reply, **kw)
        made.append(gen)
        return gen

    yield make
    for gen in made:
        gen.assert_spent()


@pytest.fixture
def recording_tracer() -> RecordingTracer:
    """A fresh `RecordingTracer`."""
    return RecordingTracer()


@pytest.fixture
def fake_clock() -> FakeClock:
    """A fresh `FakeClock` at 0.0."""
    return FakeClock()


@pytest.fixture
def llm_replay(request: pytest.FixtureRequest) -> Iterator[Callable[..., Any]]:
    """Return `open(path, live=None, *, secrets=())` for a pydantic-ai model over a cassette.

    Replaying (the default), it returns `replay(cassette)` and fails at teardown when an entry was
    never played. With `--llm-record` and a `live` model, it returns a version-2 `Recorder` over
    `live` and writes the cassette at teardown. Needs the `pydantic-ai` extra.
    """
    from llm_ports.adapters.pydantic_ai.replay import Recorder, replay

    record = bool(request.config.getoption("--llm-record"))
    opened: list[tuple[Path, Any]] = []

    def open_(path: str | Path, live: Any = None, *, secrets: tuple[str, ...] = ()) -> Any:
        p = Path(path)
        if record:
            if live is None:
                raise ContractViolation("record_needs_live", f"--llm-record on {p} needs a live model")
            model: Any = Recorder(live, version=2, secrets=secrets)
        else:
            model = replay(json.loads(p.read_text(encoding="utf-8")))
        opened.append((p, model))
        return model

    yield open_
    for p, model in opened:
        if isinstance(model, Recorder):
            p.write_text(json.dumps(model.cassette(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif model.unplayed():
            raise ContractViolation("cassette_unplayed", f"{p}: never played {', '.join(model.unplayed())}")


__all__ = [
    "LIVE_MARK",
    "fake_clock",
    "llm_replay",
    "pytest_addoption",
    "pytest_configure",
    "recording_tracer",
    "stub_generator",
]
