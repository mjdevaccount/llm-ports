"""Test doubles for every port, a fake clock, and conformance suites.

`testing.conformance` (no extra) checks any implementation of a port against its contract;
`testing.pytest_plugin` is the pytest plugin (loaded by pytest's entry point, never by
`import llm_ports`); `testing.langfuse` (the `langfuse` extra) is imported by path.
"""

from llm_ports.testing._clock import FakeClock
from llm_ports.testing._generator import STUB_MODEL, Reply, ScriptedReply, StubGenerator
from llm_ports.testing._stubs import (
    RecordedGeneration,
    RecordingTracer,
    StubCiter,
    StubDetector,
    StubEmbedder,
    StubPromptSource,
    StubReranker,
)
from llm_ports.testing._violation import ContractViolation

__all__ = [
    "STUB_MODEL",
    "ContractViolation",
    "FakeClock",
    "RecordedGeneration",
    "RecordingTracer",
    "Reply",
    "ScriptedReply",
    "StubCiter",
    "StubDetector",
    "StubEmbedder",
    "StubGenerator",
    "StubPromptSource",
    "StubReranker",
]
