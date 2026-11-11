"""Test doubles for every port, and conformance suites for backends.

`testing.conformance` (no extra) checks a `RunLedger`, `ExampleStore` or `ModelRegistry` backend
against its port's contract; `testing.langfuse` (the `langfuse` extra) is imported by path.
"""

from llm_ports.testing._stubs import (
    STUB_MODEL,
    RecordedGeneration,
    RecordingTracer,
    Reply,
    StubCiter,
    StubDetector,
    StubEmbedder,
    StubGenerator,
    StubPromptSource,
    StubReranker,
)

__all__ = [
    "STUB_MODEL",
    "RecordedGeneration",
    "RecordingTracer",
    "Reply",
    "StubCiter",
    "StubDetector",
    "StubEmbedder",
    "StubGenerator",
    "StubPromptSource",
    "StubReranker",
]
