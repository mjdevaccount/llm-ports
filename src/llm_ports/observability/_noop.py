"""`NoopTracer`: a `Tracer` with no sink; every generation is a `NoopGeneration`.

It is also the null trace `call.generate` uses when no `trace` hook is given, so the package has
one null sink.
"""

from __future__ import annotations

from typing import Any


class NoopGeneration:
    """A `Generation` that records nothing.

    It is what a sink-less `Tracer` returns, and so what `call.generate` observes into when no
    `trace` hook is given.
    """

    def end(self, **_kw: Any) -> None:
        """Record nothing."""
        return None

    def score(self, *_a: Any, **_kw: Any) -> None:
        """Record nothing."""
        return None


class NoopTracer:
    """A `Tracer` for a process with no observability sink.

    Every `start_generation` returns a `NoopGeneration`, so a producer never branches on whether a
    sink is configured.
    """

    def start_generation(self, **_kw: Any) -> NoopGeneration:
        """Return a `NoopGeneration`."""
        return NoopGeneration()


__all__ = ["NoopGeneration", "NoopTracer"]
