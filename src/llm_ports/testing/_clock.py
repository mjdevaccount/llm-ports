"""`FakeClock`: a monotonic clock and an async sleep that only move when told to.

Pass the clock where a component takes `clock=` (a `Deadline`, `TeeTracer`, `OtelTracer`,
`observe`) and its `sleep` where one takes a sleep; a test then advances time instead of waiting.

Example:
    clock = FakeClock()
    await clock.sleep(2.0)       # returns at once; the clock reads 2.0
    clock.advance(0.5)           # 2.5
    assert clock.sleeps == [2.0]
"""

from __future__ import annotations


class FakeClock:
    """A clock that starts at `start` and moves only on `advance` or `sleep`.

    Args:
        start: The first reading.

    Attributes:
        sleeps: Every `sleep` duration asked for, in order.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """The current reading."""
        return self._now

    def advance(self, s: float) -> None:
        """Move the clock forward by `s` seconds.

        Raises:
            ValueError: `s` is negative; a monotonic clock never goes back.
        """
        if s < 0:
            raise ValueError(f"a monotonic clock never goes back: advance({s})")
        self._now += s

    async def sleep(self, s: float) -> None:
        """Record the sleep and advance by it (a negative sleep is zero), without waiting."""
        self.sleeps.append(s)
        self._now += max(0.0, s)


__all__ = ["FakeClock"]
