"""`ContractViolation`: what a conformance check and a scripted stub raise when a contract breaks."""

from __future__ import annotations


class ContractViolation(AssertionError):
    """An implementation broke one invariant of its port's contract.

    Attributes:
        invariant: The invariant's name (`round_trip`, `never_raises`, ...).
    """

    def __init__(self, invariant: str, detail: str) -> None:
        """Name the broken invariant and what was seen."""
        super().__init__(f"{invariant}: {detail}")
        self.invariant: str = invariant


__all__ = ["ContractViolation"]
