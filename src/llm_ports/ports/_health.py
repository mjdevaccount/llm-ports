"""The `Probe` port: is a backend up, and which models does it serve.

Invariants:
    - `probe` never raises; the result's `state` names what was learned.
    - `unknown` means the PROBE failed (it broke, or could not judge the answer), never that the
      backend is down: a consumer's breaker opens on `down` only.

Rationale: ADR-0022.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

#: What a probe learned: `up`, `down`, or `unknown` (the probe itself failed).
HealthState = Literal["up", "down", "unknown"]


@dataclass(frozen=True, slots=True)
class Health:
    """One probe's answer.

    Attributes:
        state: `up`, `down`, or `unknown` when the probe itself failed.
        models: The model ids the backend lists; None when it listed none or was not asked.
        reason: One line naming why the state is not `up`; None when it is.
        latency_s: Seconds the probe took; None when it never reached the backend.
    """

    state: HealthState
    models: tuple[str, ...] | None = None
    reason: str | None = None
    latency_s: float | None = None


@runtime_checkable
class Probe(Protocol):
    """A health check on one backend."""

    async def probe(self) -> Health:
        """Ask the backend once. Never raises.

        Returns:
            `up` with what it serves, `down` when it could not be reached or answered with an
            error, `unknown` when the probe itself failed.
        """
        ...


__all__ = ["Health", "HealthState", "Probe"]
