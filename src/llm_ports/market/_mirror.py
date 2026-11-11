"""The registry mirror: the consumer's store copy of every declared action, which an operator tunes.

Code owns what an action IS (handler, model, cost, cadence, output table); the operator owns how it
is SCHEDULED (lane, weight, ttl, enabled). A consumer implements `ActionMirror` over its own table:
`sync_action` upserts the code-owned fields and seeds the operator-owned ones only on first insert,
so a sync never clobbers a tuning edit. The table's name and schema are the consumer's.

Invariants:
    - `sync_all` syncs EVERY registered spec, paused ones included: the mirror's code fields are
      code truth, and a paused row naming a handler the code no longer has is a provenance lie.
    - `sync_all` never raises: each failed sync is emitted as `market_sync_failed` (WARNING) and
      returned, and the remaining specs still sync.

Rationale: ADR-0018.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from llm_ports.market._registry import ActionRegistry
from llm_ports.market._spec import ActionSpec
from llm_ports.observability import EventHook, log_events

_log_event = log_events("llm_ports.market")

#: Fields the code owns: a sync overwrites them on every pass.
CODE_OWNED: tuple[str, ...] = ("handler_path", "model", "est_cost_s", "cadence", "cache_table", "age_col")
#: Fields the operator owns once a row exists: a sync seeds them on first insert only.
OPERATOR_TUNED: tuple[str, ...] = ("lane", "value_weight", "desired_ttl", "enabled")


class ActionMirror(Protocol):
    """A consumer's store copy of the registry."""

    async def sync_action(self, spec: ActionSpec) -> None:
        """Upsert `spec`'s `CODE_OWNED` fields; seed `OPERATOR_TUNED` ones only on first insert.

        Raises:
            Exception: Any store failure; `sync_all` catches and reports it.
        """
        ...


@dataclass(frozen=True)
class SyncReport:
    """What one `sync_all` pass did.

    Attributes:
        synced: Names of the specs synced.
        failed: `(name, reason)` for each spec whose sync raised.
    """

    synced: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]


async def sync_all(registry: ActionRegistry, mirror: ActionMirror, *, on_event: EventHook | None = None) -> SyncReport:
    """Sync every spec of `registry` into `mirror`. Never raises.

    Args:
        registry: The declared actions.
        mirror: The consumer's store copy.
        on_event: Receives `market_sync_failed`; without one, the stdlib logger `llm_ports.market`.

    Returns:
        The names synced and the failures, in declaration order.
    """
    emit = on_event or _log_event
    synced: list[str] = []
    failed: list[tuple[str, str]] = []
    for spec in registry.registered():
        try:
            await mirror.sync_action(spec)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:200]
            emit(
                logging.WARNING,
                "market_sync_failed",
                {"registry": registry.name, "action": spec.name, "reason": reason},
            )
            failed.append((spec.name, reason))
        else:
            synced.append(spec.name)
    return SyncReport(synced=tuple(synced), failed=tuple(failed))


__all__ = ["CODE_OWNED", "OPERATOR_TUNED", "ActionMirror", "SyncReport", "sync_all"]
