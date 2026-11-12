"""Pruning the registry mirror: a row whose action the code no longer declares stops looking runnable.

`sync_all` upserts and never removes, so an action deleted from code stays in the consumer's mirror
as an enabled row forever: the operator reads it as "on" and nothing runs it. `prune_mirror` reads
the mirror's rows, and retires each ENABLED row whose action the syncing registry does not hold.

The consumer decides what retiring means for its table (`PrunableMirror.retire_action`): deleting
the row, or setting it disabled. The library never writes `enabled` itself: that is the operator's
column, and a row whose action IS registered is never handed to `retire_action`.

Scope (a process prunes only what it is sure it owns):
    - A row whose `origin` is known is pruned only when that origin is not one that failed to
      load in this process (`unloaded`).
    - A row whose `origin` is unknown (the table does not record it) is pruned only when nothing
      failed to load: a daemon that failed to load one pack cannot tell that pack's rows from
      retired ones, so it holds them all, emits `market_prune_held` (WARNING) and reports them.

Invariants:
    - `prune_mirror` never raises: a failed read is `market_prune_failed` (WARNING) and an empty
      report with `error` set; a failed retire is `market_prune_failed` for that row, and the rest
      still prune.
    - Every retired row is named in one `market_action_pruned` event (INFO), and the report
      counts them, so a caller can gauge the prune.

Rationale: ADR-0018.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from llm_ports.market._mirror import ActionMirror
from llm_ports.market._registry import ActionRegistry
from llm_ports.observability import EventHook, log_events

_log_event = log_events("llm_ports.market")


@dataclass(frozen=True)
class MirrorRow:
    """One row of the consumer's mirror, as `prune_mirror` reads it.

    Attributes:
        name: The action's `feature.action` key.
        enabled: The operator's switch.
        origin: The pack (or module) that declared it, when the table records one; else None.
    """

    name: str
    enabled: bool
    origin: str | None = None


class PrunableMirror(ActionMirror, Protocol):
    """An `ActionMirror` that can list its rows and retire one."""

    async def mirrored(self) -> Sequence[MirrorRow]:
        """Every row of the mirror.

        Raises:
            Exception: Any store failure; `prune_mirror` catches and reports it.
        """
        ...

    async def retire_action(self, name: str) -> None:
        """Retire the row of an action no longer declared: delete it, or set it disabled.

        Raises:
            Exception: Any store failure; `prune_mirror` catches and reports it.
        """
        ...


@dataclass(frozen=True)
class PruneReport:
    """What one `prune_mirror` pass did.

    Attributes:
        pruned: Names of the rows retired.
        held: Names of unregistered enabled rows NOT retired because a pack failed to load and the
            row's origin is unknown or is that pack's.
        failed: `(name, reason)` for each row whose retire raised.
        error: Why the mirror could not be read; None when it was.
    """

    pruned: tuple[str, ...] = ()
    held: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()
    error: str | None = None

    @property
    def count(self) -> int:
        """How many rows were retired."""
        return len(self.pruned)


def _reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


async def prune_mirror(
    registry: ActionRegistry,
    mirror: PrunableMirror,
    *,
    unloaded: Iterable[str] = (),
    on_event: EventHook | None = None,
) -> PruneReport:
    """Retire each enabled mirror row whose action `registry` does not hold. Never raises.

    Args:
        registry: The actions this process declares (after every pack loaded).
        mirror: The consumer's store copy.
        unloaded: The packs that did NOT load in this process (failed or blocked: with
            `adapters.pluggy.load_packs`, the `name` of each `PackLoad` whose `outcome` is not
            `ok`). Their rows are held; see the module docstring.
        on_event: Receives `market_action_pruned`, `market_prune_held` and
            `market_prune_failed`; without one, the stdlib logger `llm_ports.market`.

    Returns:
        The rows retired, held and failed, in the mirror's order.
    """
    emit = on_event or _log_event
    missing = frozenset(unloaded)
    try:
        rows = list(await mirror.mirrored())
    except Exception as exc:
        reason = _reason(exc)
        emit(logging.WARNING, "market_prune_failed", {"registry": registry.name, "action": None, "reason": reason})
        return PruneReport(error=reason)
    declared = {spec.name for spec in registry.registered()}
    pruned: list[str] = []
    held: list[str] = []
    failed: list[tuple[str, str]] = []
    for row in rows:
        if not row.enabled or row.name in declared:
            continue
        if (row.origin is None and missing) or (row.origin is not None and row.origin in missing):
            held.append(row.name)
            continue
        try:
            await mirror.retire_action(row.name)
        except Exception as exc:
            reason = _reason(exc)
            emit(
                logging.WARNING,
                "market_prune_failed",
                {"registry": registry.name, "action": row.name, "reason": reason},
            )
            failed.append((row.name, reason))
        else:
            emit(
                logging.INFO,
                "market_action_pruned",
                {"registry": registry.name, "action": row.name, "origin": row.origin},
            )
            pruned.append(row.name)
    if held:
        emit(
            logging.WARNING,
            "market_prune_held",
            {"registry": registry.name, "actions": tuple(held), "unloaded": tuple(sorted(missing))},
        )
    return PruneReport(pruned=tuple(pruned), held=tuple(held), failed=tuple(failed))


__all__: list[str] = []
