"""The work market's declaration side: background actions, their registry, and the controls a scheduler reads.

A capacity-driven scheduler runs background LLM work when its model server is free: the most
valuable due item first, fair across features. This package holds the parts every producer and
that scheduler share; the scheduler, its backlog and its store are the consumer's.

- `ActionSpec`, `ActionRegistry`: a feature declares an action with `@registry.action(...)`; a
  spec nothing could judge stale is refused at import.
- `KillSwitch`: the operator's paused `feature.action` set, from a setting the consumer reads.
- `GateCache`, `LastKnown`: cached control reads; `GateCache.admit` is a `call.Hooks.admit`,
  and refuses as `gate_unread` until the consumer's boot has read the gates once.
- `ActionMirror`, `sync_all`: the consumer's store copy of the registry, which an operator tunes.
- `PrunableMirror`, `prune_mirror`: retire the mirror rows of actions the code no longer declares.
- Packs: a distribution exposes its actions under the `ENTRY_POINT_GROUP` entry-point group, and
  `adapters.pluggy.load_packs` adds them to the scheduler's registry (extra `[market]`).

Events (through each component's `on_event` hook, else the stdlib logger `llm_ports.market`):
`market_action_refused`, `market_actions_disabled`, `market_kill_unknown`, `market_read_failed`,
`market_sync_failed`, `market_gate_unread`, `market_action_pruned`, `market_prune_held`,
`market_prune_failed`; the pack loader adds `market_pack_loaded`, `market_pack_failed` and
`market_pack_blocked`.

Invariants:
    - No store, no table name and no setting name is built in: each is passed in by the consumer.

Rationale: ADR-0018.
"""

from llm_ports.market._gate import CATEGORY_CAPPED, CATEGORY_DISABLED, GATE_UNREAD, Gate, GateCache, LastKnown
from llm_ports.market._mirror import CODE_OWNED, OPERATOR_TUNED, ActionMirror, SyncReport, sync_all
from llm_ports.market._prune import MirrorRow, PrunableMirror, PruneReport, prune_mirror
from llm_ports.market._registry import ActionRegistry, DuplicateAction, Key
from llm_ports.market._spec import SILENCE_CLOCKS, ActionSpec, Handler, StaleFn, parse_interval, refusal
from llm_ports.market._switch import KillSwitch, parse_names

#: The entry-point group a distribution lists its action packs under.
ENTRY_POINT_GROUP = "llm_ports.market"

__all__ = [
    "CATEGORY_CAPPED",
    "CATEGORY_DISABLED",
    "CODE_OWNED",
    "ENTRY_POINT_GROUP",
    "GATE_UNREAD",
    "OPERATOR_TUNED",
    "SILENCE_CLOCKS",
    "ActionMirror",
    "ActionRegistry",
    "ActionSpec",
    "DuplicateAction",
    "Gate",
    "GateCache",
    "Handler",
    "Key",
    "KillSwitch",
    "LastKnown",
    "MirrorRow",
    "PrunableMirror",
    "PruneReport",
    "StaleFn",
    "SyncReport",
    "parse_interval",
    "parse_names",
    "prune_mirror",
    "refusal",
    "sync_all",
]
