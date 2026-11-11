# Work market

Background LLM work (enrichment, research, embeddings, nightly summaries) does not need to run on
a clock. A capacity-driven scheduler runs it when the model server is free: it keeps a scored
backlog, claims the most valuable due item, runs its handler, and records the run. The scheduler,
its backlog and its store are yours. `llm_ports.market` holds what every producer and that
scheduler share, with no store, no table name and no setting name built in.

## Declare an action

A feature declares an action by decorating its handler:

```{.python test="skip"}
ACTIONS = ActionRegistry("reports")

@ACTIONS.action(feature="report", action="weekly", lane=3, cadence="7d", stale_fn=due_weeks,
                cache_table="report_out", est_cost_s=60)
async def weekly(session, work_key): ...
```

| field | what the scheduler does with it |
|---|---|
| `lane` | claims a lower lane first |
| `value_weight` | divides time used by it for fair share inside a lane |
| `desired_ttl` / `cadence` / `max_silence_h` | judges the action's output stale |
| `silence_clock` | counts silence on `wall` hours or `session` hours (your calendar) |
| `stale_fn` | `(session, ttl_s) -> work keys due`; None for an action fed only by events |
| `est_cost_s` | scales its dispatch timeout and lease |
| `source` | skips the action while a content source it reads is switched off |

**A declaration nothing could judge is refused at import.** An action with none of `desired_ttl`,
`cadence` or `max_silence_h` cannot be judged stale by a freshness check, so it would read as
unknown forever; `action(...)` raises `ValueError` naming it, and emits `market_action_refused`
first. A malformed interval literal, an unknown clock, a negative lane or a non-positive weight or
cost is refused the same way. So is a `desired_ttl` without both `cache_table` and `age_col` (the
output row whose age the TTL is measured against), and, on a registry built with
`ActionRegistry(sources=...)`, a `source` id that is not among them.

A key is declared once. Re-adding the identical spec object is a no-op; any other spec under a
held key raises `DuplicateAction`, naming both origins, even when its metadata is equal: an equal
spec with a different handler would otherwise swap the handler in silently. The registry is frozen
once everything is loaded (a strict `load_packs` freezes it; a process that loads no packs calls
`registry.freeze()` itself); a later declaration raises.

## Pause without a rebuild

`KillSwitch(read, default=...)` is the operator's paused set: comma-separated `feature.action`
names from a setting whose name you choose. Unset means `default`; set, even to the empty string,
overrides it. The set is parsed once and emitted once as `market_actions_disabled`;
`unknown(registry)` names a paused entry that matches no action (a typo pauses nothing). Pass
`KillSwitch(..., registry=registry)` and every parse checks it, emitting `market_kill_unknown`.

## Gate a call by its category

`GateCache(read)` caches a per-category switchboard (`Gate(enabled, call_cap, used)`) for `ttl_s`.
A failed read serves the last known gates and is emitted as `market_read_failed`, and the next read
still waits one TTL. `GateCache.admit` is a `call.Hooks.admit`, so a switched-off category returns
`gated` before a trace opens:

```{.python test="skip"}
gates = GateCache(read_gates, window="daily")
await gates.refresh()  # at boot: a gate never read refuses every category
hooks = call.Hooks(admit=gates.admit, trace=call.tracer_trace(tracer))
```

**A gate that was never read is not a gate.** Until the first successful read, `check` refuses
every category with `gate_unread` (`GATE_UNREAD`), so a boot that forgot `refresh()` fails closed
and says why. Each such refusal emits `market_gate_unread` (WARNING, `serving: "closed"`).
`GateCache(read, fail_open=True)` admits instead and emits the same event with `serving: "open"`,
for a caller that would rather run ungated than stop.

A failed boot read keeps the gate closed until a later `refresh()` succeeds, and the cache never
reads on its own: `check` and `admit` judge only what `refresh()` last stored. Consumers **must**
call `await gates.refresh()` at boot and on a schedule (a loop or a cron tick at about the TTL);
without the schedule, a store that was down at boot keeps every category refused.

`LastKnown(read, initial)` is the same cache for one value, such as a global work switch.

## Mirror the registry for an operator

An operator tunes scheduling (lane, weight, ttl, enabled) in your store, without a deploy.
`ActionMirror.sync_action` upserts the fields code owns (`CODE_OWNED`) and seeds the ones the
operator owns (`OPERATOR_TUNED`) only on first insert. `sync_all(registry, mirror)` syncs every
spec, paused ones included, and reports a failed sync instead of raising.

## Producers from another distribution

A distribution that declares actions for a scheduler it does not run lists a pack module under the
`llm_ports.market` entry-point group (install `llm-ports[market]`):

```{.python test="skip"}
# reports_pkg/market.py
from llm_ports.adapters.pluggy import hookimpl

PACK = ActionRegistry("reports")

@hookimpl
def market_actions() -> ActionRegistry:
    return PACK
```

```toml
[project.entry-points."llm_ports.market"]
reports = "reports_pkg.market"
```

The scheduler's process calls `adapters.pluggy.load_packs(registry, builtins=..., pin=...)` once
at boot, and the load freezes `registry` when it finishes, so declare the process's own actions
before it (or pass them as `builtins`). Built-ins register first by name; entry points follow,
sorted by name. `pin` is the value of your pin setting (`PACKS_PIN_ENV`, `LLM_PORTS_PLUGINS`, by
convention; the library reads no environment): None loads everything installed, `""` the
built-ins only, `"a,b"` exactly those, and a pinned name that is not installed raises.

`load_packs` raises `PackLoadError` on the first pack that fails, naming the group, the pack and
the stage (`import`, `register`, `hook`, `duplicate`, `pin`, `pending`), so a missing producer
stops the boot. `load_packs(..., strict=False)` records each pack as a `PackLoad` instead (`ok`,
`failed` with its stage, or `blocked`) for a diagnostic, and leaves the registry open. Each failure
emits `market_pack_failed` with `stage` in the `PackLoad.stage` vocabulary (`import`, `validate`,
`pack`, `join`, `pin`) and `plugin_stage` in the `PackLoadError.stage` one. Every load emits `plugins_loaded` with
the names and their origins (`builtin` or `<distribution>==<version>`); `packs_loaded(loads)` is
the same record as a `Loaded`.

## The whole declaration side

```{.python title="examples/work_market.py"}
--8<-- "examples/work_market.py"
```

Source: [`examples/work_market.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/work_market.py)
