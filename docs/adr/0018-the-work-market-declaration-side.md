---
status: accepted
date: 2026-11-11
---

# The work market's declaration side is a core subpackage; packs from other distributions load through pluggy entry points

Decided in 1.22.0.

## Context and problem statement

A capacity-driven scheduler runs background LLM work when its model server is free. Its producers
declare actions with a decorator: a handler, a stale function, a lane, a weight, a freshness rule.
A consumer had this decorator, its registry, a kill list read from an environment variable and a
cached per-category gate in its own codebase. A second codebase now holds producers for the same
scheduler, so both must declare on one contract, and the scheduler's one process must load the
second codebase's producers without importing it by name.

Three parts are generic: the declaration (with the refusal of an action nothing could judge
stale), the operator's paused set, and the cached control reads `call.generate`'s `admit` hook
judges by. The scheduler itself (backlog, claim order, leases, capacity reads) has one consumer
and stays there.

## Decision

1. `llm_ports.market`, a core subpackage beside `evals`, `rag` and `provenance` (it imports `call`
   for the `admit` hook's `Role`, and the standard library). No extra.
2. `ActionRegistry.action(...)` refuses at decoration a spec with none of `desired_ttl`, `cadence`
   or `max_silence_h`, a malformed interval literal, an unknown silence clock, a negative lane and a
   non-positive weight or cost. An equal re-declaration replaces the held spec (the newest
   callables run); a different one raises. The registry is a `MutableMapping`, so test teardown by
   `pop` and `del` works unchanged.
3. Every input is a parameter: the kill list's setting arrives as a reader (`KillSwitch(read)`),
   the gate's store as a read (`GateCache(read)`), and the registry's store copy as a port
   (`ActionMirror`). No table name and no setting name is in the package.
4. Each failure that would otherwise be silent is an event on an injected hook: a refused
   declaration, a paused set, a paused name matching no action, a failed control read (served from
   the last known value), a failed mirror sync, a pack that loaded, failed or was blocked.
5. Packs from other distributions are found under the `llm_ports.market` entry-point group and
   loaded by `adapters.pluggy.load_packs` (extra `[market]`). A pack module returns its own
   `ActionRegistry` from a `market_actions` hook; it joins the scheduler's registry all or nothing.

## Considered options

- **A stdlib entry-point loader in the core** (`importlib.metadata.entry_points` and a callable
  per entry point). Lighter, but it would hand-write what pluggy already does: validating a hook
  implementation's signature against a spec at registration, a blocked set, and the
  plugin-to-distribution map. The second codebase's source registry is planned on a pluggy
  loader too, so one plugin mechanism serves both.
- **Depend on the consumer's platform package for its duplicate-guarded `Registry`.** That package
  is not a dependency of this library, and taking it would pull a deployment-specific package into
  a public one. The guard is twenty lines here, with the same two policies (an equal spec is a
  re-import, a different one raises).
- **A process-global registry in the library.** Rejected: two schedulers, or a test, could not hold
  separate tables. Each consumer owns its registry object; a pack owns its own and joins one.
- **Move the scheduler too.** One consumer runs it; the extraction rule waits for a second.

## Consequences

- A producer in any distribution declares on the same contract and is judged by the same refusal.
- A pack that fails to import or conflicts is named with its stage, not skipped silently; the
  consumer decides whether that stops the boot.
- The cached gate's cap is enforced against a use count up to one TTL old, so a burst can
  overshoot it by about one TTL's calls. That is stated in the docstring, not found later.
