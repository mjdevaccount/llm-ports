---
status: accepted
date: 2026-11-03
---

# Data rights and lineage in PROV-O terms, with one deterministic print gate

Decided in 0.7.0 (`content`). Recorded in 0.12.0.

## Context and problem statement

A consumer prints generated items (cards, notes, briefs) built from sources with different rights:
public statistics, the owner's own data, a vendor's personal-use licence, pages scraped without a
licence. Who may see an item depends on the most restrictive thing it was made from, and whether
its claims cite evidence decides whether it may be printed at all. Without a shared shape, each
surface re-derives this, and a lineage reader has to learn each surface's vocabulary.

## Decision drivers

- One rule, pure and deterministic, that says "print" or names exactly one refusal.
- Vocabulary a lineage reader already knows.
- No policy engine to deploy for a single-owner library.

## Considered options

1. Ad-hoc flags per surface.
2. A policy engine evaluating ODRL policies.
3. Small frozen types named in W3C PROV-O terms, ODRL's words for the rules, and one pure gate.

## Decision outcome

Chosen option 3, in `content`:

- `ContentItem` is a `prov:Entity`, `Activity` a `prov:Activity` (trace, role, pack@sha, model,
  git sha); `uses` is `prov:used`, `as_of` is `prov:generatedAtTime`. An unrecorded field is
  absent, never faked.
- `DataClass` holds rights (`public < owned < licensed < scraped`), `private` and `link_required`.
  An item's class is the join of what it uses: the most restrictive tier, private if any input is.
- `CEILING` maps each audience to the tiers it may see.
- `print_gate(item, audience)` returns ok, or one named refusal in a fixed order: `rights`,
  `unclaimed` (the body was never decomposed into claims), `uncited`, `unsourced`, `link`.
  `unclaimed` is separate so a body nobody checked is never reported as uncited by guess.

### Consequences

- Good: lineage exports as PROV-JSON with no translation layer.
- Good: a refusal is a name a gauge or a reader can count.
- Bad: ODRL supplies words, not enforcement; the gate is as strong as the classes callers assign.
