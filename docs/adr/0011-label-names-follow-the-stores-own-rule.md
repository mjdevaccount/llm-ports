---
status: accepted
date: 2026-11-07
---

# Label names follow the store's own naming rule

Decided in 1.9.1.

## Context and problem statement

The Langfuse label store named every label score `<prefix>:<role>` (`label:<role>`,
`label.model:<role>`, `label.judge:<role>`), and named each role's categorical score config and
annotation queue after the human label. Langfuse validates a score-config name against
`^[\p{L}\p{N}_ .()-]+$` with at most 80 characters (`packages/shared/src/domain/score-configs.ts`),
so the first config create for every role was refused with a 400 and no queue was ever made.
The unit tests passed because the client double accepted any name. A refusal reason carried only
`Error 400`; the server's message naming the cause was cut with the body.

## Decision drivers

- A name the store refuses must fail where it is built, not on the first remote write.
- Scores already written under the `:` spelling must stay readable.
- A double that stands in for a server must refuse what the server refuses.

## Considered options

1. Keep `:` for model and judge scores (score names are not validated) and change only the human
   config name.
2. One separator for every source, valid for a config name, and reads that accept the old one.

## Decision outcome

Option 2. The separator is `-`: `label-<role>`, `label.model-<role>`, `label.judge-<role>`. The
human score is named after its config, and agreement pairs a human label with the model and judge
labels on the same role, so one spelling for all three keeps the pairing a string operation.
`parse_name` and `examples` accept both spellings. `CONFIG_NAME` and `CONFIG_NAME_MAX` encode the
server's rule (length in UTF-16 code units, as the server counts), and `enqueue` and `put` refuse a
role that breaks it as a `request_fault` before any write. A failure reason appends the first line
of the server's validation message, within `REASON_MAX`. Reads split the name filter into
`NAMES_PER_READ` names per request, since both spellings double the query string.

### Consequences

- Good: an invalid role fails locally with a reason that names the rule.
- Good: the test double enforces the server's rule, so the class of bug is caught in the suite.
- Neutral: a read sends twice the names until no score carries the `:` spelling.
- Bad: the rule is copied from the server; a server change to it needs a release here.
