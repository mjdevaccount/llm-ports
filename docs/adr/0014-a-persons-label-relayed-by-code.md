---
status: accepted
date: 2026-11-08
---

# A person's label can be relayed by code, named by the person

Decided in 1.12.0.

## Context and problem statement

`LangfuseExampleStore.put` refused every `human` label: a human label was given only in
Langfuse's annotation-queue UI. The labels a judge or a fine-tune needs come from the person who
reads the replies, and that person reads them on whatever surface they already use. A consumer
that wants a good / bad button beside the reply had two choices: keep a second label store of its
own, or send the person to a different tool for every label. A second store splits the counts
`coverage` and `agreement` read, so the gap and the judge's kappa would each be half a number.

## Decision drivers

- One label store: a label given anywhere is read by the same `examples`, `coverage` and
  `agreement`.
- A human label must stay a person's judgement. Code may carry it; code may not invent it.
- The review queue must not hold a reply already labelled, or a nightly top-up stalls on it.

## Considered options

1. Keep the refusal; a consumer keeps its own table of button labels.
2. `put` accepts a `human` example only when `by` names the person, writes it under the same
   `label-<role>` score config the annotation UI uses, and completes the reply's queue item.

## Decision outcome

Option 2. A relayed human label is scored against the role's categorical config (created when
missing, as `enqueue` does), so Langfuse shows it beside annotation-UI labels. Its score id is keyed
by the person as well as the role and reply: one person relabelling replaces their label, two
people's labels on one reply are two examples, and `learn.examples.latest` still counts one per
source. After the scores land, each relayed reply's pending queue item is marked `COMPLETED`, the
status the annotation UI sets. An empty `by` is a `request_fault`; an `outcome` label is still
never written by code.

### Consequences

- Good: a consumer's button and the annotation UI feed one count and one agreement.
- Good: the person is on every human label (`metadata.by`), whichever surface it came from.
- Neutral: Langfuse records a relayed label's source as `API`, not `ANNOTATION`; the store reads
  both the same.
- Bad: a queue-item update that fails after the scores landed leaves that item pending until the
  next relabel; `put` reports `unavailable` with every score counted, so the caller can see it.
