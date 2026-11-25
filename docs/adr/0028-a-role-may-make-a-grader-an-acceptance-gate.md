---
status: accepted
date: 2026-11-25
---

# A role may make a grader an acceptance gate

Decided in 1.27.0. Extends ADR-0001 and ADR-0019; changes neither.

## Context and problem statement

`call.generate` scores every `ok` reply with the graders its role names. The scores are written on
the observation, stamped on the result and handed to `Hooks.on_scores`. The outcome stays `ok`:
a grader measures, and the caller decides.

In practice no caller decided. A role's reply went from `generate` to a table or a screen with a
failed score beside it, and the only reader of the score was a counter. For most roles that is
the right trade: a summary with one copied sentence is still worth showing. For a role whose
reply a person acts on, a figure no input rendered is not a quality note, it is a reply that
should not have been returned.

A caller could refuse on `stamp["scores"]` itself. Each one would then write the same four lines,
none would get a second attempt out of the model, and the refusal would be a caller's private
convention with no outcome, no event and no trace.

## Decision drivers

- The reply is refused where it is graded, by the call path every caller already uses.
- The model gets the chance to fix it that a validator's problem already gets.
- A refused reply cannot be mistaken for an accepted one by a caller that reads only `outcome`.
- Whatever counted the failure before the role was gated still counts it.
- Declared per role, and off unless declared: a grader stays a measurement everywhere else.

## Considered options

1. A `refuse_on_scores` argument to `generate`.
2. A new outcome, `refused`.
3. `Role.gates`: the role names which of its graders are gates; the call path enforces them
   through the validator seam and returns `invalid`.

## Decision outcome

Chosen option 3.

- **`Role.gates`** is a tuple of grader names, each of which must be in `Role.graders`
  (`ValueError` otherwise). Empty by default.
- **A gate is its key in the grader table.** A table may bind a built-in under another key
  (`{"complete_nullable": partial(schema_complete, nullable=...)}`), and that grader's score
  still says `schema_complete`. So a score is matched to its gate by its position in
  `Role.graders`, never by `CallScore.name`, and the reason, the re-ask and
  `stamp[call.GATES_FAILED]` name the gate by the key. A gate with no score refuses.
- **A gate the table does not hold is a `request_fault`.** It could only crash, and a crashed
  gate refuses: every reply would be refused after its re-ask. The call is refused before any
  wire request, and `llm_call_failed` says which gate.
- **A failing reply is re-asked.** `generate` adds the gates to the call's validator. The caller's
  validator runs first and its problems win; a reply it accepts is scored by the gates, and each
  failure is a problem naming the grader and its reason. The adapter re-asks while
  `Role.output_retries` remain, exactly as for any validator problem. The re-ask message for gate
  problems is the library's; any other problem list goes to the caller's `retry_message`.
- **A re-ask leaves a witness.** A reply the re-ask fixed passes every score, so nothing on it
  would say an earlier reply was refused. Each gate re-ask emits `llm_gate_reasked` (the role,
  the gates, their reasons), and the result carries `stamp[call.GATES_REASKED]`, which is also
  written on the observation's metadata. A `Generator` asks for the re-ask message exactly when
  it sends a re-ask; that is where the re-ask is recorded, so the witness and the text the model
  was sent cannot disagree.
- **A reply that still fails is `invalid`.** After the graders score the kept reply, a failed gate
  turns `ok` into `invalid` (`failure.kind == "invalid_output"`), with `value=None`,
  reason `gate: <grader>: <the grader's reason>`, and `stamp[call.GATES_FAILED]` naming the gates.
  The observation ends in error with the same outcome, and `llm_call_failed` is emitted.
- **The scores survive the refusal.** `stamp["scores"]` is kept and `Hooks.on_scores` is awaited
  with it, so a gauge of failed graders counts a refused reply exactly as it counted an accepted
  one. `Hooks.on_ok` does not see it. When a table lists `invalid_output` in a tier's
  `fallback_on`, the refused tier's scores are handed to `on_scores` too, before the answering
  tier's: the tier that answered did not fail, and the one that did must still be counted.
- **A gate follows its grader, and the result says when it left.** A call that drops a grader
  for its inputs (an image, a web tier's pages: `graders_for_inputs`) drops the gate with it:
  the grader cannot read what the model read, so its verdict would refuse honest replies. The
  reply then goes back unchecked by that gate, which nothing outside the call could otherwise
  tell from a pass, so the result carries `stamp[call.GATES_DROPPED]` naming the gates and
  `llm_call_warning` (`setting="gates"`) is emitted once per process for the role. A consumer
  that must never return an unchecked reply reads the stamp.
- **A crashed gate refuses.** A grader that raises is a failed score (`CRASHED`), so it refuses;
  a gate that cannot run must not read as a pass.

`invalid` is in no tier's default `fallback_on`, so a refusal does not fail over unless the table
says so. `call.chat` and `call.stream` are not gated: a tool loop and a token stream have no
single reply to refuse before it is returned. They grade the turn with the role's graders, as
before, and never refuse it.

A gate is only as good as its grader. A role should gate on a grader whose failures, read on that
role's own replies, are failures: gating on a grader that fails honest replies turns a noisy score
into a silent outage. ADR-0027 is the change that made `numbers_from_inputs` fit to gate on.

### Consequences

- Good: a caller that handles `outcome != "ok"` (every caller, since ADR-0001) refuses a gated
  failure with no new code.
- Good: one more attempt before the refusal, with the failure named to the model.
- Good: the refusal is an outcome, an event, a trace level and a stamp, not a convention.
- Bad: a gate grader runs twice on the kept reply, once as a validator and once as a score.
  Graders are pure and cheap; the second run is what writes the score.
- Bad: a gated role spends a re-ask on a failure it used to return. The cost is one request per
  failing first attempt.
- Neutral: a refusal and a schema failure share the outcome `invalid`. The reason and
  `stamp[call.GATES_FAILED]` tell them apart; a consumer that counts refusals reads the stamp.

### Option 1, rejected

Whether a role's reply may be returned with a failed grader is a property of the role, declared
once beside its graders, not an argument each call site remembers.

### Option 2, rejected

The outcome vocabulary is closed and every consumer switches on it (ADR-0001, ADR-0020). A reply
the model produced and the contract rejects is what `invalid` already means.
