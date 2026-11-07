# Learning loops

Every learning loop is the same moves: read evidence, fit or measure, hold the result to a gate,
act on the verdict, and record the run. `llm_ports.learn` holds the parts that do not depend on
what is learned, on the standard library alone, so it lifts into any codebase.

A loop implements four stages and nothing else:

| stage | what it does | may raise? |
|---|---|---|
| `evidence()` | read what it learns from (a repository call, a file) | yes: recorded as `could_not_run` |
| `fit(evidence)` | fit or measure; returns `Fit(model, metrics)` | yes: same |
| `clauses(fit, evidence)` | the gate's evaluated clauses | yes: same |
| `apply(verdict, fit)` | act: store the fit on `pass`, keep the incumbent otherwise | yes: recorded as `apply_failed`, the verdict stands |

`run(loop, ledger)` drives them, decides the verdict with `gate.decide`, and appends exactly one
ledger row per run, every exit included. It never raises.

## The verdict

A clause is a condition, evaluated: `passed` is True, False, or None when it could not be
evaluated. Its `role` says what failing it means:

- `bar` (default): measured and did not clear: `fail`.
- `floor`: a sample floor, or a condition not yet proven: `insufficient`. A failing `floor` also
  turns a failing `bar` into `insufficient`, because a bar measured on too small a sample is noise.
- `veto`: a disqualifier: `fail`, whatever else holds.

A clause that is None never passes. By default it reads as a failure; a loop with
`unevaluated = "insufficient"` reads it as unproven instead, which is how a promote / hold /
reject gate maps on exactly: `pass` = promote, `insufficient` = hold, `fail` = reject.

The verdict keeps every clause (`Verdict.blocking` lists each one that did not pass) and names
the first that decided it (`blocker`), so a page can show all the reasons, not only one.

## The ledger

`RunLedger` has two methods, `append` and `latest`. `MemoryRunLedger` and `JsonlRunLedger` ship
here; a database-backed ledger is an adapter in your application over its own table. A row's
verdict lives under `metrics["learn_verdict"]` (`VERDICT_KEY`), a name chosen so it never collides
with keys an existing ledger already uses. `latest(kind, decisions=("pass", "fail"))` reads the
last decisive run, so a later `could_not_run` does not hide what the last real measurement said.

## Labelled examples

`Example(topic, subject, value, source, by, at, reason, ref)` is one label from a `human`, a
`model`, a production `judge`, or an `outcome` resolved from the world. An `ExampleStore` reads
them by topic; stores are adapters over wherever labels already live, and `stores.union` reads
several as one. Three pure readers serve a coverage report and a judge:

- `examples.coverage(examples, topics, target=...)`: per topic, how many reference labels exist
  and the gap to the target; a topic with no label is reported with zeros.
- `examples.agreement(examples)`: per topic, how a judge or model agrees with the human labels
  (share, Cohen's kappa, Gwet's AC1).
- `examples.few_shot(examples, k, topic=...)`: a deterministic pick balanced across classes, for a
  judge's prompt.

## The whole loop

```{.python title="examples/learning_loop.py"}
--8<-- "examples/learning_loop.py"
```

Source: [`examples/learning_loop.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/learning_loop.py)
