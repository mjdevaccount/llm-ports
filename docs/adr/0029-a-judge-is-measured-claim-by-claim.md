---
status: accepted
date: 2026-11-26
---

# A judge is measured claim by claim, against four labels

Decided in 1.28.0. Extends ADR-0012 (a judge earns trust from labels); changes nothing it decided.

## Context and problem statement

ADR-0012 measures a judge by Cohen's kappa between its pass/fail and a reference's pass/fail over
whole replies. That number says whether a judge can be trusted. It does not say why it cannot. A
local judge measured that way failed most replies a reference graded good, and reading its reasons
showed three different mistakes under one verdict: a figure called "invented" that the Input
carried, an inference (a difference of two given values, a weekday from a date) called
"invented" because no Input sentence stated it, and a field the Input asked to hold a quote called
an echo. A reply-level kappa cannot separate them, so it cannot say which change to the judge
would help.

A judge also cannot be asked to cite what it checked against if the Input is a block of prose
with no handles.

## Decision drivers

- A measurement that names the class of mistake, so one tuning round can target it.
- Inference that follows from the Input is a different thing from invention, and is scored apart.
- A slip in a clause nobody acts on is not the same failure as a wrong figure a reader acts on.
- The consumer's domain rules (what may be inferred) are data the consumer passes, never
  vocabulary in the library.
- The same gold file must load whether it was labelled by a person, a reference model, or by
  construction (a known mutation of a known-good reply).
- No model, clock or network in the measurement.

## Considered options

1. Keep reply-level pass/fail, add a free-text reason taxonomy.
2. Three NLI labels (entailed / contradicted / neutral), as natural-language inference uses.
3. Four claim labels: `supported`, `derived`, `not_in_input`, `contradicted`, with a materiality
   flag per claim.

## Decision outcome

Option 3, in `evals.claims`.

- A reply is decomposed into claims, each with its own label, as claim-level faithfulness methods
  do (FActScore, Min et al., EMNLP 2023; RAGAS faithfulness, Es et al., EACL 2024 demo track).
- The NLI labels are kept and one is split: `contradicted` is NLI's contradiction,
  `not_in_input` its neutral, and entailment is split into `supported` (a fact states it) and
  `derived` (it follows from cited facts by a stated operation `op`). The split is the point:
  the mistake to see is a judge that labels `derived` claims `not_in_input`.
- `case_label`: a case fails when a material claim is `not_in_input` or `contradicted`. A
  claim's `material` defaults to True; an immaterial slip is counted and does not fail the case.
- `agreement` scores each label one-against-the-rest with Cohen's kappa and its 95% interval
  (`learn.stats.kappa_ci`), because the library's kappa is the two-class one, plus `grounded`
  (supported or derived against the rest), the question a case verdict turns on. An abstention
  is left out, never scored as wrong.
- `split` puts a case in train / dev / test by a stable hash of its id, so a tuning round that
  reads dev never reads test, and adding a case moves no other.
- `flips` counts verdicts a reworded prompt changed. Judge verdicts are reported to move under
  rewording that keeps meaning (Thakur et al., "Judging the Judges", 2024), and a judge that
  flips is reading the wording.
- `fact_table` renders facts as `<id> <text>` lines, so a judge can name the ids it checked.
- `judge.rubric(rules=...)`: the consumer's inference rules reach the judge as law 5, one per
  line. With none passed the rubric is unchanged.
- `read_cases` loads any JSON-lines family carrying the core keys; every other key rides in
  `extra`, and `rule_aliases` renames rule ids so data written under an older rule set loads
  unedited. A bad row is a named problem, never a raise (ADR-0001).

Checked before writing: ragas `Faithfulness` and deepeval `FaithfulnessMetric` decompose a reply
into claims and ask a model for a yes / no / unknown verdict on each; neither scores a judge
against gold labels or reports an interval. `judgy` estimates a population pass rate corrected for
a judge's known true-positive and true-negative rates; it consumes an agreement measurement and
does not produce one. `learn.examples.Agreement` already held kappa with its interval and is
reused here.

### Consequences

- Good: a judge's report names its mistake class, and the trust rule of ADR-0012 can read the
  `grounded` agreement of a claim-level gold set instead of a handful of whole replies.
- Good: one loader serves hand-labelled production replies and generated mutation fixtures.
- Bad: claim-level gold costs more to label than a pass/fail per reply.
- Neutral: re-checking a `derived` claim's `op` in code is left to the consumer for now; `op` is
  carried as text so the check can be added without a data change.
