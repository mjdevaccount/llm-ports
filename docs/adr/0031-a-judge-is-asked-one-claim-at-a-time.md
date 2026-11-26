---
status: accepted
date: 2026-11-27
---

# A judge is asked one claim at a time, after code settles what it can

Decided in 1.32.0. Builds on ADR-0029 (claims, four labels, materiality) and ADR-0030 (code is
authoritative where it can be); changes neither.

## Context and problem statement

ADR-0029 measures a judge claim by claim, but the judge itself was still asked about a whole reply
at once: one prompt, one pass/fail. A small local judge asked that way failed most replies a
reference graded good. Its reasons showed it could not tell a value copied from the Input from one
the reply invented, so it called both invented: a figure the Input carried with a sign, a category
chosen from a closed set, a name copied verbatim. Given a whole reply and a whole Input, it had
nothing to point at and no smaller question to answer.

## Decision drivers

- The judge stays a small local model; grounding is what helps it, not a larger model.
- What code can check, code checks: a judge is never asked a question a string match answers.
- A "this is wrong" verdict must point at the fact that makes it wrong.
- The question must fit the field: a value chosen from a closed set cannot be "not in the Input".
- The consumer's field contract (which fields are quotes, which are choices, which count) is data
  the consumer passes; the library knows no field.
- No model, clock or network anywhere but the one call per claim.

## Considered options

1. Keep one prompt per reply; add the consumer's rules and a fact table to it.
2. Ask the model to decompose the reply into claims, then judge each (FActScore-style).
3. Decompose in code (one claim per JSON scalar, one per prose sentence), settle in code what a
   verbatim match or the number law licenses, ask the judge one claim per call under a grammar
   chosen by the question, and hold each verdict to a quote from the facts.

## Decision outcome

Option 3: `evals.decompose` (pure) and `evals.claim_judge`.

- **Decomposition is code's.** `decompose.leaves` makes each JSON scalar one claim at its path
  (`items[0].name`); an empty list, object or null is one claim that nothing applies; prose is one
  claim per sentence. A model never decides what a claim is, so two runs see the same claims.
- **The Input becomes a fact table.** `decompose.facts_from_text` numbers its lines (`F1`, ...),
  dropping blanks and repeats and splitting long lines at sentence ends, so a verdict can cite ids
  it was shown (`claims.fact_table`).
- **Code settles only `supported`.** `decompose.prematch` rules a `quote` or `extract` value that
  occurs in the Input as whole words (`occurs`: a value of five characters or fewer is matched as
  written, so a short symbol never matches inside a word), and a bare figure the number law licenses
  (`numbers.loose`, rerendered, ADR-0027). Everything else goes to the judge with what code saw as
  hints. Code never rules a claim ungrounded: a value code cannot find may still be a fair reading.
  A `free` value found verbatim is a hint, not a ruling: an id copied from the Input can still be
  used to claim what the Input does not say. A `classify` value found is a hint: finding a category
  word proves only that it is a word.
- **The question fits the field.** `question_for` asks a `classify` value or a stated absence only
  whether a fact contradicts it (`consistent` / `contradicted`, read as `derived` / `contradicted`);
  everything else gets the four labels. `verdict_schema` makes that the grammar, with the cited ids
  an enum of the table's ids and the evidence fields before the label, so the model writes its
  evidence before its answer.
- **A verdict is held to its evidence.** `read_verdict` abstains when the quote is not in the facts,
  and when `contradicted` carries no quote. An abstention is left out of the case rule, never
  scored as a fail.
- **The case rule is ADR-0029's, with abstention.** `aggregate`: a material claim `not_in_input` or
  `contradicted` fails the case; else a material claim that could not run, or material claims of
  which none reached a label, make it `unjudgeable`; else it passes.
- **The consumer's contract is a `Policy`.** `kind_of` (the field map), `material_of`,
  `question_of` (a class that asserts a cause may be asked `grounding`, since a cause needs
  evidence), `pre` (the consumer's own stronger match, run first; like `prematch` it only rules
  `supported`) and `allowed` (`decompose.enums(schema)`).
- **One call per claim, through the consumer's `generate`.** `GenerateClaimJudge` renders the
  consumer's pack with `claim_slots` (context and facts first, so every call of one case shares a
  prefix a server's prompt cache reuses) and calls the consumer's `generate`, as `refute` does
  (ADR-0024): the consumer's gate, trace and retry apply. Material claims go first, up to a cap;
  an immaterial claim is skipped unless asked for, since it cannot move the case.

### Evidence

Replayed against a 31-case claim-labelled gold set from one production consumer (extraction,
classification and prose roles), on that consumer's deployed local judge model, before release.
Case kappa is pass/fail against the gold, with its 95% interval:

| judge | ran | case kappa | agree | gold-pass recall | gold-fail catch |
|---|---|---|---|---|---|
| whole reply, one prompt (the consumer's previous judge) | 31 | 0.10 [-0.10, 0.29] | 0.45 | 5 / 21 | 9 / 10 |
| claim by claim, as handed in | 30 | 0.32 [-0.01, 0.65] | 0.67 | 13 / 20 | 7 / 10 |
| claim by claim, field definitions as hints | 30 | 0.55 [0.23, 0.87] | 0.80 | 17 / 20 | 7 / 10 |

Of 433 claims, code settled 58 and the judge was asked about 119; the rest were immaterial and
skipped. Code's settlements were checked against the gold's claim labels offline first: 0 of them
overturned a gold label. The consumer's field map and its field definitions were written by
someone who had read the gold, so these are in-sample numbers: the intervals overlap, and the
gain is not proven until a larger held-out gold set measures it.

What the judge still got wrong: a sentiment field "contradicted" by a quote that agrees with it,
a cause class read as needing its word in the text, and an instruction line read as a fact.

Published method: claim-level faithfulness decomposes an answer into atomic claims and checks each
against the source (FActScore, Min et al., EMNLP 2023; RAGAS faithfulness, Es et al., EACL 2024).
Fact-checking work reports that small checker models judge one claim against grounding documents
nearly as well as large ones when the claim is atomic (MiniCheck, Tang et al. 2024,
arXiv:2404.10774). Requiring a cited span before a verdict follows attributed QA practice
(Bohnet et al. 2022, arXiv:2212.08037).

Checked before writing: RAGAS and DeepEval decompose with a model call, which is option 2; a model
that decides what a claim is gives two runs different claim sets, and the decomposition is the
step a small model gets wrong. MiniCheck is a model, not a decomposer. No library decomposes a
JSON reply by schema path. `numbers.loose` (ADR-0003, ADR-0027), `claims.fact_table` and the
`refute` call shape (ADR-0024) are reused.

### Consequences

- Good: a value the reply copied never reaches the judge, so the judge's commonest false fail
  (calling a copied value invented) cannot happen on it.
- Good: a fail names the fact and the words that refute the claim; a fail without them abstains.
- Good: the count of claims code settled, the judge ruled, and the judge abstained on is a
  per-case measurement (`CaseRuling.counts`), so a tuning round can see which step moved.
- Bad: one call per claim costs more calls than one per reply; the cap bounds it and the shared
  prefix keeps each call cheap on a server with a prompt cache.
- Bad: prose is split by a sentence rule, so a claim that spans two sentences is judged in halves.
- Neutral: the field map is the consumer's to write and to keep honest; a wrong `kind` (a free
  value marked `extract`) lets code settle what it should not.
