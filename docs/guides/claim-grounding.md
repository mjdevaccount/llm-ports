# Measuring a judge claim by claim

A judge measured only by pass/fail over whole replies can say *how often* it disagrees with a
reference, not *why*. `llm_ports.evals.claims` grades each claim of a reply against the facts its
Input carried, with four labels (ADR-0029):

| label | meaning |
|---|---|
| `supported` | a fact states it |
| `derived` | it follows from cited facts by a stated operation (`op`) |
| `not_in_input` | no fact states or licenses it |
| `contradicted` | a fact states otherwise |

A case fails when a **material** claim is `not_in_input` or `contradicted`
(`claims.case_label`). A claim is material unless its row says `"material": false`.

## Gold files

`claims.read_cases` reads JSON lines. Each row needs `case_id`, `input`, `output`, `claims` and
`case_label` (`pass`, `fail` or `unjudgeable`); each claim needs `id`, `quote`, `text` and
`label`. `facts` may be a list of `{id, span}` or a mapping of `{id: value}`. Any other key is
kept in `extra`, so a hand-labelled file and a generated mutation file load through the same
call. A bad row is reported in `Loaded.problems` with its line number; nothing raises.

## Scoring a judge

`claims.agreement(pairs)` takes `(judged label, gold label)` per claim and returns Cohen's kappa
with its 95% interval for each label one-against-the-rest, plus `grounded` (supported or derived
against the rest). A judge that calls inference "invented" shows up as a low `derived` and
`grounded` kappa beside a high `supported` one.

`claims.split(case_id)` assigns train / dev / test from a hash of the id, so a tuning round reads
dev and the reported number comes from test. `claims.flips(run_a, run_b)` counts verdicts that
changed when the judge's prompt was reworded.

## The consumer's rules

`judge.rubric(..., rules=("R1: ...", "R2: ..."))` adds the consumer's inference rules as law 5. A
claim a rule licenses is grounded though no Input sentence states it. The library carries no
domain rule of its own.

## Arithmetic is code's, not the judge's

A `derived` claim carries its operation as an op chain, `expr = numeric = result`
(`abs(a - b) / a * 100 = abs(120.0 - 114.0) / 120.0 * 100 = 5.0`). `grounding.derive.recheck`
evaluates it in a closed grammar (numbers, names, `+ - * /`, `abs`, `sqrt`, comparison chains),
holds it to its written result and to the facts, and judges the figure the reply shows:
`holds`, `breaks`, `abstain` (outside the claim's tolerance, inside `band` times it) or
`unchecked` (code cannot read it). Model-written text is never executed (ADR-0030).

`evals.settle.settle(claim, allowed=..., facts=...)` turns that into a ruling on a labelled claim.
A breaking figure is `contradicted` whatever the judge said. A holding figure confirms a grounded
label but never overturns an ungrounded one: the op certifies the figure, not the words around it.
The figures the number law refuses come back as `refused`, a hint for the judge, never a ruling.

## Judging claim by claim

A judge asked about a whole reply at once has nothing to point at. `evals.claim_judge.grade`
asks it about one claim at a time, and only about the claims code cannot settle (ADR-0031):

1. `decompose.leaves(reply)` makes each JSON scalar one claim at its path (`items[0].name`); an
   empty list or null is one claim that nothing applies; prose is one claim per sentence.
2. `decompose.facts_from_text(input)` numbers the Input's lines `F1`, `F2`, ... so a verdict can
   cite what it checked.
3. `Policy.prematch` settles what code can. A `quote` or `extract` value found verbatim
   (`decompose.occurs`) is `supported`, and so is a bare figure the number law licenses. Nothing
   else is ruled, and code never rules a claim ungrounded; what code saw rides to the judge as hints.
4. Each remaining claim gets one call. A `classify` value or a stated absence is asked only
   whether a fact contradicts it (`question_for`); everything else gets the four labels. The
   grammar (`verdict_schema`) asks for the cited ids and the quote before the label.
5. `read_verdict` abstains on a quote that is not in the facts, and on a `contradicted` with no
   quote. `aggregate` applies the case rule; `CaseRuling.counts()` says how many claims code
   settled, the judge ruled and the judge abstained on.

The consumer's contract is a `Policy`: `kind_of` (its field map), `material_of`, `question_of`,
`pre` (its own stronger match, or a field definition passed as a hint) and `allowed`
(`decompose.enums(schema)`). `GenerateClaimJudge` renders the consumer's own claim pack with
`claim_slots` and calls the consumer's own `generate`, so its gate, trace and retry apply.

```{.python title="examples/claim_judge/main.py"}
--8<-- "examples/claim_judge/main.py"
```

## Example

```{.python title="examples/claim_grounding.py"}
--8<-- "examples/claim_grounding.py"
```
