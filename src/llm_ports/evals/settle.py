"""Code's ruling on a labelled claim: arithmetic is code's last word, never a judge's.

A judge labels each claim and, for a `derived` one, names its op chain (`Claim.op`). `settle`
re-runs the chain with `grounding.derive.recheck` and rules only in the direction the evidence
carries:

- `breaks` makes the claim `contradicted`: a wrong figure is wrong whatever the words around it say.
- `holds` confirms a grounded label. It does not overturn an ungrounded one: the op certifies the
  figure, and a claim can state a true figure with a flipped direction or an invented cause.
- `abstain` and `unchecked` leave the claim to the judge, and so does a claim with no op.

The figures the number law refuses (`grounding.numbers.loose`) ride along as `refused`, a hint for
the judge's prompt, and never rule: free text renders a figure in ways the law does not read (a
date as "Nov 6", a decimal comma, a figure the prompt asked the model to estimate), so a refusal
alone does not make a claim ungrounded.

Invariants:
    - Pure: no model, no I/O. Never raises.
    - Only the recheck rules; `ruled` is False whenever it did not.

Rationale: ADR-0030.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from llm_ports.evals.claims import Claim, ClaimLabel
from llm_ports.grounding import numbers
from llm_ports.grounding.derive import SEPARATOR, Tolerance, Verdict, recheck


@dataclass(frozen=True)
class Settled:
    """Code's ruling on one claim.

    Attributes:
        claim_id: The claim's id.
        grounded: True or False when the recheck ruled; None when code has nothing to add.
        label: The label the recheck names, or None.
        verdict: The recheck verdict, or None when the claim had no op.
        reason: One line naming what code found.
        refused: The claim's figures the number law refuses: a hint for the judge, never a ruling.
    """

    claim_id: str
    grounded: bool | None
    label: ClaimLabel | None
    verdict: Verdict | None
    reason: str
    refused: tuple[str, ...] = ()

    @property
    def ruled(self) -> bool:
        """True when the recheck ruled on the claim."""
        return self.grounded is not None


def _bare(op: str) -> bool:
    """Return whether the op's expression is one fact id, the shape of a `supported` claim."""
    try:
        return isinstance(ast.parse(op.split(SEPARATOR)[0].strip(), mode="eval").body, ast.Name)
    except SyntaxError:
        return False


def settle(
    claim: Claim,
    *,
    allowed: set[str],
    words_ok: str = "",
    facts: Mapping[str, object] | None = None,
    band: float = 2.0,
) -> Settled:
    """Rule on `claim` where its op chain lets code; otherwise leave it to the judge.

    Args:
        claim: The labelled claim. Its `op`, and the `shown` and `tol` extra keys, drive the
            recheck.
        allowed: The Input's licensed digit runs (`grounding.numbers.digits`), for `refused`.
        words_ok: The Input text, for spelled numbers (`grounding.numbers.loose`).
        facts: The Input's facts by id, so the recheck also holds the op's expression to them.
        band: The recheck's abstain band, a multiple of the claim's tolerance.

    Returns:
        The `Settled` ruling. Never raises.
    """
    refused = tuple(numbers.loose(claim.quote, allowed, words_ok, rerender=True))
    if not claim.op:
        return Settled(claim.id, None, None, None, "no op chain; the label is the judge's", refused)
    tol_raw, shown_raw = claim.extra.get("tol"), claim.extra.get("shown")
    r = recheck(
        claim.op,
        shown_raw if isinstance(shown_raw, str) else None,
        tol=Tolerance.of(tol_raw) if isinstance(tol_raw, Mapping) else None,
        facts=facts,
        band=band,
    )
    if r.verdict == "breaks":
        return Settled(claim.id, False, "contradicted", r.verdict, r.reason, refused)
    if r.verdict == "holds" and claim.grounded:
        label: ClaimLabel = "supported" if _bare(claim.op) else "derived"
        return Settled(claim.id, True, label, r.verdict, r.reason, refused)
    why = "the figure holds; the label rests on the words" if r.verdict == "holds" else r.reason
    return Settled(claim.id, None, None, r.verdict, why, refused)


__all__ = ["Settled", "settle"]
