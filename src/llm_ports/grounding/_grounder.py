"""`NumberLawGrounder`: a `Grounder` that applies the number law (`numbers.uncited`).

Every figure in a claim must appear in a cited span. It judges numbers only, a necessary check
and never a sufficient one.
"""

from __future__ import annotations

from collections.abc import Sequence

from llm_ports.grounding import numbers
from llm_ports.ports import Chunk, Claim, GroundingVerdict


class NumberLawGrounder:
    """A `Grounder` that applies the number law (`numbers.uncited`).

    Every digit run and spelled number in the claim must appear in a cited span.

    Invariants:
        - It judges numbers only: a necessary check, never a sufficient one. A numberless claim
          passes with reason `no_numbers` because this checker has nothing to say about it, not
          because a span entails it. Entailment is another `Grounder` behind the same port.
        - No spans means unsupported, always.
        - `score` is None: the verdict has no scale.

    Rationale: ADR-0003.
    """

    checker = "number_law"

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        """Return supported only when every number in the claim appears in one of `spans`."""
        if not spans:
            return GroundingVerdict(False, "no evidence spans", checker=self.checker)
        missing = numbers.uncited(claim.text, ((s.title, s.text) for s in spans))
        if missing:
            return GroundingVerdict(False, "number_not_in_span: " + ", ".join(missing[:4]), checker=self.checker)
        if not (numbers.digits(claim.text) or numbers.spelled(claim.text)):
            return GroundingVerdict(True, "no_numbers: nothing for number_law to judge", checker=self.checker)
        return GroundingVerdict(True, "every number cited", checker=self.checker)


__all__ = ["NumberLawGrounder"]
