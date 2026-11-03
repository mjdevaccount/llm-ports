"""LettuceDetectGrounder -- LettuceDetect (token-span hallucination detector) behind the `Grounder` port.

Chosen over the number law alone in a spike: it agrees with the number law on every numeric miss,
also rejects paraphrase/structure claims the law is blind to, returns the offending TOKENS with a
confidence each, and reads 4K tokens. ModernBERT-base on CPU: ~16 s warm load, ~1.4 s per claim --
a validator's cost, not a hot path's.

Dependency: the `lettucedetect` extra (`llm-ports[lettucedetect]`: torch + transformers + the
model, hundreds of MB). The import is LAZY and the detector is INJECTABLE, so the port's shape is
graded anywhere with a fake (`llm_ports.testing.StubDetector`) and the real model only where it is
installed. `score` = 1 - the worst span's confidence: this adapter's own scale, never compared
with another checker's; None when any flagged span carries no confidence (unknown is never zero --
1.0 would read as fully grounded).

What fails silently here: a context past the model's window. 4,096 tokens is the model's cap and
nothing measured what happens past it, so this adapter names it: a context over MAX_CONTEXT_CHARS
(a char proxy) is still judged, and the verdict's reason starts with `context_over_window` so a
reader can see the judgement rested on a truncated premise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from llm_ports.ports import Chunk, Claim, GroundingVerdict

MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
# ~4,096 tokens of English prose at ~3 chars/token. A proxy, deliberately conservative.
MAX_CONTEXT_CHARS = 12_000
WINDOW_REASON = "context_over_window"
_SPAN_PREVIEW = 40


class Detector(Protocol):
    """The slice of lettucedetect.models.inference.HallucinationDetector this adapter uses."""

    def predict(self, *, context: list[str], answer: str,
                output_format: str) -> list[dict[str, Any]]: ...


def load_detector(model: str = MODEL) -> Detector:
    """The real model. Raises a RuntimeError naming the missing extra when lettucedetect is not
    installed, so a caller sees 'install this' and never a bare ImportError."""
    try:
        from lettucedetect.models.inference import HallucinationDetector
    except ImportError as exc:
        raise RuntimeError("lettucedetect is not installed: install the `llm-ports[lettucedetect]` extra "
                           "(or the consumer's dependency group that carries lettucedetect)") from exc
    return HallucinationDetector(method="transformer", model_path=model)


class LettuceDetectGrounder:
    """`Grounder` over LettuceDetect token spans: a claim is supported when the model flags NO
    span of it as unsupported by the cited chunks' text. No spans -> unsupported, always."""

    checker = "lettucedetect"

    def __init__(self, detector: Detector | None = None, *, model: str = MODEL) -> None:
        self._detector = detector if detector is not None else load_detector(model)

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        if not spans:
            return GroundingVerdict(False, "no evidence spans", checker=self.checker)
        # Title joins the text: the number law (`numbers.uncited`) counts a digit in a span's TITLE
        # as cited, so two Grounders behind one port must read the same premise. `title: text` and
        # not a newline or a second passage -- MEASURED on the real model: both of those added a
        # weak false flag (`%`, confidence ~0.55) on the CITED claim; the prefix form left every
        # verdict as text-only.
        context = [f"{s.title}: {s.text}" if s.title else s.text for s in spans]
        found = self._detector.predict(context=context, answer=claim.text, output_format="spans")
        # UNKNOWN IS NEVER ZERO: a flagged span with no confidence must not score 1.0 (this
        # scale's 'fully supported' end). No score at all is the honest reading; `supported`
        # still carries the gate.
        confs = [f.get("confidence") for f in found]
        score = (None if any(c is None for c in confs)
                 else round(1.0 - max((float(c) for c in confs), default=0.0), 4))
        why = ("; ".join(str(f.get("text") or "")[:_SPAN_PREVIEW] for f in found[:3])
               if found else "no unsupported span")
        n_chars = sum(len(c) for c in context)
        if n_chars > MAX_CONTEXT_CHARS:
            why = f"{WINDOW_REASON}: {n_chars} chars; {why}"
        return GroundingVerdict(not found, why, score=score, checker=self.checker)


__all__ = ["MAX_CONTEXT_CHARS", "MODEL", "WINDOW_REASON", "Detector", "LettuceDetectGrounder", "load_detector"]
