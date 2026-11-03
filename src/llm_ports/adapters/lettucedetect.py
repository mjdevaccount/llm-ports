"""LettuceDetect, a token-span hallucination detector, behind the `Grounder` port.

The detector returns the offending tokens of a claim with a confidence each, and reads a context of
up to 4,096 tokens. It costs what a validator may, not what a hot path may.

The `lettucedetect` extra (`llm-ports[lettucedetect]`: torch, transformers and the model, hundreds
of MB) is imported lazily, and the detector is injectable, so the port's shape is tested anywhere
with a fake (`llm_ports.testing.StubDetector`) and the real model only where it is installed.

`score` is 1 minus the worst flagged span's confidence. It is this adapter's own scale, never
compared with another checker's.

Invariants:
    - `score` is None when any flagged span carries no confidence: unknown is never zero, and 1.0
      would read as fully grounded.
    - A context longer than `MAX_CONTEXT_CHARS` (a character proxy for the model's 4,096-token
      window) is still judged, and the verdict's reason starts with `WINDOW_REASON`, so a reader
      can see the judgement rested on a truncated premise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from llm_ports.ports import Chunk, Claim, GroundingVerdict

MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
# About 4,096 tokens of English prose at ~3 characters per token; a deliberately conservative proxy.
MAX_CONTEXT_CHARS = 12_000
WINDOW_REASON = "context_over_window"
_SPAN_PREVIEW = 40


class Detector(Protocol):
    """The slice of `lettucedetect.models.inference.HallucinationDetector` this adapter uses."""

    def predict(self, *, context: list[str], answer: str, output_format: str) -> list[dict[str, Any]]:
        """Return the spans of `answer` that `context` does not support."""
        ...


def load_detector(model: str = MODEL) -> Detector:
    """Load the real LettuceDetect model.

    Raises:
        RuntimeError: lettucedetect is not installed. The message names the extra to install, so a
            caller never sees a bare ImportError.
    """
    try:
        from lettucedetect.models.inference import HallucinationDetector
    except ImportError as exc:
        raise RuntimeError(
            "lettucedetect is not installed: install the `llm-ports[lettucedetect]` extra "
            "(or the consumer's dependency group that carries lettucedetect)"
        ) from exc
    detector: Detector = HallucinationDetector(method="transformer", model_path=model)
    return detector


class LettuceDetectGrounder:
    """A `Grounder` over LettuceDetect token spans.

    A claim is supported when the model flags no span of it as unsupported by the cited chunks'
    text.

    Args:
        detector: The detector to use; None loads the real model named by `model`.
        model: The checkpoint `load_detector` loads when `detector` is None.

    Invariants:
        - A claim with no spans is unsupported.
    """

    checker = "lettucedetect"

    def __init__(self, detector: Detector | None = None, *, model: str = MODEL) -> None:
        self._detector = detector if detector is not None else load_detector(model)

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        """Return whether the text of `spans` supports `claim`."""
        if not spans:
            return GroundingVerdict(False, "no evidence spans", checker=self.checker)
        # The title joins the text, because the number law (`numbers.uncited`) counts a digit in a
        # span's title as cited and two Grounders behind one port must read the same premise. On
        # the real model, a newline or a second passage instead adds a weak false flag to a cited
        # claim; the `title: text` prefix does not.
        context = [f"{s.title}: {s.text}" if s.title else s.text for s in spans]
        found = self._detector.predict(context=context, answer=claim.text, output_format="spans")
        # Unknown is never zero: a flagged span with no confidence must not score 1.0, this
        # scale's fully-supported end. No score is the honest reading; `supported` still gates.
        confs = [f.get("confidence") for f in found]
        known = [float(c) for c in confs if c is not None]
        score = round(1.0 - max(known, default=0.0), 4) if len(known) == len(confs) else None
        why = "; ".join(str(f.get("text") or "")[:_SPAN_PREVIEW] for f in found[:3]) if found else "no unsupported span"
        n_chars = sum(len(c) for c in context)
        if n_chars > MAX_CONTEXT_CHARS:
            why = f"{WINDOW_REASON}: {n_chars} chars; {why}"
        return GroundingVerdict(not found, why, score=score, checker=self.checker)


__all__ = ["MAX_CONTEXT_CHARS", "MODEL", "WINDOW_REASON", "Detector", "LettuceDetectGrounder", "load_detector"]
