"""Test doubles for the library's ports, so a consumer's tests do not hand-roll their own.

- `StubGenerator` fakes `Generator` with a reply function and records every request.
- `RecordingTracer` fakes `Tracer` and records every generation, its end and its scores.
- `StubEmbedder` fakes `Embedder` with deterministic unit vectors hashed from each text, and
  records every call.
- `StubReranker` fakes `Reranker` with a predictable word-overlap score, and records every call.
- `StubCiter` and `StubPromptSource` fake the `Citer` and `PromptSource` seams with no model and
  no store.
- `StubDetector` fakes the LettuceDetect model under `adapters.lettucedetect.LettuceDetectGrounder`.

Invariants:
    - Every port double passes its port's `isinstance` check.
    - No double performs I/O.

Example:
    gen = StubGenerator(lambda req, output: {"category": "groceries"})
    tracer = RecordingTracer()
    res = await call.generate(ROLE, prompt="...", output=Reply, generator=gen,
                              hooks=Hooks(trace=tracer_trace(tracer)))
    assert gen.requests[0].model == ROLE.resolved_model
    assert tracer.generations[0].ended["usage"] is None
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_ports.ports import (
    Chunk,
    CitedAnswer,
    Embedded,
    EmbedInput,
    Generated,
    GenRequest,
    Outcome,
    Prompt,
    Reranked,
)
from llm_ports.ports.outcomes import failure_reason

Reply = Callable[[GenRequest, type[Any]], Any]


class StubGenerator:
    """A `Generator` that answers from a reply function and records every request.

    The reply function's result maps onto a `Generated`: a plain value is wrapped as `ok`, a
    `Generated` is returned as-is (any outcome), and a raised exception becomes `unavailable`.

    Args:
        reply: A function of `(request, output type)`, or a non-callable value returned for every
            request.

    Attributes:
        requests: Every `GenRequest` received, in call order.

    Invariants:
        - `generate` never raises.
        - `validate` runs on an `ok` value and its problems are kept on the result; there is no
          re-ask.
    """

    def __init__(self, reply: Reply | Any) -> None:
        self._reply: Reply = reply if callable(reply) else (lambda _req, _out: reply)
        self.requests: list[GenRequest] = []

    async def generate[T](
        self,
        req: GenRequest,
        output: type[T],
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> Generated[T]:
        """Return the reply function's answer to `req` as a `Generated`; `retry_message` is unused."""
        del retry_message
        self.requests.append(req)
        try:
            got = self._reply(req, output)
        except Exception as exc:
            return Generated(
                value=None, outcome="unavailable", model=req.model, reason=failure_reason(exc), retries=None
            )
        if isinstance(got, Generated):
            return got
        problems = tuple(validate(got)) if validate is not None else ()
        return Generated(value=got, outcome="ok", model=req.model, problems=problems)


@dataclass
class RecordedGeneration:
    """One `start_generation` call: what it was opened with, how it ended and how it was scored.

    Attributes:
        opened: The keyword arguments `start_generation` received.
        ended: The keyword arguments of `end`, or None while the generation is open.
        scores: Every `(name, value, comment)` passed to `score`, in call order.
    """

    opened: dict[str, Any]
    ended: dict[str, Any] | None = None
    scores: list[tuple[str, float, str | None]] = field(default_factory=list)

    def end(
        self,
        *,
        output: Any = None,
        usage: Mapping[str, int | None] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record how the generation ended, replacing any earlier end."""
        self.ended = {
            "output": output,
            "usage": usage,
            "level": level,
            "status_message": status_message,
            "metadata": metadata,
        }

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """Record one score."""
        self.scores.append((name, value, comment))


class RecordingTracer:
    """A `Tracer` that records every generation.

    Attributes:
        generations: Every `RecordedGeneration`, in open order.
    """

    def __init__(self) -> None:
        self.generations: list[RecordedGeneration] = []

    def start_generation(
        self,
        *,
        name: str,
        model: str,
        provider: str,
        input: Any,
        metadata: Mapping[str, Any],
        session_id: str | None,
        prompt_name: str | None,
    ) -> RecordedGeneration:
        """Open, record and return one generation."""
        g = RecordedGeneration(
            opened={
                "name": name,
                "model": model,
                "provider": provider,
                "input": input,
                "metadata": dict(metadata),
                "session_id": session_id,
                "prompt_name": prompt_name,
            }
        )
        self.generations.append(g)
        return g


class StubEmbedder:
    """An `Embedder` with no wire: each text hashes to a deterministic unit vector.

    Args:
        dims: The length of every vector.
        stamp: The embedding stamp reported on every result.
        fail: An outcome other than `ok`; when set, every call fails with it.

    Attributes:
        calls: `(texts, input_type)` for every call, in call order.

    Invariants:
        - The same text always maps to the same vector.
    """

    def __init__(self, *, dims: int = 8, stamp: str = "stub@8", fail: Outcome | None = None) -> None:
        self.dims = dims
        self._stamp = stamp
        self._fail = fail
        self.calls: list[tuple[tuple[str, ...], EmbedInput]] = []

    @property
    def stamp(self) -> str:
        """The embedding stamp reported on every result."""
        return self._stamp

    def vector(self, text: str) -> tuple[float, ...]:
        """Return the unit vector for `text`, derived from its sha256 digest."""
        raw = hashlib.sha256(text.encode()).digest()
        vals = [(raw[i % len(raw)] - 127.5) for i in range(self.dims)]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return tuple(v / norm for v in vals)

    async def embed(self, texts: Sequence[str], *, input_type: EmbedInput = "document") -> Embedded:
        """Return one vector per text, or fail with the configured outcome."""
        self.calls.append((tuple(texts), input_type))
        if self._fail:
            return Embedded(
                vectors=(),
                outcome=self._fail,
                stamp=self._stamp,
                reason=f"stub {self._fail}",
            )
        return Embedded(
            vectors=tuple(self.vector(t) for t in texts), outcome="ok", stamp=self._stamp, input_tokens=None
        )


class StubReranker:
    """A `Reranker` with no model: a document scores the count of its words found in the query.

    Words are case-folded, so a test can predict the order.

    Args:
        model: The model name reported on every result.
        fail: An outcome other than `ok`; when set, every call fails with it.

    Attributes:
        calls: `(query, documents)` for every call, in call order.
    """

    def __init__(self, *, model: str = "stub-reranker", fail: Outcome | None = None) -> None:
        self.model = model
        self._fail = fail
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def rerank(self, query: str, documents: Sequence[str]) -> Reranked:
        """Score each document by word overlap with `query`, or fail with the configured outcome."""
        self.calls.append((query, tuple(documents)))
        if self._fail:
            return Reranked(
                scores=(),
                outcome=self._fail,
                model=self.model,
                reason=f"stub {self._fail}",
            )
        words = set(query.lower().split())
        return Reranked(
            scores=tuple(float(sum(w in words for w in d.lower().split())) for d in documents),
            outcome="ok",
            model=self.model,
        )


class StubCiter:
    """A `Citer` that asserts nothing.

    Every chunk id is known to the answer, and the answer has no claims. `model` is the empty
    string, since a stub has no model.
    """

    async def answer(self, q: str, chunks: Sequence[Chunk]) -> CitedAnswer:
        """Return an answer to `q` with no claims, keyed over `chunks`."""
        return CitedAnswer(question=q, claims=(), chunks={c.id: c for c in chunks}, model="")


class StubPromptSource:
    """A `PromptSource` with no managed store, serving fixed in-memory texts.

    Each name serves its text as the repo seed (`version` None, `source` "repo"). The requested
    label is recorded in `config["label"]`, so a test can see it was passed through.

    Args:
        texts: The text served for each prompt name.
    """

    def __init__(self, texts: Mapping[str, str]) -> None:
        self._texts = dict(texts)

    def get(self, name: str, *, label: str = "production") -> Prompt:
        """Return the fixed text for `name` as a repo-seed `Prompt`.

        Raises:
            KeyError: `name` was not given. The stub never invents text: with no seed, the port
                has nothing to serve.
        """
        return Prompt(name, self._texts[name], None, "repo", config={"label": label})


class StubDetector:
    """An `adapters.lettucedetect.Detector` with no model.

    It flags every digit run in the answer that no context string contains, in LettuceDetect's
    span shape with confidence 0.9: a stand-in for the number law.

    Attributes:
        calls: The keyword arguments of every `predict` call, in call order.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, *, context: list[str], answer: str, output_format: str) -> list[dict[str, Any]]:
        """Return one span for each digit run in `answer` that no `context` string contains."""
        self.calls.append({"context": context, "answer": answer, "output_format": output_format})
        joined = " ".join(context)
        return [
            {"start": m.start(), "end": m.end(), "text": m.group(0), "confidence": 0.9}
            for m in _DIGITS.finditer(answer)
            if m.group(0) not in joined
        ]


_DIGITS = re.compile(r"\d+(?:\.\d+)?")


__all__ = [
    "RecordedGeneration",
    "RecordingTracer",
    "Reply",
    "StubCiter",
    "StubDetector",
    "StubEmbedder",
    "StubGenerator",
    "StubPromptSource",
    "StubReranker",
]
