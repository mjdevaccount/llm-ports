"""Split a document for embedding, and decide whether it needs re-embedding at all.

Example:
    chunks = chunk_text(doc, header="Handbook, 'Returns policy': ")
    if should_ingest(stored_hash, content_hash(header + doc)): ...  # else skip: nothing changed
    budget = max_chars_for_chunks(12)  # the longest text guaranteed to fit 12 chunks

Pure and deterministic. `chunk_text` cuts windows of about `size` characters with `overlap`
characters of continuity. It prefers a paragraph break, then a line break, then a sentence end,
but only inside the tail third of the window (`break_floor`), so a chunk stays near its target
size and rarely cuts mid-thought. `max_chars_for_chunks` is the exact worst case of that walk,
derived from the same floor, so a caller's character budget for "at most n chunks" cannot drift
from the chunker.
"""

from __future__ import annotations

from llm_ports.packs import sha

CHUNK_CHARS = 2400
CHUNK_OVERLAP = 300
_SEPARATORS = ("\n\n", "\n", ". ")


def content_hash(text: str) -> str:
    """Return the full sha256 hex of the text: the change detector (same hash, skip the re-embed)."""
    return sha(text, n=64)


def should_ingest(existing_hash: str | None, new_hash: str) -> bool:
    """Return whether to ingest: the document is new or its content changed."""
    return existing_hash != new_hash


def break_floor(size: int) -> int:
    """Return the offset in a window at or after which `chunk_text` may break: its tail third."""
    return size * 2 // 3


def max_chars_for_chunks(n: int, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> int:
    """Return the longest header-free text `chunk_text` is guaranteed to split into at most `n` chunks.

    In the worst case every window breaks at the floor on a one-character separator, so each chunk
    advances the start by `floor + 1 - overlap`; the last window then covers `size`. The bound is
    tight: an adversarial text one character longer can need `n + 1` chunks.
    """
    return size + (n - 1) * (break_floor(size) + 1 - overlap)


def chunk_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP, header: str = "") -> list[str]:
    """Split text into chunks of about `size` characters with `overlap` characters of continuity.

    A break prefers a paragraph boundary (a blank line), then a line break, then a sentence end,
    inside the tail third of the window. A text no longer than `size` is a single chunk; a blank one
    is no chunks. `header` is prepended to every chunk so no chunk floats free of its provenance; a
    chunk may exceed `size` by the header's length.
    """
    text = text.strip()
    if not text:
        return []
    if header:
        return [header + c for c in chunk_text(text, size=size, overlap=overlap)]
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    floor = break_floor(size)
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in _SEPARATORS:
                pos = window.rfind(sep)
                if pos >= floor:
                    end = start + pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # overlap for continuity, and always advance
    return chunks


__all__ = [
    "CHUNK_CHARS",
    "CHUNK_OVERLAP",
    "break_floor",
    "chunk_text",
    "content_hash",
    "max_chars_for_chunks",
    "should_ingest",
]
