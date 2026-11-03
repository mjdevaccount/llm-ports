"""chunking -- split a document for embedding, and decide whether it needs re-embedding at all.

    chunks = chunk_text(doc, header="Source, 2026-06-02, 'title': ")
    if should_ingest(stored_hash, content_hash(header + doc)): ...   # else skip: nothing changed
    budget = max_chars_for_chunks(12)    # the longest text GUARANTEED to fit 12 chunks

Pure and deterministic. `chunk_text` cuts ~`size`-char windows with `overlap` chars of continuity,
preferring a paragraph break, then a line break, then a sentence end, but only inside the TAIL
THIRD of the window (`break_floor`), so a chunk stays near its target size and rarely cuts
mid-thought. `max_chars_for_chunks` is the exact worst case of that walk -- derived from the same
floor, so a caller's char budget for "at most n chunks" can never drift from the chunker.
"""

from __future__ import annotations

from llm_ports.packs import sha

CHUNK_CHARS = 2400
CHUNK_OVERLAP = 300
_SEPARATORS = ("\n\n", "\n", ". ")


def content_hash(text: str) -> str:
    """Full sha256 hex of the text -- the change detector (same hash = skip the re-embed)."""
    return sha(text, n=64)


def should_ingest(existing_hash: str | None, new_hash: str) -> bool:
    """The skip-unchanged rule: ingest when the doc is new or its content changed."""
    return existing_hash != new_hash


def break_floor(size: int) -> int:
    """`chunk_text` breaks only at or after this offset in a window: its tail third."""
    return size * 2 // 3


def max_chars_for_chunks(n: int, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> int:
    """The longest text `chunk_text` is GUARANTEED to split into <= n chunks (no header). Worst case
    every window breaks at the floor on a 1-char separator, so each chunk advances the start by
    floor + 1 - overlap; the last window then covers `size`. Tight: an adversarial text one char
    longer can need n + 1."""
    return size + (n - 1) * (break_floor(size) + 1 - overlap)


def chunk_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP, header: str = "") -> list[str]:
    """Split into ~`size`-char chunks with `overlap` chars of continuity, preferring paragraph
    boundaries (\\n\\n), then line breaks, then sentence ends, inside the tail third of the window.
    A text <= size is a single chunk; a blank one is no chunks. `header` is prepended to EVERY
    chunk so no chunk floats free of its provenance; chunks may exceed `size` by its length."""
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
        start = max(end - overlap, start + 1)  # overlap for continuity; always advance
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
