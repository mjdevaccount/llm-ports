"""chunking: the paragraph-aware overlapping chunker, its tight worst-case bound, and the hash-skip
rule. Ported from trade-data tests/test_rag_text.py (the RAG index's chunker moved here, v0.9.0)."""

from __future__ import annotations

import hashlib

from llm_ports.chunking import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    break_floor,
    chunk_text,
    content_hash,
    max_chars_for_chunks,
    should_ingest,
)

THREE_PARAS = "A" * 1000 + "\n\n" + "B" * 1000 + "\n\n" + "C" * 1000


def test_empty_and_blank_are_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []
    assert chunk_text("", header="Source — 'x': ") == []


def test_a_small_doc_is_one_chunk_and_exactly_size_is_still_one() -> None:
    text = "short doc " * 10
    assert chunk_text(text) == [text.strip()]
    assert chunk_text("x" * CHUNK_CHARS) == ["x" * CHUNK_CHARS]


def test_the_split_prefers_a_paragraph_boundary() -> None:
    chunks = chunk_text(THREE_PARAS, size=2400, overlap=300)
    assert len(chunks) == 2
    # the split lands on the second \n\n (in the tail third of the window), not mid-C
    assert chunks[0].endswith("B") and chunks[1].endswith("C")


def test_the_overlap_carries_continuity() -> None:
    chunks = chunk_text(THREE_PARAS, size=2400, overlap=300)
    assert chunks[1].startswith("B" * 100) and chunks[1][:100] in chunks[0]


def test_no_separator_still_advances_and_covers() -> None:
    chunks = chunk_text("x" * 10_000, size=2400, overlap=300)
    assert all(len(c) <= 2400 for c in chunks)
    assert sum(len(c) for c in chunks) >= 10_000  # overlap => total >= source; nothing dropped


def test_a_separator_before_the_tail_third_is_not_a_break() -> None:
    early = "a" * 100 + "\n\n" + "b" * 3000          # the only paragraph break is at offset 100
    chunks = chunk_text(early, size=2400, overlap=300)
    assert len(chunks[0]) == 2400, "a break before break_floor would have cut a 102-char chunk"


def test_the_header_prefixes_every_chunk_and_changes_no_split() -> None:
    header = "Daily feed, 2026-06-02, 'levels': "
    chunks = chunk_text(THREE_PARAS, size=2400, overlap=300, header=header)
    assert len(chunks) == 2 and all(c.startswith(header) for c in chunks)
    assert [c[len(header):] for c in chunks] == chunk_text(THREE_PARAS, size=2400, overlap=300)


def test_deterministic() -> None:
    text = ("para one text. " * 40 + "\n\n") * 20
    assert chunk_text(text) == chunk_text(text)


# ---- the worst-case char bound ----


def _worst(length: int, size: int, overlap: int) -> str:
    """The text that drives chunk_text's advance to its minimum: a one-char separator exactly at
    each window's break floor, and nothing else it may break on, so every window cuts at
    floor + 1 and the next starts `overlap` back."""
    floor = break_floor(size)
    chars = ["x"] * length
    for pos in range(floor, length, floor + 1 - overlap):
        chars[pos] = "\n"
    return "".join(chars)


def test_the_bound_is_reached_and_never_passed_and_is_tight() -> None:
    """MUST FIRE: a looser bound (e.g. `n * size`) fails the first assert's sibling below, a tighter
    one fails `== cap`."""
    for size, overlap in ((CHUNK_CHARS, CHUNK_OVERLAP), (1000, 150), (2400, 0)):
        for cap in (1, 2, 5, 12):
            bound = max_chars_for_chunks(cap, size=size, overlap=overlap)
            assert len(chunk_text(_worst(bound, size, overlap), size=size, overlap=overlap)) == cap
            # tight: one char more needs cap + 1, so the bound is not loose either
            assert len(chunk_text(_worst(bound + 1, size, overlap), size=size, overlap=overlap)) == cap + 1


def test_the_hand_number_a_bound_replaces_was_far_looser() -> None:
    cap = 12
    assert len(chunk_text(_worst(cap * CHUNK_CHARS, CHUNK_CHARS, CHUNK_OVERLAP))) > cap


def test_the_defaults_are_the_bound_defaults() -> None:
    assert max_chars_for_chunks(3) == max_chars_for_chunks(3, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP)


# ---- hash-skip ----


def test_content_hash_is_the_full_sha256_hex_stable_and_sensitive() -> None:
    a = "doc text"
    assert content_hash(a) == hashlib.sha256(a.encode("utf-8")).hexdigest()   # continuity: stored hashes
    assert content_hash(a) == content_hash(a) and content_hash(a) != content_hash(a + " changed")


def test_should_ingest_new_changed_unchanged() -> None:
    h = content_hash("doc")
    assert should_ingest(None, h) is True          # new doc
    assert should_ingest("otherhash", h) is True   # changed doc
    assert should_ingest(h, h) is False            # unchanged -> skip (the idempotency)
