"""Retrieval plumbing: chunk, skip what did not change, embed, rerank, and score the ranking.

`StubEmbedder` and `StubReranker` stand in for models (real ones: `PydanticAIEmbedder` and
`HttpReranker`). The metrics read a ranking against an any-of relevant set, which is how a
parity gate compares a new retriever with the one it would replace.
"""

import asyncio

from llm_ports.rag import chunking, metrics
from llm_ports.testing import StubEmbedder, StubReranker

DOCS = {
    "billing": "Refunds post within five business days. A duplicate charge is reversed automatically.",
    "shipping": "Orders ship from the nearest warehouse. Tracking arrives by email once the parcel leaves.",
    "returns": "Return any item within thirty days. A return label is printed from the order page.",
}


async def main() -> None:
    stored: dict[str, str] = {}  # doc id -> content hash of what was last embedded
    chunks: dict[str, str] = {}
    for doc_id, text in DOCS.items():
        digest = chunking.content_hash(text)
        if not chunking.should_ingest(stored.get(doc_id), digest):
            continue  # unchanged: no re-embed
        stored[doc_id] = digest
        for i, chunk in enumerate(chunking.chunk_text(text, size=60, overlap=10, header=doc_id)):
            chunks[f"{doc_id}:{i}"] = chunk

    embedded = await StubEmbedder(dims=16, stamp="stub@16").embed(list(chunks.values()))
    print(embedded.outcome, len(embedded.vectors), "vectors in space", embedded.stamp)

    query = "how long does a refund take"
    ranked = await StubReranker().rerank(query, list(chunks.values()))
    order = [cid for _, cid in sorted(zip(ranked.scores, chunks, strict=True), key=lambda p: -p[0])]
    relevant = {cid for cid in chunks if cid.startswith("billing:")}
    print("first relevant at rank", metrics.first_rank(relevant, order))
    print("hit@3", metrics.hit_at_k(relevant, order, 3), "MRR", metrics.reciprocal_rank(relevant, order))


asyncio.run(main())
