"""Adapters over HTTP (httpx): a reranker, a grounder client, and an index-ordered transport."""

from llm_ports._extras import extra

with extra("http"):
    from llm_ports.adapters._index_order import INDEXED_LISTS, IndexOrderedTransport
from llm_ports.adapters.http import grounder, reranker
from llm_ports.adapters.http.grounder import GrounderUnavailable, HttpGrounder
from llm_ports.adapters.http.reranker import HttpReranker

__all__ = [
    "INDEXED_LISTS",
    "GrounderUnavailable",
    "HttpGrounder",
    "HttpReranker",
    "IndexOrderedTransport",
    "grounder",
    "reranker",
]
