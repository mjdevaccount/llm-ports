"""Adapters over HTTP (httpx): a reranker, a grounder client, a health probe, and an index-ordered transport."""

from llm_ports._extras import extra

with extra("http"):
    from llm_ports.adapters._index_order import INDEXED_LISTS, IndexOrderedTransport
from llm_ports.adapters.http import grounder, probe, reranker
from llm_ports.adapters.http.grounder import GrounderUnavailable, HttpGrounder
from llm_ports.adapters.http.probe import PROBE_DEADLINE_S, HttpProbe
from llm_ports.adapters.http.reranker import HttpReranker

__all__ = [
    "INDEXED_LISTS",
    "PROBE_DEADLINE_S",
    "GrounderUnavailable",
    "HttpGrounder",
    "HttpProbe",
    "HttpReranker",
    "IndexOrderedTransport",
    "grounder",
    "probe",
    "reranker",
]
