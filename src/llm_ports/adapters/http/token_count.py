"""The `TokenCounter` port over a gateway's count endpoint (LiteLLM's `/utils/token_counter` shape).

The implementation is shared with the pydantic-ai factory (`adapters._token_count`); this module is
its public home.

Example:
    counter = HttpTokenCounter("http://localhost:4000/utils/token_counter", models={"gpt-5-mini"})
    got = await counter.count("gpt-5-mini", system="", messages=history)

Rationale: ADR-0024.
"""

from llm_ports.adapters._token_count import COUNT_TIMEOUT_S, UNSUPPORTED_STATUS, HttpTokenCounter, count_payload

__all__ = ["COUNT_TIMEOUT_S", "UNSUPPORTED_STATUS", "HttpTokenCounter", "count_payload"]
