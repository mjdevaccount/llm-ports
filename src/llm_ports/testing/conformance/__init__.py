"""Conformance suites: the contract every implementation of a port must keep.

A consumer that writes its own adapter (a wrapper, a second provider, a database-backed ledger)
runs the port's `check_*` in its own CI; the library runs the same checks against its reference
adapters and its stubs, so the stub cannot silently differ from the adapter. Each check is a
coroutine that returns the case names it checked and raises `ContractViolation` naming the first
case broken. `CHECKS` maps every runtime-checkable port to its check; a test proves it total.

Rationale: ADR-0016.
"""

from collections.abc import Callable, Mapping
from typing import Any

from llm_ports.testing._violation import ContractViolation
from llm_ports.testing.conformance._chat import TOOLS, check_chat, check_token_counter
from llm_ports.testing.conformance._generator import CHECK_MODEL, check_generator
from llm_ports.testing.conformance._learning import (
    check_example_store,
    check_model_registry,
    check_run_ledger,
    check_unavailable_model_registry,
    check_unavailable_run_ledger,
    check_unreadable_example_store,
)
from llm_ports.testing.conformance._ports import (
    check_citer,
    check_embedder,
    check_evaluator,
    check_gateway_reader,
    check_generation,
    check_grounder,
    check_judge,
    check_limiter,
    check_pricer,
    check_probe,
    check_prompt_source,
    check_reranker,
    check_retriever,
    check_tracer,
)

#: Port name to its conformance check.
CHECKS: Mapping[str, Callable[..., Any]] = {
    "Chat": check_chat,
    "Citer": check_citer,
    "Embedder": check_embedder,
    "Evaluator": check_evaluator,
    "ExampleStore": check_example_store,
    "GatewayReader": check_gateway_reader,
    "Generation": check_generation,
    "Generator": check_generator,
    "Grounder": check_grounder,
    "Judge": check_judge,
    "Limiter": check_limiter,
    "ModelRegistry": check_model_registry,
    "Pricer": check_pricer,
    "Probe": check_probe,
    "PromptSource": check_prompt_source,
    "Reranker": check_reranker,
    "Retriever": check_retriever,
    "RunLedger": check_run_ledger,
    "TokenCounter": check_token_counter,
    "Tracer": check_tracer,
}

__all__ = [
    "CHECKS",
    "CHECK_MODEL",
    "TOOLS",
    "ContractViolation",
    "check_chat",
    "check_citer",
    "check_embedder",
    "check_evaluator",
    "check_example_store",
    "check_gateway_reader",
    "check_generation",
    "check_generator",
    "check_grounder",
    "check_judge",
    "check_limiter",
    "check_model_registry",
    "check_pricer",
    "check_probe",
    "check_prompt_source",
    "check_reranker",
    "check_retriever",
    "check_run_ledger",
    "check_token_counter",
    "check_tracer",
    "check_unavailable_model_registry",
    "check_unavailable_run_ledger",
    "check_unreadable_example_store",
]
