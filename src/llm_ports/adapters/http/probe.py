"""The `Probe` port over HTTP: `GET` an OpenAI-wire server's `/v1/models` under a 5 s deadline.

`up` lists the model ids the server serves; `down` is a backend that could not be reached, timed
out, or answered with an error status; `unknown` is a probe that failed itself. Richer capacity
(GPU state, model swaps) stays with the consumer.

Example:
    probe = HttpProbe("http://localhost:4000/v1/models")
    health = await probe.probe()
    if health.state == "down": ...  # open the breaker; `unknown` does not

Rationale: ADR-0022.
"""

from llm_ports._extras import extra

with extra("http"):
    from llm_ports.adapters._probe import PROBE_DEADLINE_S, HttpProbe

__all__ = ["PROBE_DEADLINE_S", "HttpProbe"]
