# Named outcomes

A fallback policy is a `match` on `res.outcome`: retry the same role on another tier when the first is `unavailable`, stop when a cap says `gated`, and treat the rest as unusable. No `try` block appears, because `generate` never raises for a model, wire or gate failure.

The outcome is coarse on purpose (seven members; `refused` is the newest). The detail is `res.failure.kind`: a policy that must tell a spent quota from a full queue reads `kind` (`quota` is never `retryable`, `rate_limited` is, with `retry_after_s`), and a caller bug is `internal` under `request_fault`, never an outage (ADR-0020).

```{.python title="examples/named_outcomes.py"}
--8<-- "examples/named_outcomes.py"
```

Source: [`examples/named_outcomes.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/named_outcomes.py)
