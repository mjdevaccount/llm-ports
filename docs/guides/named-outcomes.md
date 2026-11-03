# Named outcomes

A fallback policy is a `match` on `res.outcome`: retry the same role on another tier when the first is `unavailable`, stop when a cap says `gated`, and treat the rest as unusable. No `try` block appears, because `generate` never raises for a model, wire or gate failure.

```{.python title="examples/named_outcomes.py"}
--8<-- "examples/named_outcomes.py"
```

Source: [`examples/named_outcomes.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/named_outcomes.py)
