"""A loop's run linked to a registered model version, promoted by alias, on swappable backends.

The run ledger ids the row it writes (`Written.ref`); the fit is registered as a numbered model
version that points back at that row (`run_ref`); promotion is moving the `champion` alias. Every
piece here is a port with a standard-library backend, and each backend passes the conformance
suite (`llm_ports.testing.conformance`), the check a bigger engine's adapter (MLflow:
`llm_ports.adapters.mlflow`) passes too. Swapping engines is changing the two constructors.
"""

import asyncio
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_ports.learn import Clause, Fit, Verdict, gate, run
from llm_ports.learn.registry import JsonlModelRegistry
from llm_ports.learn.stores import JsonlRunLedger
from llm_ports.testing.conformance import check_model_registry, check_run_ledger


class MeanLoop:
    """Learn a mean; the gate asks for at least three points."""

    kind, subject = "mean", "demo"
    params: Mapping[str, Any] = {"window": 3}

    async def evidence(self) -> list[float]:
        return [1.0, 2.0, 4.5]

    def fit(self, xs: list[float]) -> Fit:
        return Fit({"mean": sum(xs) / len(xs)}, {"n": len(xs)})

    def clauses(self, fit: Fit, xs: list[float]) -> Sequence[Clause]:
        return [gate.at_least("n", fit.metrics["n"], 3, role="floor")]

    async def apply(self, verdict: Verdict, fit: Fit) -> str:
        return "registered" if verdict.passed else "held"


async def main(root: Path) -> None:
    ledger = JsonlRunLedger(root / "runs.jsonl")  # or MlflowRunLedger(MlflowClient(...))
    registry = JsonlModelRegistry(root / "models.jsonl")  # or MlflowModelRegistry(MlflowClient(...))

    got = await run(MeanLoop(), ledger)
    assert got.verdict.passed and got.fit is not None and got.recorded.ref

    version = await registry.register("mean-demo", model=got.fit.model, run_ref=got.recorded.ref)
    assert version.ok and version.ref == "1"
    assert (await registry.set_alias("mean-demo", "champion", int(version.ref))).ok

    champion = (await registry.by_alias("mean-demo", "champion")).version
    assert champion is not None and champion.model == {"mean": 2.5}
    row = (await ledger.latest("mean")).row
    assert row is not None and row.ref == champion.run_ref  # the champion's lineage, one hop

    # The outside witness: both backends keep their port's contract.
    assert len(await check_run_ledger(ledger)) == 9
    assert len(await check_model_registry(registry)) == 9
    print(f"champion v{champion.version} mean={champion.model['mean']} from run {champion.run_ref}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(main(Path(tmp)))
