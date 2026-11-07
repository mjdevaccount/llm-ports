"""Grade a voice, calibrate it, learn its hedge words, and pool it with another.

Standard library only: the fitting libraries (scikit-learn, venn-abers, River, scoringrules) sit
behind adapters, and what they fit is applied here without them.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from llm_ports.forecast import Forecast, HedgeMap, LogitPool, PiecewiseLinear
from llm_ports.forecast.score import ece, reliability, score

rng = random.Random(1)

# 1. A forecast is a record of what a voice said, kept with its evidence.
said = Forecast(
    producer="desk",
    subject="SPY",
    kind="range",
    asserted_at=datetime(2026, 3, 2, 14, 30, tzinfo=UTC),
    resolves_at=datetime(2026, 3, 2, 21, 0, tzinfo=UTC),
    evidence="SPY holds 560-566 into the close",
    p=0.68,
)

# 2. Score it. This voice states 0.68 for every range and the ranges hold 38% of the time.
history = [(said.p or 0.0, rng.random() < 0.38) for _ in range(2000)]
raw = score(history)
assert raw.overconfidence is not None and raw.overconfidence > 0.25
assert raw.brier_skill is not None and raw.brier_skill < 0  # worse than stating the base rate
print(f"raw: n={raw.n} claims {raw.mean_p:.2f} hits {raw.hit_rate:.2f} brier {raw.brier:.3f}")

# 3. Calibrate. A fitted isotonic map (adapters.sklearn) is a PiecewiseLinear; stored as JSON,
#    it is applied with no fitting library. Here it is the map such a fit returns for this voice.
fitted = PiecewiseLinear.from_dict({"xs": [0.0, 0.68, 1.0], "ys": [0.0, 0.38, 0.38], "n": 2000})
calibrated = score((fitted(p), y) for p, y in history)
assert calibrated.brier is not None and raw.brier is not None and calibrated.brier < raw.brier
assert abs(fitted(said.p or 0.0) - 0.38) < 1e-9
print(f"calibrated: states {fitted(0.68):.2f}, brier {calibrated.brier:.3f}")

# 4. Verbal confidence. "likely" from author a has meant ~0.75; from author b ~0.45.
words = [("a", "likely", rng.random() < 0.75) for _ in range(300)]
words += [("b", "Likely", rng.random() < 0.45) for _ in range(300)]
words += [("c", "likely", True), ("c", "likely", True)]  # two lucky calls
hedges = HedgeMap.fit(words)
pa, pb, pc = hedges.p("a", "likely"), hedges.p("b", "likely"), hedges.p("c", "likely")
assert pa is not None and pb is not None and pc is not None
assert pa > pb and pc < 0.9  # c is shrunk toward the pool, not trusted on two calls
print(f"'likely': a={pa:.2f} b={pb:.2f} c={pc:.2f} (pooled {hedges.pooled['likely']:.2f})")

# 5. Pool two voices in log-odds; the equal-weight pool is the baseline a learned combiner
#    (adapters.river) has to beat.
pool = LogitPool()
both = pool.combine({"desk": fitted(0.68), "author-a": pa})
assert both is not None and min(fitted(0.68), pa) < both < max(fitted(0.68), pa)
print(f"pooled: {both:.2f}; ece of the calibrated voice {ece(reliability((fitted(p), y) for p, y in history)):.3f}")
assert said.resolves_at - said.asserted_at < timedelta(days=1)
