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
    author="desk",
    subject="SPY",
    kind="range",
    forecast_time=datetime(2026, 3, 2, 14, 30, tzinfo=UTC),
    resolve_time=datetime(2026, 3, 2, 21, 0, tzinfo=UTC),
    text="SPY holds 560-566 into the close",
    probability=0.68,
)

# 2. Score it. This voice states 0.68 for every range and the ranges hold 38% of the time.
stated = [said.probability or 0.0] * 2000
happened = [rng.random() < 0.38 for _ in stated]
raw = score(happened, stated)  # scoringrules order: observation first
assert raw.overconfidence is not None and raw.overconfidence > 0.25
assert raw.brier_skill_score is not None and raw.brier_skill_score < 0  # worse than stating the base rate
print(f"raw: n={raw.n} claims {raw.mean_p:.2f} hits {raw.hit_rate:.2f} brier {raw.brier_score:.3f}")

# 3. Calibrate. A fitted isotonic map (adapters.sklearn, `IsotonicCalibrator().fit(X, y).map_`)
#    is a PiecewiseLinear; stored as JSON, it is applied with no fitting library. Here it is the
#    map such a fit returns for this voice.
fitted = PiecewiseLinear.from_dict({"xs": [0.0, 0.68, 1.0], "ys": [0.0, 0.38, 0.38], "n": 2000})
calibrated = score(happened, fitted.predict(stated))
assert calibrated.brier_score is not None and raw.brier_score is not None
assert calibrated.brier_score < raw.brier_score
assert abs(fitted(said.probability or 0.0) - 0.38) < 1e-9
print(f"calibrated: states {fitted(0.68):.2f}, brier {calibrated.brier_score:.3f}")

# 4. Verbal confidence. "likely" from author a has meant ~0.75; from author b ~0.45.
#    HedgeMap is a scikit-learn estimator: X is (source, hedge) rows, y what happened.
words = [("a", "likely")] * 300 + [("b", "Likely")] * 300 + [("c", "likely")] * 2
outcomes = [rng.random() < 0.75 for _ in range(300)] + [rng.random() < 0.45 for _ in range(300)]
outcomes += [True, True]  # two lucky calls from c
hedges = HedgeMap().fit(words, outcomes)
pa, pb, pc = (hedges.probability(src, "likely") for src in ("a", "b", "c"))
assert pa is not None and pb is not None and pc is not None
assert pa > pb and pc < 0.9  # c is shrunk toward the pool, not trusted on two calls
print(f"'likely': a={pa:.2f} b={pb:.2f} c={pc:.2f} (pooled {hedges.pooled_['likely']:.2f})")

# 5. Pool two voices in log-odds; the equal-weight pool is the baseline a learned combiner
#    (adapters.river) has to beat. Combiners speak River's protocol: x is voice -> probability.
pool = LogitPool()
both = pool.predict_proba_one({"desk": fitted(0.68), "author-a": pa}).get(True)
assert both is not None and min(fitted(0.68), pa) < both < max(fitted(0.68), pa)
print(f"pooled: {both:.2f}; ece of the calibrated voice {ece(reliability(happened, fitted.predict(stated))):.3f}")
assert said.resolve_time - said.forecast_time < timedelta(days=1)
