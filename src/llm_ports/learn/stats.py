"""The statistics a gate reads: agreement between two labellers, and a multiple-testing mask.

- `cohen_kappa` / `gwet_ac1` / `kappa_ci`: chance-corrected agreement of two labellers on the same binary
  items. Read kappa; read AC1 beside it where one answer dominates (kappa's chance correction
  degenerates there).
- `bh_fdr`: which of a family of p-values survive false-discovery-rate control, Benjamini-Yekutieli
  by default (valid under arbitrary dependence), plain Benjamini-Hochberg on request.

Invariants:
    - Pure, standard library only.
    - A statistic the data cannot define is None, never a number.

Rationale: ADR-0010.
"""

from __future__ import annotations

from collections.abc import Sequence

#: The positive label of a binary item.
YES = "yes"
#: The two-sided 95% normal quantile, `NormalDist().inv_cdf(0.975)`.
Z95 = 1.959963984540054
#: `(labeller a, labeller b)` labels of the same items.
Pairs = Sequence[tuple[str, str]]


def _decided(pairs: Pairs) -> tuple[int, float] | None:
    """Return `(n, observed agreement)`, or None below two pairs."""
    n = len(pairs)
    if n < 2:
        return None
    return n, sum(1 for a, b in pairs if a == b) / n


def cohen_kappa(pairs: Pairs) -> float | None:
    """Return Cohen's kappa over binary `(a, b)` label pairs.

    Both labels are `"yes"` or anything else (read as no); abstentions belong outside `pairs`.

    Returns:
        Kappa, or None when it is undefined: under two pairs, or both labellers one-sided on the
        same answer (chance agreement 1.0).
    """
    got = _decided(pairs)
    if got is None:
        return None
    n, po = got
    pa = sum(1 for a, _ in pairs if a == YES) / n
    pb = sum(1 for _, b in pairs if b == YES) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def kappa_ci(pairs: Pairs, z: float = Z95) -> tuple[float, float] | None:
    """Return the confidence interval of Cohen's kappa over binary `(a, b)` label pairs.

    The interval is `kappa +- z * SE`, with SE the large-sample standard error of Fleiss, Cohen and
    Everitt (1969), the one statsmodels' `cohens_kappa` reports; each end is clipped to [-1, 1].

    Args:
        pairs: As `cohen_kappa`.
        z: The normal quantile; `Z95` (default) for a 95% interval.

    Returns:
        `(low, high)`, or None wherever `cohen_kappa` is None.
    """
    k = cohen_kappa(pairs)
    if k is None:
        return None
    n = len(pairs)
    cells = {
        (x, y): sum(1 for a, b in pairs if (a == YES) == x and (b == YES) == y) / n
        for x in (True, False)
        for y in (True, False)
    }
    row = {x: cells[(x, True)] + cells[(x, False)] for x in (True, False)}
    col = {y: cells[(True, y)] + cells[(False, y)] for y in (True, False)}
    pe = sum(row[c] * col[c] for c in (True, False))
    diag = sum(cells[(c, c)] * (1 - (row[c] + col[c]) * (1 - k)) ** 2 for c in (True, False))
    off = (1 - k) ** 2 * sum(
        cells[(x, y)] * (col[x] + row[y]) ** 2 for x in (True, False) for y in (True, False) if x != y
    )
    var = max(diag + off - (k - pe * (1 - k)) ** 2, 0.0) / (n * (1 - pe) ** 2)
    se = var**0.5
    return max(k - z * se, -1.0), min(k + z * se, 1.0)


def gwet_ac1(pairs: Pairs) -> float | None:
    """Return Gwet's AC1 over binary `(a, b)` label pairs.

    Chance agreement is `2 * pi * (1 - pi)`, with `pi` the pooled share of `"yes"`, so the statistic
    stays defined and bounded where one answer dominates.

    Returns:
        AC1, or None under two pairs.
    """
    got = _decided(pairs)
    if got is None:
        return None
    n, po = got
    pi = sum((a == YES) + (b == YES) for a, b in pairs) / (2 * n)
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe)


def bh_fdr(pvals: Sequence[float | None], alpha: float = 0.10, *, dependent: bool = True) -> list[bool]:
    """Return which p-values survive false-discovery-rate control at `alpha`, in input order.

    The step-up compares each p to `rank / (m * c) * alpha` (never `p * m * c / rank` to `alpha`:
    the two float orders disagree at the threshold on grid p-values such as `k / (B + 1)`).

    Args:
        pvals: The family's p-values; a None is an untested hypothesis and never passes.
        alpha: The FDR level.
        dependent: True (default) for Benjamini-Yekutieli, which multiplies the threshold's
            denominator by `c(m) = sum(1/i)` and controls FDR under any dependence between the
            tests; False for plain Benjamini-Hochberg (valid under independence or PRDS).

    Returns:
        One bool per p-value. Never raises.

    Invariants:
        - The family size `m` counts tested hypotheses only: a None neither passes nor loosens
          the bar for the others.
    """
    m = len(pvals)
    if m == 0:
        return []
    n_real = sum(1 for p in pvals if p is not None)
    if n_real == 0:
        return [False] * m
    c_m = sum(1.0 / i for i in range(1, n_real + 1)) if dependent else 1.0
    order = sorted(range(m), key=lambda i: (pvals[i] is None, pvals[i] if pvals[i] is not None else 1.0))
    kmax = 0
    for rank, i in enumerate(order, start=1):
        p = pvals[i]
        if p is not None and p <= rank / (n_real * c_m) * alpha:
            kmax = rank
    passed = [False] * m
    for i in order[:kmax]:
        passed[i] = True
    return passed


__all__ = ["YES", "Z95", "Pairs", "bh_fdr", "cohen_kappa", "gwet_ac1", "kappa_ci"]
