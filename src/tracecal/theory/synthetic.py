"""Toy constructions that check the stated results numerically (used by tests and a supplementary figure).

P1  Mondrian validity under cell-reweighting shift: per-cell thresholds keep coverage when the target reweights
    cells arbitrarily (conditional law within each cell unchanged); one marginal threshold does not.
L4  Non-identifiability: for a non-injective label-free statistic g, a covariate shift can keep the law of g(X)
    fixed while concentrating mass on inputs where the classifier errs, so coverage drops undetectably.
P3  Conformal outlier p-values are super-uniform under exchangeability.
"""

from __future__ import annotations

import numpy as np

from ..conformal.core import conformal_quantile, mondrian_thresholds


def cell_mixture(n: int, weights: np.ndarray, rng: np.random.Generator, n_classes: int = 10,
                 noise_by_cell: np.ndarray | None = None):
    """Cells b with different difficulty: within cell b the classifier's probability of the true label is
    Beta-distributed with a cell-specific mean. Returns (cells, true-label scores)."""
    k = len(weights)
    noise_by_cell = np.linspace(0.05, 0.8, k) if noise_by_cell is None else noise_by_cell
    cells = rng.choice(k, size=n, p=weights / weights.sum())
    mean = 1.0 - noise_by_cell[cells]
    a = 5.0 * mean + 1e-3
    b = 5.0 * (1.0 - mean) + 1e-3
    p_true = rng.beta(a, b)
    return cells, 1.0 - p_true  # LAC score of the true label


def p1_check(alpha: float = 0.1, n_cal: int = 5000, n_test: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    k = 4
    w_src = np.full(k, 1.0 / k)
    w_tgt = np.array([0.02, 0.08, 0.2, 0.7])  # shift towards the hard cells
    c_cal, s_cal = cell_mixture(n_cal, w_src, rng)
    c_tst, s_tst = cell_mixture(n_test, w_tgt, rng)
    q_marg = conformal_quantile(s_cal, alpha)
    q_cell = mondrian_thresholds(c_cal, s_cal, k, alpha)
    return {"marginal_cov": float((s_tst <= q_marg).mean()), "mondrian_cov": float((s_tst <= q_cell[c_tst]).mean())}


def l4_check(alpha: float = 0.1, n: int = 40000, seed: int = 0) -> dict:
    """g(x) = cell only; within each cell 20% of the inputs are 'hard'. The target keeps every cell's mass (law of
    g unchanged) but samples only hard inputs -> coverage collapses (~0.5 at alpha = 0.1), g cannot tell."""
    rng = np.random.default_rng(seed)
    k = 4
    cells = rng.integers(0, k, n)
    hard = rng.random(n) < 0.2
    p_true = np.where(hard, rng.beta(1.0, 4.0, n), rng.beta(8.0, 1.0, n))
    score = 1.0 - p_true
    cal = rng.random(n) < 0.5
    q_cell = mondrian_thresholds(cells[cal], score[cal], k, alpha)
    test_src = ~cal
    test_tgt = ~cal & hard
    cov_src = float((score[test_src] <= q_cell[cells[test_src]]).mean())
    cov_tgt = float((score[test_tgt] <= q_cell[cells[test_tgt]]).mean())
    g_src = np.bincount(cells[test_src], minlength=k) / test_src.sum()
    g_tgt = np.bincount(cells[test_tgt], minlength=k) / test_tgt.sum()
    return {"cov_source": cov_src, "cov_target": cov_tgt, "tv_g": float(0.5 * np.abs(g_src - g_tgt).sum())}
