"""Split / Mondrian / weighted conformal prediction for classification (numpy, CPU).

Scores follow the convention "larger = less conforming"; a prediction set is {y : score(x, y) <= q}.
LAC (Sadinle et al.): score = 1 - p_y. APS (Romano et al., randomised): sum of probabilities strictly larger than
p_y plus u * p_y. RAPS (Angelopoulos et al.): APS + lam * max(0, rank(y) - k_reg).
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest calibration score; +inf when that index exceeds n."""
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    if k < 1:
        return -math.inf
    return float(np.partition(s, k - 1)[k - 1])


# ----------------------------------------------------------------------------------------------------------------
# scores
# ----------------------------------------------------------------------------------------------------------------
def lac_all(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs


def lac_true(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(labels)), labels]


def _aps_parts(probs: np.ndarray):
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum_before = np.cumsum(sorted_p, axis=1) - sorted_p
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(probs.shape[1])[None, :].repeat(len(probs), 0), axis=1)
    before = np.take_along_axis(cum_before, rank, axis=1)  # sum of probs ranked strictly before each label
    return before, rank


def aps_all(probs: np.ndarray, u: np.ndarray, lam: float = 0.0, k_reg: int = 0) -> np.ndarray:
    before, rank = _aps_parts(probs)
    s = before + u[:, None] * probs
    if lam > 0:
        s = s + lam * np.maximum(0, rank + 1 - k_reg)
    return s


def aps_true(probs: np.ndarray, labels: np.ndarray, u: np.ndarray, lam: float = 0.0, k_reg: int = 0) -> np.ndarray:
    return aps_all(probs, u, lam, k_reg)[np.arange(len(labels)), labels]


def score_fns(name: str, lam: float = 0.0, k_reg: int = 0):
    """Return (all_scores(probs, u), true_scores(probs, labels, u))."""
    if name == "lac":
        return (lambda p, u: lac_all(p)), (lambda p, y, u: lac_true(p, y))
    if name == "aps":
        return (lambda p, u: aps_all(p, u)), (lambda p, y, u: aps_true(p, y, u))
    if name == "raps":
        return (lambda p, u: aps_all(p, u, lam, k_reg)), (lambda p, y, u: aps_true(p, y, u, lam, k_reg))
    raise ValueError(name)


# ----------------------------------------------------------------------------------------------------------------
# set evaluation
# ----------------------------------------------------------------------------------------------------------------
def evaluate_sets(all_scores: np.ndarray, labels: np.ndarray, q: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Per-example (covered, size) for sets {y : score <= q_i}; q may be a scalar or per-example vector."""
    qv = np.broadcast_to(np.asarray(q, dtype=np.float64), (all_scores.shape[0],))
    inset = all_scores <= qv[:, None]
    covered = inset[np.arange(len(labels)), labels]
    return covered, inset.sum(1).astype(np.int32)


# ----------------------------------------------------------------------------------------------------------------
# Mondrian (cell-conditional) thresholds
# ----------------------------------------------------------------------------------------------------------------
def mondrian_thresholds(cells: np.ndarray, true_scores: np.ndarray, n_cells: int, alpha: float) -> np.ndarray:
    q = np.full(n_cells, math.inf)
    for b in range(n_cells):
        m = cells == b
        if m.any():
            q[b] = conformal_quantile(true_scores[m], alpha)
    return q


# ----------------------------------------------------------------------------------------------------------------
# weighted split conformal (Tibshirani et al. 2019): per-test-point threshold
# ----------------------------------------------------------------------------------------------------------------
def weighted_thresholds(cal_scores: np.ndarray, cal_w: np.ndarray, test_w: np.ndarray, alpha: float) -> np.ndarray:
    """q(x) = Quantile_{1-alpha}( sum_i w_i delta_{s_i} + w(x) delta_{+inf} ) after normalisation."""
    order = np.argsort(cal_scores, kind="stable")
    s_sorted = np.asarray(cal_scores, dtype=np.float64)[order]
    w_sorted = np.asarray(cal_w, dtype=np.float64)[order]
    cw = np.cumsum(w_sorted)
    total = cw[-1] + np.asarray(test_w, dtype=np.float64)
    target = (1.0 - alpha) * total
    idx = np.searchsorted(cw, target, side="left")
    q = np.full(len(test_w), math.inf)
    ok = idx < len(s_sorted)
    q[ok] = s_sorted[idx[ok]]
    return q
