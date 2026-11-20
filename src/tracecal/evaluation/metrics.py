"""Coverage / efficiency / calibration metrics."""

from __future__ import annotations

import numpy as np


def mcs(coverages: np.ndarray, alpha: float) -> float:
    """Mean coverage shortfall over conditions: mean_c max(0, 1 - alpha - cov_c)."""
    c = np.asarray(coverages, dtype=np.float64)
    return float(np.maximum(0.0, (1.0 - alpha) - c).mean())


def sscv(covered: np.ndarray, size: np.ndarray, alpha: float, bins, min_count: int = 20) -> float:
    """Size-stratified coverage violation (Angelopoulos et al. 2021): max over size bins of |cov - (1-alpha)|."""
    worst = 0.0
    for lo, hi in bins:
        m = (size >= lo) & (size <= hi)
        if m.sum() >= min_count:
            worst = max(worst, abs(float(covered[m].mean()) - (1.0 - alpha)))
    return worst


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def adaptive_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-mass bins."""
    order = np.argsort(conf, kind="stable")
    out = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            out += len(chunk) / len(conf) * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(out)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    oh = np.zeros_like(probs)
    oh[np.arange(len(labels)), labels] = 1.0
    return float(((probs - oh) ** 2).sum(1).mean())
