"""Statistics: hierarchical bootstrap of Delta MCS, Holm, Wilcoxon, Spearman."""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats


def shortfall(cov: np.ndarray, alpha: float) -> np.ndarray:
    return np.maximum(0.0, (1.0 - alpha) - cov)


def multinomial_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n, n_boot) bootstrap counts (each column sums to n)."""
    rng = np.random.default_rng(seed)
    w = np.empty((n, n_boot), dtype=np.float32)
    for b in range(n_boot):
        w[:, b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return w


def hierarchical_delta_mcs(cov_a: dict, cov_b: dict, alpha: float, n_boot: int = 1000, seed: int = 0,
                           groups: dict | None = None) -> dict[str, float]:
    """Delta MCS = MCS(a) - MCS(b), averaged over replicates.

    cov_a / cov_b: {replicate_key: bool array (n_conditions, n_images)}; all replicates share the image axis.
    groups: {replicate_key: group_key} (e.g. training seed) for two-level resampling (groups, then replicates
    within group); None -> one level. Images are resampled jointly for all replicates and both methods.
    """
    keys = sorted(cov_a)
    assert sorted(cov_b) == keys
    n_img = next(iter(cov_a.values())).shape[1]
    w = multinomial_weights(n_img, n_boot, seed)
    point, boot = {}, {}
    for k in keys:
        a = cov_a[k].astype(np.float32)
        b = cov_b[k].astype(np.float32)
        point[k] = float(shortfall(a.mean(1), alpha).mean() - shortfall(b.mean(1), alpha).mean())
        ca = (a @ w) / n_img
        cb = (b @ w) / n_img
        boot[k] = shortfall(ca, alpha).mean(0) - shortfall(cb, alpha).mean(0)  # (n_boot,)
    rng = np.random.default_rng(seed + 1)
    if groups is None:
        groups = {k: k for k in keys}
    gkeys = sorted(set(groups.values()))
    members = {g: [k for k in keys if groups[k] == g] for g in gkeys}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        gs = rng.choice(len(gkeys), size=len(gkeys), replace=True)
        vals = []
        for gi in gs:
            mem = members[gkeys[gi]]
            pick = rng.choice(len(mem), size=len(mem), replace=True)
            vals.extend(boot[mem[j]][b] for j in pick)
        draws[b] = float(np.mean(vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p_two = float(min(1.0, 2.0 * min((draws >= 0).mean(), (draws <= 0).mean())))
    return {"delta": float(np.mean(list(point.values()))), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": p_two, "n_replicates": len(keys), "n_boot": n_boot}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    prev_reject = True
    for i, (key, p) in enumerate(items):
        thr = alpha / (m - i)
        reject = prev_reject and p <= thr
        out[key] = {"p": p, "threshold": thr, "reject": reject}
        prev_reject = reject
    return out


def wilcoxon_less(deltas: list[float]) -> dict[str, float]:
    if len(deltas) < 2 or np.allclose(deltas, 0):
        return {"stat": float("nan"), "p": float("nan"), "n": len(deltas)}
    r = sstats.wilcoxon(deltas, alternative="less")
    return {"stat": float(r.statistic), "p": float(r.pvalue), "n": len(deltas)}


def spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    r = sstats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue)}


def spearman_boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        r = sstats.spearmanr(x[i], y[i]).statistic
        if np.isfinite(r):
            vals.append(r)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"rho": float(sstats.spearmanr(x, y).statistic), "ci_lo": float(lo), "ci_hi": float(hi)}
