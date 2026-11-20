"""SECONDARY (not pre-registered): is RU-CP complementary to transductive recalibration?

QTC (Yilmaz & Heckel) estimates, per test condition, a recalibration level beta from the max-softmax of the clean
calibration set and of the unlabeled test condition, and recalibrates the base threshold at beta. We apply the SAME
per-condition threshold to (a) the base set (QTC) and (b) the restorative union (QTC + RU), sweep alpha', and compare
coverage-shortfall vs set-size curves (normalised area over the common window), per run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..conformal.core import conformal_quantile
from ..paths import get_paths
from .bundles import load_bundle
from .frontier import ALPHAS, RESTORERS, _softmax3, cov_size
from .run import PoolData, test_conditions
from ..config import view_names


def _q(s: np.ndarray, c: float) -> float:
    srt = np.sort(s)
    k = max(1, math.ceil(c * len(srt)))
    return float(srt[min(k, len(srt)) - 1])


def qtc_beta(src_conf: np.ndarray, tgt_conf: np.ndarray, alpha: float) -> float:
    beta_t = float(np.mean(src_conf < _q(tgt_conf, alpha)))
    beta_s = 1.0 - float(np.mean(tgt_conf < _q(src_conf, 1.0 - alpha)))
    return min(beta_t, beta_s)


def run(tag: str, dataset: str, resplit: int, out_root: Path | None = None) -> Path:
    out = (out_root or get_paths().raw_results_root / "complementary") / tag / f"r{resplit}"
    if (out / ".ok").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    vn = ["base"] + view_names(fam)
    ru_idx = [0] + [vn.index(v) for v in RESTORERS]
    pd = PoolData(tag, dataset, resplit, with_views=False)
    cal = pd.versions[0].take(pd.pos["cal"])
    p_cal = cal.probs
    t_base = 1.0 - p_cal[np.arange(len(cal)), cal.labels]
    src_conf = p_cal.max(1)
    conds = test_conditions(dataset)
    A, K = len(ALPHAS), len(conds)
    res = {k: np.zeros((A, K), np.float32) for k in ("qtc_cov", "qtc_size", "qtcru_cov", "qtcru_size")}
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        P = _softmax3(b.all_view_logits()[:, ru_idx])
        tgt_conf = P[:, 0].max(1)
        q = np.array([conformal_quantile(t_base, be) if (be := qtc_beta(src_conf, tgt_conf, a)) > 0 else np.inf
                      for a in ALPHAS])
        c1, s1 = cov_size(1.0 - P[:, 0], b.labels, q)
        c2, s2 = cov_size((1.0 - P).min(1), b.labels, q)
        res["qtc_cov"][:, ci], res["qtc_size"][:, ci] = c1.mean(0), s1.mean(0)
        res["qtcru_cov"][:, ci], res["qtcru_size"][:, ci] = c2.mean(0), s2.mean(0)
    np.savez_compressed(out / "curves.npz", **res, alphas=ALPHAS)
    (out / "meta.json").write_text(json.dumps({"tag": tag, "resplit": resplit, "conditions": [c for c, _ in conds]}))
    (out / ".ok").write_text("ok")
    return out


def delta(tag: str, resplit: int) -> dict[str, float]:
    from .gates_v2_eval import auc

    d = np.load(get_paths().raw_results_root / "complementary" / tag / f"r{resplit}" / "curves.npz")
    m1 = np.maximum(0, 0.9 - d["qtc_cov"]).mean(1)
    s1 = d["qtc_size"].mean(1)
    m2 = np.maximum(0, 0.9 - d["qtcru_cov"]).mean(1)
    s2 = d["qtcru_size"].mean(1)
    lo, hi = max(s1.min(), s2.min()), min(s1.max(), s2.max())
    if hi <= lo:
        return {"delta": float("nan")}
    i10 = int(np.argmin(np.abs(np.log(ALPHAS) - np.log(0.1))))
    return {"delta": auc(s2, m2, lo, hi) - auc(s1, m1, lo, hi), "window": [float(lo), float(hi)],
            "qtc_mcs_a01": float(m1[i10]), "qtc_size_a01": float(s1[i10]),
            "qtcru_mcs_a01": float(m2[i10]), "qtcru_size_a01": float(s2[i10])}
