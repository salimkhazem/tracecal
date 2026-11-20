"""SECONDARY (not pre-registered): mixed acquisition streams.

Each test index i is observed under ONE condition c_i drawn uniformly from {clean} + 95 corruption conditions (seeded
by the re-split). Transductive QTC recalibrates once on the whole mixed stream (as a deployed system would on a
batch); RU-CP and split CP act per image. We report coverage per condition subgroup within the stream (MCS over
subgroups) as a function of mean set size, sweeping alpha'. Methods: split CP (base), RU-CP, QTC (base sets),
QTC + RU (union sets with the stream-level QTC threshold).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import view_names
from ..conformal.core import conformal_quantile
from ..paths import get_paths
from .bundles import load_bundle
from .complementary import qtc_beta
from .frontier import ALPHAS, RESTORERS, _softmax3, cov_size
from .run import PoolData, test_conditions

METHODS = ["base", "ru", "qtc", "qtc_ru"]


def run(tag: str, dataset: str, resplit: int, out_root: Path | None = None) -> Path:
    out = (out_root or get_paths().raw_results_root / "mixed_stream") / tag / f"r{resplit}"
    if (out / ".ok").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    vn = ["base"] + view_names(fam)
    ru_idx = [0] + [vn.index(v) for v in RESTORERS]
    pd = PoolData(tag, dataset, resplit, with_views=False)
    cal = pd.versions[0].take(pd.pos["cal"])
    t_base = 1.0 - cal.probs[np.arange(len(cal)), cal.labels]
    src_conf = cal.probs.max(1)
    conds = test_conditions(dataset)
    first = load_bundle(tag, conds[0][1], 0, cache=False)
    n = len(first)
    rng = np.random.default_rng(5000 + resplit)
    assign = rng.integers(0, len(conds), n)
    P = None
    y = np.empty(n, np.int64)
    for ci, (cn, src) in enumerate(conds):
        rows = np.flatnonzero(assign == ci)
        if not len(rows):
            continue
        b = load_bundle(tag, src, 0, cache=False).take(rows)
        Pc = _softmax3(b.all_view_logits()[:, ru_idx])
        if P is None:
            P = np.empty((n,) + Pc.shape[1:], np.float32)
        P[rows] = Pc
        y[rows] = b.labels
    assert P is not None
    base_scores = 1.0 - P[:, 0]
    ru_scores = (1.0 - P).min(1)
    tgt_conf = P[:, 0].max(1)
    q_split = np.array([conformal_quantile(t_base, a) for a in ALPHAS])
    q_qtc = np.array([conformal_quantile(t_base, be) if (be := qtc_beta(src_conf, tgt_conf, a)) > 0 else np.inf
                      for a in ALPHAS])
    res = {}
    for m, (U, q) in {"base": (base_scores, q_split), "ru": (ru_scores, q_split), "qtc": (base_scores, q_qtc),
                      "qtc_ru": (ru_scores, q_qtc)}.items():
        cov, size = cov_size(U, y, q)
        sub_cov = np.stack([cov[assign == ci].mean(0) for ci in range(len(conds))], 1)   # (A, K)
        res[f"{m}_subcov"] = sub_cov.astype(np.float32)
        res[f"{m}_size"] = size.mean(0).astype(np.float32)
        res[f"{m}_cov"] = cov.mean(0).astype(np.float32)
    np.savez_compressed(out / "curves.npz", **res, alphas=ALPHAS, assign=assign)
    (out / "meta.json").write_text(json.dumps({"tag": tag, "resplit": resplit, "conditions": [c for c, _ in conds]}))
    (out / ".ok").write_text("ok")
    return out


def summarize(tag: str, resplit: int) -> dict:
    from .gates_v2_eval import auc

    d = np.load(get_paths().raw_results_root / "mixed_stream" / tag / f"r{resplit}" / "curves.npz")
    curves = {m: (np.maximum(0, 0.9 - d[f"{m}_subcov"]).mean(1), d[f"{m}_size"]) for m in METHODS}
    la = np.log(ALPHAS)
    w = (float(np.interp(np.log(0.10), la, curves["base"][1])), float(np.interp(np.log(0.01), la, curves["base"][1])))
    out = {}
    for b in ("base", "qtc", "qtc_ru"):
        mr, sr = curves["ru"]
        mb, sb = curves[b]
        lo, hi = max(w[0], sr.min(), sb.min()), min(w[1], sr.max(), sb.max())
        out[f"ru_vs_{b}"] = float("nan") if hi - lo < 0.1 * (w[1] - w[0]) else auc(sr, mr, lo, hi) - auc(sb, mb, lo, hi)
    i10 = int(np.argmin(np.abs(la - np.log(0.1))))
    for m in METHODS:
        out[f"{m}_subgroup_mcs_a01"] = float(curves[m][0][i10])
        out[f"{m}_size_a01"] = float(curves[m][1][i10])
        out[f"{m}_overall_cov_a01"] = float(d[f"{m}_cov"][i10])
    return out
