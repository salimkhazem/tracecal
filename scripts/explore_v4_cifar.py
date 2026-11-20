#!/usr/bin/env python
"""EXPLORATION ONLY (CIFAR development models, seeds 1-4 of v2): the v3 procedure (lambda selected on the tune split's
synthetic versions) with the ensemble anchor over all 13 views (v3) or over identity + flip only (candidate v4).
Re-split 0, point estimates, dense alpha' grid; efficiency-matched areas over the 96 CIFAR-C test conditions.

  uv run python scripts/explore_v4_cifar.py c10_resnet18_s1      (on ws; writes results/explore/v4_cifar_<tag>.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracecal.analysis.bundles import load_bundle, softmax  # noqa: E402
from tracecal.analysis.dense_check import DENSE, window  # noqa: E402
from tracecal.analysis.frontier import RESTORERS, TARGET, _softmax3  # noqa: E402
from tracecal.analysis.gates_v2_eval import auc  # noqa: E402
from tracecal.analysis.run import PoolData, test_conditions  # noqa: E402
from tracecal.config import view_names  # noqa: E402
from tracecal.conformal.core import conformal_quantile  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
VN = ["base"] + view_names("cifar")
R_IDX = [VN.index(v) for v in RESTORERS]
ANCHORS = {"avg13": list(range(len(VN))), "avg2": [0, VN.index("hflip")]}


def scores(vw: np.ndarray) -> dict[str, np.ndarray]:
    P = _softmax3(vw)
    rmax = P[:, R_IDX].max(1)
    out = {"base": 1.0 - P[:, 0], "ru": 1.0 - P[:, [0] + R_IDX].max(1)}
    for a, idx in ANCHORS.items():
        anc = softmax(vw[:, idx].mean(1))
        ex = np.maximum(rmax - anc, 0.0)
        for lam in LAMBDAS:
            out[f"{a}_{lam:g}"] = 1.0 - (anc + lam * ex)
    return out


def curve(U, y, q):
    n, A = len(y), len(q)
    order = np.argsort(q, kind="stable")
    qs = q[order]
    size = np.cumsum(np.bincount(np.searchsorted(qs, U.ravel(), side="left"), minlength=A + 1)[:A]) / n
    cov = np.cumsum(np.bincount(np.searchsorted(qs, U[np.arange(n), y], side="left"), minlength=A + 1)[:A]) / n
    o = np.empty_like(order)
    o[order] = np.arange(A)
    return cov[o], size[o]


def area(size, mcs, w):
    lo, hi = max(w[0], size.min()), min(w[1], size.max())
    return float("nan") if hi - lo < 0.1 * (w[1] - w[0]) else 100 * auc(size, mcs, lo, hi)


def pair(size, mcs, a, b, w):
    lo = max(w[0], size[a].min(), size[b].min())
    hi = min(w[1], size[a].max(), size[b].max())
    if hi - lo < 0.1 * (w[1] - w[0]):
        return float("nan")
    return 100 * (auc(size[a], mcs[a], lo, hi) - auc(size[b], mcs[b], lo, hi))


def main() -> None:
    tag = sys.argv[1]
    dataset = {"c10": "cifar10", "c100": "cifar100"}[tag.split("_")[0]]
    pd = PoolData(tag, dataset, 0, with_views=True)
    cal = pd.versions[0].take(pd.pos["cal"])
    Sc = scores(cal.all_view_logits())
    yc = cal.labels
    q = {m: np.array([conformal_quantile(s[np.arange(len(yc)), yc], a) for a in DENSE]) for m, s in Sc.items()}
    # selection on the tune split's synthetic versions (as in v3)
    K = len(pd.versions)
    mcs = {m: np.zeros(len(DENSE)) for m in q}
    size = {m: np.zeros(len(DENSE)) for m in q}
    for v in range(K):
        b = pd.versions[v].take(pd.pos["tune"])
        S = scores(b.all_view_logits())
        for m in q:
            cv, sz = curve(S[m], b.labels, q[m])
            mcs[m] += np.maximum(0.0, TARGET - cv) / K
            size[m] += sz / K
    w = window(size["base"])
    lam_star = {}
    for a in ANCHORS:
        ar = {lam: area(size[f"{a}_{lam:g}"], mcs[f"{a}_{lam:g}"], w) for lam in LAMBDAS}
        fin = {k: v for k, v in ar.items() if np.isfinite(v)}
        lam_star[a] = min(LAMBDAS, key=lambda l: fin.get(l, np.inf))
    # test: all 96 CIFAR-C conditions
    conds = test_conditions(dataset)
    tm = {m: np.zeros(len(DENSE)) for m in q}
    ts = {m: np.zeros(len(DENSE)) for m in q}
    for cn, src in conds:
        b = load_bundle(tag, src, 0, cache=False)
        S = scores(b.all_view_logits())
        for m in q:
            cv, sz = curve(S[m], b.labels, q[m])
            tm[m] += np.maximum(0.0, TARGET - cv) / len(conds)
            ts[m] += sz / len(conds)
    wt = window(ts["base"])
    sel = {a: f"{a}_{lam_star[a]:g}" for a in ANCHORS}
    names = ["base", "avg13", "ru"]
    out = {"tag": tag, "lam_star": lam_star, "delta": {}}
    for a, m in sel.items():
        out["delta"][f"rbe_{a}_sel"] = {b: pair(ts, tm, m, b if b != "avg13" else "avg13_0", wt) for b in names}
    out["delta"]["v4_minus_v3"] = pair(ts, tm, sel["avg2"], sel["avg13"], wt)
    p = get_paths().results_root / "explore" / f"v4_cifar_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
