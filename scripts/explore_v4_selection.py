#!/usr/bin/env python
"""EXPLORATION ONLY (v2 development models): which lambda does the v3 selection rule (tune split, synthetic program
versions v0..v8, efficiency-matched area, ties -> smallest lambda) pick when the ensemble anchor is restricted?

Anchors: avg13 (all 13 views, = v3), avg2 (identity + flip). Writes results/explore/v4_selection_<tag>.json.

  uv run python scripts/explore_v4_selection.py in_rn50      (on ws)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracecal.analysis.bundles import softmax  # noqa: E402
from tracecal.analysis.dense_check import DENSE, window  # noqa: E402
from tracecal.analysis.frontier import RESTORERS, TARGET, _softmax3  # noqa: E402
from tracecal.analysis.gates_v2_eval import auc  # noqa: E402
from tracecal.analysis.run import PoolData  # noqa: E402
from tracecal.config import view_names  # noqa: E402
from tracecal.conformal.core import conformal_quantile  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
VN = ["base"] + view_names("imagenet")
R_IDX = [VN.index(v) for v in RESTORERS]
ANCHORS = {"avg13": list(range(len(VN))), "avg2": [0, VN.index("hflip")]}


def scores(vw: np.ndarray) -> dict[str, np.ndarray]:
    P = _softmax3(vw)
    rmax = P[:, R_IDX].max(1)
    out = {"base": 1.0 - P[:, 0]}
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


def main() -> None:
    tag = sys.argv[1]
    resplits = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    out = {"model": tag, "per_resplit": {}}
    for r in resplits:
        pd = PoolData(tag, "imagenet", r, with_views=True)
        cal = pd.versions[0].take(pd.pos["cal"])
        Sc = scores(cal.all_view_logits())
        yc = cal.labels
        q = {m: np.array([conformal_quantile(s[np.arange(len(yc)), yc], a) for a in DENSE]) for m, s in Sc.items()}
        tune = pd.pos["tune"]
        K = len(pd.versions)
        mcs = {m: np.zeros(len(DENSE)) for m in q}
        size = {m: np.zeros(len(DENSE)) for m in q}
        for v in range(K):
            b = pd.versions[v].take(tune)
            S = scores(b.all_view_logits())
            for m in q:
                cv, sz = curve(S[m], b.labels, q[m])
                mcs[m] += np.maximum(0.0, TARGET - cv) / K
                size[m] += sz / K
        w = window(size["base"])
        res = {}
        for a in ANCHORS:
            areas = {}
            for lam in LAMBDAS:
                m = f"{a}_{lam:g}"
                lo, hi = max(w[0], size[m].min()), min(w[1], size[m].max())
                areas[lam] = float("nan") if hi - lo < 0.1 * (w[1] - w[0]) else 100 * auc(size[m], mcs[m], lo, hi)
            fin = {k: v for k, v in areas.items() if np.isfinite(v)}
            res[a] = {"areas": {f"{k:g}": v for k, v in areas.items()}, "lam_star": min(LAMBDAS, key=lambda l: fin.get(l, np.inf))}
        out["per_resplit"][r] = res
        print(tag, f"r{r}", {a: (v["lam_star"], {k: round(x, 3) for k, x in v["areas"].items()}) for a, v in res.items()},
              flush=True)
    p = get_paths().results_root / "explore" / f"v4_selection_{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print("written", p)


if __name__ == "__main__":
    main()
