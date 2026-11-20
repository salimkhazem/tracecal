#!/usr/bin/env python
"""Ablation addendum G_majority_vote (audits/ablations_v5c.yaml): merging the 13 per-view conformal sets by majority
vote (Gasparin & Ramdas, 2024). Each view T has its own clean split-CP thresholds q_T(alpha') on the dense grid; a label
is in the vote set at alpha' iff it is in >= 7 of the 13 per-view sets. Exact and cheap: per-view thresholds are
monotone in alpha', so the label enters view T's set from grid order k_T on, and the vote set from the 7th smallest k_T.
Compared with RBE-CP and split CP through the curves that run_ablations.py saved for the same re-splits.
  uv run python scripts/run_majority_vote.py c10_resnet18_s5      (on ws; results/tables/majority_<tag>.json)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ablations import pair, quantiles  # noqa: E402

from tracecal.analysis.bundles import load_bundle  # noqa: E402
from tracecal.analysis.dense_check import DENSE, curves_boot, window  # noqa: E402
from tracecal.analysis.frontier import P0, TARGET, _softmax3  # noqa: E402
from tracecal.analysis.run import PoolData, test_conditions  # noqa: E402
from tracecal.config import families  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

ORDER = np.argsort(-DENSE, kind="stable")          # alpha' descending = thresholds ascending, for every view
POS = np.empty_like(ORDER)
POS[ORDER] = np.arange(len(DENSE))                  # grid index -> order position
MAJ = 7                                             # > 13 / 2


def thresholds(views_cal: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """(V, A) per-view thresholds in ORDER (non-decreasing along the last axis)."""
    P = _softmax3(views_cal)
    return np.stack([quantiles(1.0 - P[np.arange(len(yc)), t, yc])[ORDER] for t in range(P.shape[1])])


def vote_scores(vw: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """(n, C) order position from which each label is in >= MAJ per-view sets (A = never)."""
    P = _softmax3(vw)
    k = np.empty((P.shape[1],) + P.shape[:1] + P.shape[2:], dtype=np.int16)
    for t in range(P.shape[1]):
        k[t] = np.searchsorted(Q[t], 1.0 - P[:, t], side="left")
    return np.partition(k, MAJ - 1, axis=0)[MAJ - 1].astype(np.float64)


def main() -> None:
    tag = sys.argv[1]
    dataset = {"c10": "cifar10", "c100": "cifar100"}.get(tag.split("_")[0], "imagenet")
    t0 = time.time()
    rs = [0, 1, 2, 3, 4]
    ref = dict(np.load(get_paths().results_root / "tables" / f"ablate_{tag}_curves.npz"))
    Q = {}
    for r in rs:
        pd = PoolData(tag, dataset, r, with_views=True)
        ctx = pd.ctx(P0, families(), 0.1, "lac", None, views=True)
        Q[r] = thresholds(ctx.clean_cal_views, ctx.clean_cal.labels)
    conds = test_conditions(dataset)
    A = len(DENSE)
    pm = {r: np.zeros(A) for r in rs}
    ps = {r: np.zeros(A) for r in rs}
    qpos = POS.astype(np.float64)                     # pseudo-threshold of grid index a = its order position
    for ci, (_, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        vw = b.all_view_logits()
        for r in rs:
            cp, sz, _, _ = curves_boot(vote_scores(vw, Q[r]), b.labels, qpos)
            pm[r] += np.maximum(0.0, TARGET - cp) / len(conds)
            ps[r] += sz / len(conds)
        if ci % 16 == 0:
            print(f"[{tag}] {ci + 1}/{len(conds)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"tag": tag}
    for refname, key in (("A3_rbe", "delta_vs_rbe"), ("A0_split", "delta_vs_split")):
        vals = []
        for r in rs:
            w = window(ref[f"ps__{r}__A0_split"])
            vals.append(pair(ps[r], pm[r], ref[f"ps__{r}__{refname}"], ref[f"pm__{r}__{refname}"], w))
        a = np.array(vals, dtype=float)
        out[key] = {"mean": float(np.nanmean(a)), "per_resplit": vals,
                    "se": float(np.nanstd(a, ddof=1) / np.sqrt(np.isfinite(a).sum())) if np.isfinite(a).sum() > 1 else None}
    i10 = int(np.argmin(np.abs(np.log(DENSE) - np.log(0.1))))
    out["mean_size_a01"] = float(np.mean([ps[r][i10] for r in rs]))
    p = get_paths().results_root / "tables" / f"majority_{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(tag, {k: round(v["mean"], 3) for k, v in out.items() if isinstance(v, dict)}, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
