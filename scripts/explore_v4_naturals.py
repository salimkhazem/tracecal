#!/usr/bin/env python
"""EXPLORATION ONLY (development models of v2; never the fresh v3 models): why do test-time views lose on ImageNet-A,
and does a variant of the RBE score avoid it without losing on ImageNet-C?

Point estimates, re-split 0, dense alpha' grid, efficiency-matched area over the split-CP window of each benchmark
(ImageNet-C: MCS over the 96 conditions; natural sets: shortfall on the set). Scores:
  base, ru (identity + restorers union), avg13 (all 13 views), avg8 (identity, flip, 6 restorers), avg2 (identity,
  flip), and rbe{anchor}_{lam} = anchor + lam * (max restorer - anchor)_+ for anchors avg13 / avg8 / avg2 / base.

  uv run python scripts/explore_v4_naturals.py in_rn50     (on ws)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracecal.analysis.bundles import load_bundle, softmax  # noqa: E402
from tracecal.analysis.dense_check import DENSE, window  # noqa: E402
from tracecal.analysis.frontier import RESTORERS, TARGET, _softmax3  # noqa: E402
from tracecal.analysis.gates_v2_eval import auc  # noqa: E402
from tracecal.analysis.run import test_conditions  # noqa: E402
from tracecal.config import view_names  # noqa: E402
from tracecal.conformal.core import conformal_quantile  # noqa: E402
from tracecal.data.imagenet import natural_index  # noqa: E402
from tracecal.data.splits import pool_resplit  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

VN = ["base"] + view_names("imagenet")
R_IDX = [VN.index(v) for v in RESTORERS]
ANCHORS = {"avg13": list(range(len(VN))), "avg8": [0, VN.index("hflip")] + R_IDX, "avg2": [0, VN.index("hflip")],
           "base": [0]}
LAMS = (0.25, 0.5)
NATS = ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")


def scores(vw: np.ndarray) -> dict[str, np.ndarray]:
    P = _softmax3(vw)
    rmax = P[:, R_IDX].max(1)
    out = {"base": 1.0 - P[:, 0], "ru": 1.0 - P[:, [0] + R_IDX].max(1)}
    for a, idx in ANCHORS.items():
        anc = P[:, 0] if a == "base" else softmax(vw[:, idx].mean(1))
        if a != "base":
            out[a] = 1.0 - anc
        ex = np.maximum(rmax - anc, 0.0)
        for lam in LAMS:
            out[f"rbe_{a}_{lam:g}"] = 1.0 - (anc + lam * ex)
    return out


def curve(U: np.ndarray, y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    A = len(q)
    order = np.argsort(q, kind="stable")
    qs = q[order]
    k = np.searchsorted(qs, U.ravel(), side="left")
    size = np.cumsum(np.bincount(k, minlength=A + 1)[:A]) / n
    kc = np.searchsorted(qs, U[np.arange(n), y], side="left")
    cov = np.cumsum(np.bincount(kc, minlength=A + 1)[:A]) / n
    o = np.empty_like(order)
    o[order] = np.arange(A)
    return cov[o], size[o]


def thresholds(cal_vw: np.ndarray, yc: np.ndarray) -> dict[str, np.ndarray]:
    S = scores(cal_vw)
    return {m: np.array([conformal_quantile(s[np.arange(len(yc)), yc], a) for a in DENSE]) for m, s in S.items()}


def areas(mcs: dict, size: dict) -> dict[str, float]:
    w = window(size["base"])
    out = {}
    for m in mcs:
        lo = max(w[0], size[m].min(), size["base"].min())
        hi = min(w[1], size[m].max(), size["base"].max())
        out[m] = float("nan") if hi - lo < 0.1 * (w[1] - w[0]) else \
            100 * (auc(size[m], mcs[m], lo, hi) - auc(size["base"], mcs["base"], lo, hi))
    return out


def main() -> None:
    tag = sys.argv[1]
    t0 = time.time()
    pool = load_bundle(tag, "imagenet:pool", 0, cache=False)
    pos = pool_resplit(len(pool.ids), "imagenet", 0)["cal"]
    cal = pool.take(pos)
    res: dict = {"model": tag, "resplit": 0, "delta_vs_base": {}}
    # ImageNet-C (+ clean), all 96 conditions
    q = thresholds(cal.all_view_logits(), cal.labels)
    conds = test_conditions("imagenet")
    mcs = {m: np.zeros(len(DENSE)) for m in q}
    size = {m: np.zeros(len(DENSE)) for m in q}
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        S = scores(b.all_view_logits())
        for m in q:
            cov, sz = curve(S[m], b.labels, q[m])
            mcs[m] += np.maximum(0.0, TARGET - cov) / len(conds)
            size[m] += sz / len(conds)
        if ci % 24 == 0:
            print(f"[{tag}] {ci + 1}/{len(conds)} ({time.time() - t0:.0f}s)", flush=True)
    res["delta_vs_base"]["imagenet_c"] = areas(mcs, size)
    # natural sets: thresholds on the clean cal images of the set's classes
    for nat in NATS:
        sub = natural_index(nat)["class_subset"]
        c2 = cal.restrict(sub)
        qn = thresholds(c2.all_view_logits(), c2.labels)
        b = load_bundle(tag, nat, 0, cache=False).restrict(sub)
        S = scores(b.all_view_logits())
        mc, sz = {}, {}
        for m in qn:
            cov, s_ = curve(S[m], b.labels, qn[m])
            mc[m], sz[m] = np.maximum(0.0, TARGET - cov), s_
        res["delta_vs_base"][nat] = areas(mc, sz)
        print(f"[{tag}] {nat} done ({time.time() - t0:.0f}s)", flush=True)
    out = get_paths().results_root / "explore" / f"v4_naturals_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    names = list(res["delta_vs_base"]["imagenet_c"])
    print(f"{'score':18s}" + "".join(f"{k[:12]:>13s}" for k in res["delta_vs_base"]))
    for m in names:
        print(f"{m:18s}" + "".join(f"{res['delta_vs_base'][k][m]:+13.2f}" for k in res["delta_vs_base"]))


if __name__ == "__main__":
    main()
