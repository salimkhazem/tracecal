#!/usr/bin/env python
"""E2 (internal review 2026-10-23): sensitivity of RBE-CP to the restorer magnitudes. The boost term uses the six
restorers of the default bank, of the 'weak' bank (every magnitude halved) or of the 'strong' bank (the pre-defined
M = 2 magnitudes) (configs/core/banks_secondary.yaml); the anchor is always the 13-view default average. lambda is
selected with the v3 rule for each bank; calibration on clean data. Fresh CIFAR models (seeds 5-8).
Delta = area(bank variant) - area(default RBE-CP), the v3 pair rule, mean over re-splits.
  uv run python scripts/run_magnitude_ablation.py c10_resnet18_s5      (on ws; results/tables/magnitude_<tag>.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ablations import LAMS, area, pair, quantiles  # noqa: E402

from tracecal.analysis.bundles import load_bundle, softmax  # noqa: E402
from tracecal.analysis.dense_check import DENSE, curves_boot, window  # noqa: E402
from tracecal.analysis.frontier import RESTORERS, TARGET, _softmax3  # noqa: E402
from tracecal.analysis.run import test_conditions  # noqa: E402
from tracecal.config import families, view_names  # noqa: E402
from tracecal.data.splits import pool_resplit  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

BANKS = ("default", "weak", "strong")


def scores(z_def: np.ndarray, z_bank: np.ndarray, r_idx: list[int], lam: float) -> np.ndarray:
    a = softmax(z_def.mean(1))
    if lam == 0:
        return 1.0 - a
    b = _softmax3(z_bank[:, r_idx]).max(1)
    return 1.0 - (a + lam * np.maximum(b - a, 0.0))


def main() -> None:
    tag = sys.argv[1]
    ds = "cifar10" if tag.startswith("c10_") else "cifar100"
    vn = ["base"] + view_names("cifar")
    r_def = [vn.index(r) for r in RESTORERS]
    r_bank = list(range(1, 7))   # secondary banks: base + their 6 restorers (+ hflip)
    K = len(families()) + 2
    pools = {b: [load_bundle(tag, f"{ds}:pool", v, bank=b, cache=False) for v in range(K)] for b in BANKS}
    conds = test_conditions(ds)
    res = {b: [] for b in BANKS}
    lam_star = {b: [] for b in BANKS}
    tests = [(load_bundle(tag, src, 0, cache=False), {b: load_bundle(tag, src, 0, bank=b, cache=False)
                                                      for b in BANKS if b != "default"}) for _, src in conds]
    for r in range(5):
        sp = pool_resplit(len(pools["default"][0].ids), "cifar", r)
        st = {}
        for b in BANKS:
            idx = r_def if b == "default" else r_bank

            def S(vdef, vb, lam):
                return scores(vdef.all_view_logits(), (vdef if b == "default" else vb).all_view_logits(), idx, lam)

            cal_d, cal_b = pools["default"][0].take(sp["cal"]), pools[b][0].take(sp["cal"])
            yc = cal_d.labels
            q = {l: quantiles(S(cal_d, cal_b, l)[np.arange(len(yc)), yc]) for l in LAMS}
            mcs = {l: np.zeros(len(DENSE)) for l in LAMS}
            size = {l: np.zeros(len(DENSE)) for l in LAMS}
            base_size = np.zeros(len(DENSE))
            qb = quantiles(1.0 - cal_d.probs[np.arange(len(yc)), yc])
            for v in range(K):
                td, tb = pools["default"][v].take(sp["tune"]), pools[b][v].take(sp["tune"])
                for l in LAMS:
                    cp, sz, _, _ = curves_boot(S(td, tb, l), td.labels, q[l])
                    mcs[l] += np.maximum(0.0, TARGET - cp) / K
                    size[l] += sz / K
                _, sz0, _, _ = curves_boot(1.0 - td.probs, td.labels, qb)
                base_size += sz0 / K
            w = window(base_size)
            a = {l: area(size[l], mcs[l], w) for l in LAMS}
            lam = min(LAMS, key=lambda l: a[l] if np.isfinite(a[l]) else np.inf)
            lam_star[b].append(lam)
            st[b] = (lam, q[lam], idx)
        pm = {b: np.zeros(len(DENSE)) for b in BANKS}
        ps = {b: np.zeros(len(DENSE)) for b in BANKS}
        psb = np.zeros(len(DENSE))
        qb_cal = quantiles(1.0 - pools["default"][0].take(sp["cal"]).probs[
            np.arange(len(sp["cal"])), pools["default"][0].take(sp["cal"]).labels])
        for bd, extra in tests:
            for b in BANKS:
                lam, qq, idx = st[b]
                z_b = bd.all_view_logits() if b == "default" else extra[b].all_view_logits()
                cp, sz, _, _ = curves_boot(scores(bd.all_view_logits(), z_b, idx, lam), bd.labels, qq)
                pm[b] += np.maximum(0.0, TARGET - cp) / len(conds)
                ps[b] += sz / len(conds)
            _, sz0, _, _ = curves_boot(1.0 - bd.probs, bd.labels, qb_cal)
            psb += sz0 / len(conds)
        w = window(psb)
        for b in BANKS:
            res[b].append(pair(ps[b], pm[b], ps["default"], pm["default"], w))
    out = {"tag": tag, "lam_star": lam_star,
           "delta_vs_default": {b: {"mean": float(np.nanmean(v)), "per_resplit": v} for b, v in res.items()}}
    p = get_paths().results_root / "tables" / f"magnitude_{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(tag, {b: round(d["mean"], 3) for b, d in out["delta_vs_default"].items()}, lam_star, flush=True)


if __name__ == "__main__":
    main()
