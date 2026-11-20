#!/usr/bin/env python
"""v6 DEVELOPMENT (not a confirmatory analysis): camera-aware RBE-CP on the v2 development models only, evaluated on
ImageNet-ES es-val (139 conditions) and an ImageNet-C subset (clean + severities 3 and 5). es-test and the fresh models
are never read here.

Candidates = anchor x boost set, lambda selected by the v3 rule (argmin efficiency-matched area on the tune split's
synthetic pool versions, grid {0, .25, .5, .75, 1}, ties -> smallest), calibrated on clean data:
  anchors : id (identity view), sym (identity + flip), all13 (the 13 default views = RBE-CP's anchor)
  boosts  : R (the six default restorers), C (the five camera restorers), RC (both)
Reported: Delta area vs split CP and vs RBE-CP (all13 + R) on es-val and on ImageNet-C, and the selected lambda.
  uv run python scripts/explore_v6.py in_rn50       (on the GPU server; results/explore/v6_<tag>.json)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ablations import LAMS, area, pair, quantiles  # noqa: E402

from tracecal.analysis.bundles import load_bundle, softmax  # noqa: E402
from tracecal.analysis.dense_check import DENSE, curves_boot, window  # noqa: E402
from tracecal.analysis.frontier import P0, RESTORERS, TARGET, _softmax3  # noqa: E402
from tracecal.analysis.run import PoolData  # noqa: E402
from tracecal.cli import esval_conditions, inc_dev_conditions  # noqa: E402
from tracecal.config import families, probe_names, view_names  # noqa: E402
from tracecal.data.sources import imagenet_es_index  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

ANCHORS = ("id", "sym", "all13")
BOOSTS = ("R", "C", "RC")


class Joint:
    """Default-bank views [base, 12 views] + camera views [5] on the same images, in one (n, 18, C) logit array."""

    def __init__(self, z_def: np.ndarray, z_cam: np.ndarray):
        vn = ["base"] + view_names("imagenet")
        cam = probe_names("imagenet", "camera")
        self.z = np.concatenate([z_def, z_cam[:, 1:1 + len(cam)]], axis=1)
        self.P = _softmax3(self.z)
        self.idx = {"R": [vn.index(r) for r in RESTORERS], "C": list(range(len(vn), len(vn) + len(cam)))}
        self.idx["RC"] = self.idx["R"] + self.idx["C"]
        self.flip = vn.index("hflip")
        self.n_def = len(vn)

    def anchor(self, a: str) -> np.ndarray:
        if a == "id":
            return self.P[:, 0]
        if a == "sym":
            return softmax(self.z[:, [0, self.flip]].mean(1))
        return softmax(self.z[:, : self.n_def].mean(1))

    def score(self, a: str, b: str, lam: float) -> np.ndarray:
        an = self.anchor(a)
        if lam == 0:
            return 1.0 - an
        return 1.0 - (an + lam * np.maximum(self.P[:, self.idx[b]].max(1) - an, 0.0))


def joint(tag: str, src: str, version: int, rows=None, subset=None) -> tuple[Joint, np.ndarray]:
    """Rows are taken (and the label space restricted) BEFORE the logits are materialised."""
    d = load_bundle(tag, src, version, cache=False)
    c = load_bundle(tag, src, version, bank="camera", cache=False)
    assert np.array_equal(d.ids, c.ids)
    if rows is not None:
        d, c = d.take(rows), c.take(rows)
    if subset is not None:
        d, c = d.restrict(subset), c.restrict(subset)
    return Joint(d.all_view_logits(), c.all_view_logits()), d.labels


def curves(J: Joint, y: np.ndarray, key, lam_of, q) -> tuple[np.ndarray, np.ndarray]:
    U = (1.0 - J.P[:, 0]) if key == "split" else J.score(*key, lam_of[key])
    cp, sz, _, _ = curves_boot(U, y, q[key])
    return np.maximum(0.0, TARGET - cp), sz


def main() -> None:
    tag = sys.argv[1]
    t0 = time.time()
    subset = imagenet_es_index("es-val")["class_subset"]
    names = [(a, b) for a in ANCHORS for b in BOOSTS]
    tests = {"es_val": [joint(tag, s, 0, subset=subset) for _, s in esval_conditions()],
             "imagenet_c": [joint(tag, s, 0) for _, s in inc_dev_conditions()]}
    print(f"[{tag}] loaded tests ({time.time() - t0:.0f}s)", flush=True)
    out = {"tag": tag, "runs": []}
    for r in range(3):
        pd = PoolData(tag, "imagenet", r, with_views=True)
        cal_pos, tune_pos, K = pd.pos["cal"], pd.pos["tune"], len(pd.versions)
        # lambda selection ONCE per model, full label space (the RBE-CP rule)
        Jc, yc = joint(tag, "imagenet:pool", 0, rows=cal_pos)
        qsel = {(a, b, l): quantiles(Jc.score(a, b, l)[np.arange(len(yc)), yc]) for a, b in names for l in LAMS}
        qsel["split"] = quantiles(1.0 - Jc.P[:, 0][np.arange(len(yc)), yc])
        mcs = {k: np.zeros(len(DENSE)) for k in qsel}
        size = {k: np.zeros(len(DENSE)) for k in qsel}
        for v in range(K):
            Jt, yt = joint(tag, "imagenet:pool", v, rows=tune_pos)
            for k in qsel:
                U = (1.0 - Jt.P[:, 0]) if k == "split" else Jt.score(*k)
                cp, sz, _, _ = curves_boot(U, yt, qsel[k])
                mcs[k] += np.maximum(0.0, TARGET - cp) / K
                size[k] += sz / K
        w = window(size["split"])
        lam = {}
        for a, b in names:
            ar = {l: area(size[(a, b, l)], mcs[(a, b, l)], w) for l in LAMS}
            fin = {l: x for l, x in ar.items() if np.isfinite(x)}
            lam[(a, b)] = min(LAMS, key=lambda l: fin.get(l, np.inf))
        res = {"resplit": r, "lam": {f"{a}+{b}": lam[(a, b)] for a, b in names}, "delta_vs_split": {},
               "delta_vs_rbe": {}}
        for bench, sub in (("es_val", subset), ("imagenet_c", None)):
            Jb, yb = joint(tag, "imagenet:pool", 0, rows=cal_pos, subset=sub)     # clean calibration
            q = {k: quantiles(Jb.score(*k, lam[k])[np.arange(len(yb)), yb]) for k in names}
            q["split"] = quantiles(1.0 - Jb.P[:, 0][np.arange(len(yb)), yb])
            pm = {k: np.zeros(len(DENSE)) for k in names + ["split"]}
            ps = {k: np.zeros(len(DENSE)) for k in names + ["split"]}
            for J, y in tests[bench]:
                for k in names + ["split"]:
                    m, sz = curves(J, y, k, lam, q)
                    pm[k] += m / len(tests[bench])
                    ps[k] += sz / len(tests[bench])
            wb = window(ps["split"])
            res["delta_vs_split"][bench] = {f"{a}+{b}": pair(ps[(a, b)], pm[(a, b)], ps["split"], pm["split"], wb)
                                            for a, b in names}
            res["delta_vs_rbe"][bench] = {f"{a}+{b}": pair(ps[(a, b)], pm[(a, b)], ps[("all13", "R")],
                                                           pm[("all13", "R")], wb) for a, b in names}
        out["runs"].append(res)
        print(f"[{tag}] r{r} ({time.time() - t0:.0f}s) lam", res["lam"], flush=True)
        for bn in res["delta_vs_split"]:
            print(f"   {bn} vs split", {k: round(v, 2) for k, v in res["delta_vs_split"][bn].items()}, flush=True)
    p = get_paths().results_root / "explore" / f"v6_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("written", p)


if __name__ == "__main__":
    main()
