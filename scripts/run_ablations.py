#!/usr/bin/env python
"""Ablations v5c (audits/ablations_v5c.yaml; declared before any number). DESCRIPTIVE, no gates.

For one model, from the stored per-view logits (no GPU): every variant with a lambda selects it with the v3 rule (argmin
efficiency-matched area on the tune split's synthetic versions, grid {0,.25,.5,.75,1}; ties -> smallest lambda) and is
calibrated on clean data. Each variant is evaluated over all test conditions; Delta = area(variant) - area(reference)
over the common window (the v3 pair rule). Also: top-1 accuracy, size-stratified coverage violation (SSCV), class-
conditional gap (CIFAR), and the calibration-size ablation on re-split 0.

  uv run python scripts/run_ablations.py in_effnet_b0        (on ws; writes results/tables/ablate_<tag>.json)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracecal.analysis.bundles import load_bundle, softmax  # noqa: E402
from tracecal.analysis.dense_check import DENSE, curves_boot, window  # noqa: E402
from tracecal.analysis.frontier import P0, RESTORERS, TARGET, _softmax3  # noqa: E402
from tracecal.analysis.gates_v2_eval import auc  # noqa: E402
from tracecal.analysis.methods import TTALearned  # noqa: E402
from tracecal.analysis.run import PoolData, test_conditions  # noqa: E402
from tracecal.config import families, view_names  # noqa: E402
from tracecal.conformal.core import aps_all  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIXED_LAMS = (0.25, 0.5, 0.75, 1.0)
I10 = int(np.argmin(np.abs(np.log(DENSE) - np.log(0.1))))
STRATA = [(1, 1), (2, 3), (4, 10), (11, 100), (101, 10**9)]
NCAL = (500, 1000, 2500, 5000)
NCAL_DRAWS = 3
# family: (anchor, boost aggregation, dropped restorer, score)
FAMILIES = {"A2_boost_identity": ("id", "max", None, "lac"), "A3_rbe": ("all", "max", None, "lac"),
            "A4_boost_flip": ("idflip", "max", None, "lac"), "B1_mean_boost": ("all", "mean", None, "lac"),
            "B2_softavg_boost": ("all", "softavg", None, "lac"), "B3_anchor_R7": ("idR", "max", None, "lac"),
            "E_aps_rbe": ("all", "max", None, "aps"),
            "F_clean_select": ("all", "max", None, "lac")}   # = A3, but lambda selected on clean tune images only
for _r in RESTORERS:
    FAMILIES[f"C_loo_{_r}"] = ("all", "max", _r, "lac")
FIXED = ["A0_split", "A1_avg13", "tta_learned", "union13", "ru", "E_aps_split", "E_aps_avg"] + \
    [f"D_fixed_{int(100 * l)}" for l in FIXED_LAMS]


def quantiles(s_true: np.ndarray) -> np.ndarray:
    s = np.sort(np.asarray(s_true, dtype=np.float64))
    n = len(s)
    k = np.ceil((n + 1) * (1.0 - DENSE)).astype(int)
    return np.where(k > n, np.inf, s[np.clip(k - 1, 0, n - 1)])


class Views:
    def __init__(self, vw: np.ndarray, fam: str, theta: dict[int, np.ndarray] | None = None):
        vn = ["base"] + view_names(fam)
        self.R = {r: vn.index(r) for r in RESTORERS}
        self.flip = vn.index("hflip")
        self.z = np.asarray(vw, dtype=np.float32)
        self.P = _softmax3(self.z)
        self.n = len(self.z)
        self._a, self._b = {}, {}

    def anchor(self, kind: str) -> np.ndarray:
        if kind not in self._a:
            R = list(self.R.values())
            self._a[kind] = {"all": lambda: softmax(self.z.mean(1)), "id": lambda: self.P[:, 0],
                             "idR": lambda: softmax(self.z[:, [0] + R].mean(1)),
                             "idflip": lambda: softmax(self.z[:, [0, self.flip]].mean(1))}[kind]()
        return self._a[kind]

    def boost(self, kind: str, drop: str | None) -> np.ndarray:
        key = (kind, drop)
        if key not in self._b:
            R = [i for r, i in self.R.items() if r != drop]
            self._b[key] = {"max": lambda: self.P[:, R].max(1), "mean": lambda: self.P[:, R].mean(1),
                            "softavg": lambda: softmax(self.z[:, R].mean(1))}[kind]()
        return self._b[key]

    def family(self, name: str, lam: float, u: np.ndarray) -> np.ndarray:
        a_kind, b_kind, drop, score = FAMILIES[name]
        a = self.anchor(a_kind)
        pt = a + lam * np.maximum(self.boost(b_kind, drop) - a, 0.0) if lam > 0 else a
        if score == "aps":
            return aps_all(pt / pt.sum(1, keepdims=True), u)
        return 1.0 - pt

    def fixed(self, name: str, theta: np.ndarray, u: np.ndarray) -> np.ndarray:
        if name == "A0_split":
            return 1.0 - self.P[:, 0]
        if name == "A1_avg13":
            return 1.0 - self.anchor("all")
        if name == "tta_learned":
            return 1.0 - softmax(np.einsum("nmc,m->nc", self.z, theta))
        if name == "union13":
            return 1.0 - self.P.max(1)
        if name == "ru":
            return 1.0 - self.P[:, [0] + list(self.R.values())].max(1)
        if name == "E_aps_split":
            return aps_all(self.P[:, 0], u)
        if name == "E_aps_avg":
            return aps_all(self.anchor("all"), u)
        if name.startswith("D_fixed_"):
            return self.family("A3_rbe", int(name.split("_")[-1]) / 100, u)
        raise KeyError(name)


def pair(sv, mv, sr, mr, w) -> float:
    lo, hi = max(w[0], sv.min(), sr.min()), min(w[1], sv.max(), sr.max())
    if hi - lo < 0.1 * (w[1] - w[0]):
        return float("nan")
    return 100 * (auc(sv, mv, lo, hi) - auc(sr, mr, lo, hi))


def area(size, mcs, w) -> float:
    lo, hi = max(w[0], size.min()), min(w[1], size.max())
    return float("nan") if hi - lo < 0.1 * (w[1] - w[0]) else auc(size, mcs, lo, hi)


def setup(tag: str, dataset: str, r: int, fam: str) -> dict:
    pd = PoolData(tag, dataset, r, with_views=True)
    ctx = pd.ctx(P0, families(), 0.1, "lac", None, views=True)
    theta = TTALearned("clean", True).fit(ctx).theta
    Vc = Views(ctx.clean_cal_views, fam)
    yc = ctx.clean_cal.labels
    uc = np.random.default_rng(0).random(len(yc))
    true = lambda M: M[np.arange(len(yc)), yc]  # noqa: E731
    q = {(f, l): quantiles(true(Vc.family(f, l, uc))) for f in FAMILIES for l in LAMS}
    q.update({(f, None): quantiles(true(Vc.fixed(f, theta, uc))) for f in FIXED})
    # lambda selection on the tune split's synthetic versions (labelled; no test data)
    tune = pd.pos["tune"]
    K = len(pd.versions)
    mcs = {k: np.zeros(len(DENSE)) for k in list(q)}
    size = {k: np.zeros(len(DENSE)) for k in list(q)}
    mcs0 = {k: np.zeros(len(DENSE)) for k in list(q)}   # clean tune version (v0) only, for F_clean_select
    size0 = {k: np.zeros(len(DENSE)) for k in list(q)}
    sel_keys = [("A0_split", None)] + [(f, l) for f in FAMILIES for l in LAMS]
    for v in range(K):
        b = pd.versions[v].take(tune)
        Vt = Views(b.all_view_logits(), fam)
        ut = np.random.default_rng(2).random(len(b.labels))
        for k in sel_keys:
            M = Vt.fixed(k[0], theta, ut) if k[1] is None else Vt.family(k[0], k[1], ut)
            cp, sz, _, _ = curves_boot(M, b.labels, q[k])
            mcs[k] += np.maximum(0.0, TARGET - cp) / K
            size[k] += sz / K
            if v == 0:
                mcs0[k] += np.maximum(0.0, TARGET - cp)
                size0[k] += sz
    w = window(size[("A0_split", None)])
    w0 = window(size0[("A0_split", None)])
    lam_star = {}
    for f in FAMILIES:
        if f == "F_clean_select":
            a = {l: area(size0[(f, l)], mcs0[(f, l)], w0) for l in LAMS}
        else:
            a = {l: area(size[(f, l)], mcs[(f, l)], w) for l in LAMS}
        fin = {l: v for l, v in a.items() if np.isfinite(v)}
        lam_star[f] = min(LAMS, key=lambda l: fin.get(l, np.inf))
    # calibration-size ablation thresholds (re-split 0 only): split CP and RBE-CP at its selected lambda
    ncal = {}
    if r == 0:
        rng = np.random.default_rng(7)
        for n in NCAL:
            for d in range(NCAL_DRAWS):
                idx = rng.choice(len(yc), size=min(n, len(yc)), replace=False)
                ncal[(n, d)] = {
                    "A0_split": quantiles(true(Vc.fixed("A0_split", theta, uc))[idx]),
                    "A3_rbe": quantiles(true(Vc.family("A3_rbe", lam_star["A3_rbe"], uc))[idx])}
    return {"q": q, "theta": theta, "lam_star": lam_star, "ncal": ncal}


def sscv(M: np.ndarray, y: np.ndarray, qv: float, min_n: int = 100) -> float:
    sets = M <= qv
    sz = sets.sum(1)
    cov = sets[np.arange(len(y)), y]
    v = [abs(cov[(sz >= lo) & (sz <= hi)].mean() - TARGET) for lo, hi in STRATA if ((sz >= lo) & (sz <= hi)).sum() >= min_n]
    return float(max(v)) if v else float("nan")


def main() -> None:
    tag = sys.argv[1]
    dataset = {"c10": "cifar10", "c100": "cifar100"}.get(tag.split("_")[0], "imagenet")
    fam = "imagenet" if dataset == "imagenet" else "cifar"
    t0 = time.time()
    rs = [0, 1, 2, 3, 4]
    st = {r: setup(tag, dataset, r, fam) for r in rs}
    print(f"[{tag}] setup done ({time.time() - t0:.0f}s); lam*:", {r: st[r]["lam_star"]["A3_rbe"] for r in rs}, flush=True)
    variants = FIXED + list(FAMILIES)
    A = len(DENSE)
    pm = {r: {v: np.zeros(A) for v in variants} for r in rs}
    ps = {r: {v: np.zeros(A) for v in variants} for r in rs}
    npm = {k: {m: np.zeros(A) for m in ("A0_split", "A3_rbe")} for k in st[0]["ncal"]}
    nps = {k: {m: np.zeros(A) for m in ("A0_split", "A3_rbe")} for k in st[0]["ncal"]}
    ncal_clean = {k: {} for k in st[0]["ncal"]}
    top1 = {m: {"clean": [], "corrupted": []} for m in ("identity", "avg13", "rbe_sel", "tta_learned")}
    sv = {m: {"clean": [], "corrupted": []} for m in ("A0_split", "A1_avg13", "tta_learned", "A3_rbe")}
    cgap = {m: {"clean": [], "corrupted": []} for m in ("A0_split", "A1_avg13", "tta_learned", "A3_rbe")}
    conds = test_conditions(dataset)
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        V = Views(b.all_view_logits(), fam)
        y = b.labels
        u = np.random.default_rng(1).random(len(y))
        kind = "clean" if cn == "clean" else "corrupted"
        cache = {}

        def mat(v, r):
            if v in FAMILIES:
                key = (v, st[r]["lam_star"][v])
            elif v == "tta_learned":
                key = (v, r)
            else:
                key = (v, None)
            if key not in cache:
                cache[key] = (V.family(v, key[1], u) if v in FAMILIES else
                              V.fixed(v, st[r]["theta"], u))
            return cache[key]

        for r in rs:
            for v in variants:
                k = (v, st[r]["lam_star"][v]) if v in FAMILIES else (v, None)
                cp, sz, _, _ = curves_boot(mat(v, r), y, st[r]["q"][k])
                pm[r][v] += np.maximum(0.0, TARGET - cp) / len(conds)
                ps[r][v] += sz / len(conds)
            # extra metrics
            for m in sv:
                k = (m, st[r]["lam_star"][m]) if m in FAMILIES else (m, None)
                M = mat(m, r)
                sv[m][kind].append(sscv(M, y, st[r]["q"][k][I10]))
                if fam == "cifar":
                    sets = M <= st[r]["q"][k][I10]
                    covc = [sets[y == c, c].mean() for c in np.unique(y)]
                    cgap[m][kind].append(float(np.mean(np.abs(np.array(covc) - TARGET))))
            th = st[r]["theta"]
            pt = 1.0 - mat("A3_rbe", r)
            top1["rbe_sel"][kind].append(float((pt.argmax(1) == y).mean()))
            top1["tta_learned"][kind].append(float((np.einsum("nmc,m->nc", V.z, th).argmax(1) == y).mean()))
        top1["identity"][kind].append(float((V.P[:, 0].argmax(1) == y).mean()))
        top1["avg13"][kind].append(float((V.anchor("all").argmax(1) == y).mean()))
        for key, qq in st[0]["ncal"].items():
            for m in ("A0_split", "A3_rbe"):
                M = mat(m, 0)
                cp, sz, _, _ = curves_boot(M, y, qq[m])
                npm[key][m] += np.maximum(0.0, TARGET - cp) / len(conds)
                nps[key][m] += sz / len(conds)
                if kind == "clean":
                    ncal_clean[key][m] = float(cp[I10])
        if ci % 16 == 0:
            print(f"[{tag}] {ci + 1}/{len(conds)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"tag": tag, "lam_star": {r: st[r]["lam_star"] for r in rs}, "delta_vs_rbe": {}, "delta_vs_split": {}}
    for v in variants:
        dr, ds = [], []
        for r in rs:
            w = window(ps[r]["A0_split"])
            dr.append(pair(ps[r][v], pm[r][v], ps[r]["A3_rbe"], pm[r]["A3_rbe"], w))
            ds.append(pair(ps[r][v], pm[r][v], ps[r]["A0_split"], pm[r]["A0_split"], w))
        for key, vals in (("delta_vs_rbe", dr), ("delta_vs_split", ds)):
            a = np.array(vals, dtype=float)
            out[key][v] = {"mean": float(np.nanmean(a)), "se": float(np.nanstd(a, ddof=1) / np.sqrt(np.isfinite(a).sum()))
                           if np.isfinite(a).sum() > 1 else float("nan"), "per_resplit": vals}
    out["top1"] = {m: {k: float(np.mean(v)) for k, v in d.items() if v} for m, d in top1.items()}
    out["sscv"] = {m: {k: float(np.nanmean(v)) for k, v in d.items() if v} for m, d in sv.items()}
    if fam == "cifar":
        out["class_gap"] = {m: {k: float(np.nanmean(v)) for k, v in d.items() if v} for m, d in cgap.items()}
    w0 = window(ps[0]["A0_split"])
    out["ncal"] = {}
    for n in NCAL:
        deltas = [pair(nps[(n, d)]["A3_rbe"], npm[(n, d)]["A3_rbe"], nps[(n, d)]["A0_split"], npm[(n, d)]["A0_split"], w0)
                  for d in range(NCAL_DRAWS)]
        cc = {m: [ncal_clean[(n, d)][m] for d in range(NCAL_DRAWS)] for m in ("A0_split", "A3_rbe")}
        out["ncal"][n] = {"delta_rbe_vs_split": float(np.nanmean(deltas)), "deltas": deltas,
                          "clean_cov_a01": {m: [float(np.mean(v)), float(np.std(v))] for m, v in cc.items()}}
    out["ncal"]["full"] = {"delta_rbe_vs_split": out["delta_vs_split"]["A3_rbe"]["per_resplit"][0]}
    p = get_paths().results_root / "tables" / f"ablate_{tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(get_paths().results_root / "tables" / f"ablate_{tag}_curves.npz",
                        **{f"pm__{r}__{v}": pm[r][v] for r in rs for v in variants},
                        **{f"ps__{r}__{v}": ps[r][v] for r in rs for v in variants})
    print("delta vs RBE:", {v: round(d["mean"], 3) for v, d in out["delta_vs_rbe"].items()}, flush=True)
    print("written", p, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
