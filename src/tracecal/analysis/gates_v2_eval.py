"""Evaluate pre-registration v2 gates (audits/gates_v2.yaml) from results/raw/frontier/<tag>/r<k>/.

AUC statistic: for a pair (RU, baseline b) on one run, MCS(size) curves from the alpha' sweep (point or per bootstrap
draw); window W = [S_base(0.10), S_base(0.01)] (base-LAC mean size interpolated at alpha'=0.10 and 0.01 on a log
alpha axis), intersected with the size ranges of both curves; normalised trapezoid area on 64 grid points.
Delta = AUC_RU - AUC_b (< 0 means RU better). Hierarchical bootstrap: draws b are shared image resamples; for each
draw, resample seeds (CIFAR) then re-splits, average Delta. p = 2 min(P(D*>=0), P(D*<=0)); Holm over 6 baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import get_paths
from ..utils.io import read_json
from ..evaluation.stats import holm, spearman_boot_ci
from .frontier import ALPHAS, METHODS

BASELINES = ["base", "tta_avg", "tta_learned", "union_all", "base_pprime", "aps_base"]
TARGET = 0.90
GRID = 64
CORR_FAMILY = {"noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
               "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
               "brightness": ["brightness"], "contrast": ["contrast", "fog"], "jpeg": ["jpeg_compression"],
               "downup": ["pixelate"], "saturation": ["saturate"]}


def frontier_dir(tag: str, r: int) -> Path:
    return get_paths().raw_results_root / "frontier" / tag / f"r{r}"


def load(tag: str, r: int):
    d = frontier_dir(tag, r)
    if not (d / ".ok").exists():
        raise FileNotFoundError(d)
    return read_json(d / "meta.json"), np.load(d / "frontier.npz")


def curves(cov: np.ndarray, size: np.ndarray, cond_idx: np.ndarray | None = None):
    """cov/size: (M, A, K[, B]) -> mcs, msize: (M, A[, B]) averaged over conditions (subset cond_idx)."""
    if cond_idx is not None:
        cov, size = cov[:, :, cond_idx], size[:, :, cond_idx]
    return np.maximum(0.0, TARGET - cov).mean(2), size.mean(2)


def window(base_size: np.ndarray) -> tuple[float, float]:
    la = np.log(ALPHAS)
    return float(np.interp(np.log(0.10), la, base_size)), float(np.interp(np.log(0.01), la, base_size))


def auc(size: np.ndarray, mcs: np.ndarray, lo: float, hi: float) -> float:
    o = np.argsort(size, kind="stable")
    s, m = size[o], mcs[o]
    xs = np.linspace(lo, hi, GRID)
    return float(np.trapezoid(np.interp(xs, s, m), xs) / (hi - lo))


def pair_delta(size: np.ndarray, mcs: np.ndarray, i_ru: int, i_b: int, w: tuple[float, float]) -> float:
    lo = max(w[0], size[i_ru].min(), size[i_b].min())
    hi = min(w[1], size[i_ru].max(), size[i_b].max())
    if hi - lo < 0.1 * (w[1] - w[0]):
        return float("nan")
    return auc(size[i_ru], mcs[i_ru], lo, hi) - auc(size[i_b], mcs[i_b], lo, hi)


def run_deltas(tag: str, r: int, cond_names: list[str] | None = None) -> dict[str, Any]:
    meta, d = load(tag, r)
    conds = meta["conditions"]
    idx = None if cond_names is None else np.array([conds.index(c) for c in cond_names])
    mi = {m: i for i, m in enumerate(METHODS)}
    pm, ps = curves(d["point_cov"], d["point_size"], idx)
    bm, bsz = curves(d["boot_cov"], d["boot_size"], idx)   # (M, A, B)
    w = window(ps[mi["base"]])
    out: dict[str, Any] = {"window": w, "point": {}, "boot": {}}
    for b in BASELINES:
        out["point"][b] = pair_delta(ps, pm, mi["ru"], mi[b], w)
        nb = bm.shape[-1]
        out["boot"][b] = np.array([pair_delta(bsz[..., k], bm[..., k], mi["ru"], mi[b], window(bsz[mi["base"], :, k]))
                                   for k in range(nb)])
    return out


def hierarchical(deltas: dict[tuple[str, int], dict], groups: dict[tuple[str, int], str] | None, b: str,
                 seed: int = 0) -> dict[str, float]:
    keys = sorted(deltas)
    point = np.nanmean([deltas[k]["point"][b] for k in keys])
    nb = len(next(iter(deltas.values()))["boot"][b])
    rng = np.random.default_rng(seed)
    groups = groups or {k: k for k in keys}
    gkeys = sorted(set(groups.values()))
    members = {g: [k for k in keys if groups[k] == g] for g in gkeys}
    draws = []
    for j in range(nb):
        vals = []
        for gi in rng.choice(len(gkeys), size=len(gkeys), replace=True):
            mem = members[gkeys[gi]]
            for t in rng.choice(len(mem), size=len(mem), replace=True):
                vals.append(deltas[mem[t]]["boot"][b][j])
        draws.append(np.nanmean(vals))
    draws = np.array(draws)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    p = float(min(1.0, 2 * min(np.nanmean(draws >= 0), np.nanmean(draws <= 0))))
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "p_two_sided": p, "n_runs": len(keys)}


def decide(deltas, groups) -> dict[str, Any]:
    res = {b: hierarchical(deltas, groups, b) for b in BASELINES}
    hl = holm({b: v["p_two_sided"] for b, v in res.items()}, 0.05)
    for b in BASELINES:
        res[b]["holm_reject"] = bool(hl[b]["reject"])
    passed = all(res[b]["holm_reject"] and res[b]["delta"] < 0 for b in BASELINES)
    return {"comparisons": res, "pass": bool(passed)}


def h1(seeds=(1, 2, 3, 4), resplits=range(5)) -> dict[str, Any]:
    out: dict[str, Any] = {"pairs": {}}
    for ds, short in (("cifar10", "c10"), ("cifar100", "c100")):
        for arch in ("resnet18", "vits16"):
            deltas, groups = {}, {}
            for s in seeds:
                tag = f"{short}_{arch}_s{s}"
                for r in resplits:
                    try:
                        deltas[(tag, r)] = run_deltas(tag, r)
                        groups[(tag, r)] = tag
                    except FileNotFoundError:
                        pass
            if deltas:
                out["pairs"][f"{ds}/{arch}"] = decide(deltas, groups)
    n = sum(v["pass"] for v in out["pairs"].values())
    out["n_pass"], out["verdict"] = n, ("PASS" if n >= 3 and len(out["pairs"]) == 4 else "FAIL" if len(out["pairs"]) == 4 else "INCOMPLETE")
    return out


def h2(models=("in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"), resplits=range(5)) -> dict[str, Any]:
    out: dict[str, Any] = {"models": {}}
    for m in models:
        deltas = {}
        for r in resplits:
            try:
                deltas[(m, r)] = run_deltas(m, r)
            except FileNotFoundError:
                pass
        if deltas:
            out["models"][m] = decide(deltas, None)
    n = sum(v["pass"] for v in out["models"].values())
    out["n_pass"], out["verdict"] = n, ("PASS" if n >= 3 and len(out["models"]) == 5 else "FAIL" if len(out["models"]) == 5 else "INCOMPLETE")
    return out


def h3(tags: list[str], resplits=range(5)) -> dict[str, Any]:
    clean, bound_ok, n_cases = [], 0, 0
    for tag in tags:
        for r in resplits:
            try:
                meta, d = load(tag, r)
            except FileNotFoundError:
                continue
            ci = meta["conditions"].index("clean")
            i10 = None
            # RU coverage on clean at alpha' = 0.1: recompute from the stored threshold sweep by interpolation
            mi = METHODS.index("ru")
            clean.append(float(np.interp(np.log(0.1), np.log(ALPHAS), d["point_cov"][mi, :, ci])))
            for f, info in meta["restorability"]["families"].items():
                for qb in info["quintiles"]:
                    if qb["ru_cov_tune"] is None or not np.isfinite(qb["gap"]):
                        continue
                    n_cases += 1
                    bound_ok += qb["ru_cov_tune"] >= 1 - 0.1 - qb["gap"] - 0.02
            del i10
    frac = bound_ok / max(1, n_cases)
    return {"clean_cov_min": float(np.min(clean)) if clean else None, "clean_cov_mean": float(np.mean(clean)) if clean else None,
            "bound_fraction": frac, "n_cases": n_cases,
            "pass": bool(clean and min(clean) >= 0.89 and frac >= 0.95)}


def h4(tags: list[str], resplits=range(5)) -> dict[str, Any]:
    gaps, reds = [], []
    for tag in tags:
        g_acc, r_acc = {f: [] for f in CORR_FAMILY}, {f: [] for f in CORR_FAMILY}
        for r in resplits:
            try:
                meta, _ = load(tag, r)
            except FileNotFoundError:
                continue
            conds = meta["conditions"]
            for f, cl in CORR_FAMILY.items():
                names = [c for c in conds if c.split(":")[0] in cl]
                dd = run_deltas(tag, r, names)
                r_acc[f].append(-dd["point"]["base"])      # reduction = AUC_base - AUC_RU
                g_acc[f].append(meta["restorability"]["families"][f]["gap_min"])
        for f in CORR_FAMILY:
            if g_acc[f]:
                gaps.append(np.nanmean(g_acc[f]))
                reds.append(np.nanmean(r_acc[f]))
    if len(gaps) < 5:
        return {"n_pairs": len(gaps), "pass": None}
    sp = spearman_boot_ci(np.array(gaps), np.array(reds))
    return {"n_pairs": len(gaps), **sp, "pass": bool(sp["rho"] <= -0.4 and sp["ci_hi"] < 0)}


NATURALS = ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")


def natural_deltas(tag: str, r: int, name: str) -> dict[str, Any]:
    """Same statistic as run_deltas on a single natural-shift condition (MCS = max(0, 0.9 - coverage))."""
    p = frontier_dir(tag, r) / f"frontier_{name}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p)
    mi = {m: i for i, m in enumerate(METHODS)}
    pm, ps = curves(d["point_cov"], d["point_size"])
    bm, bsz = curves(d["boot_cov"], d["boot_size"])
    w = window(ps[mi["base"]])
    out: dict[str, Any] = {"window": w, "point": {}, "boot": {}}
    for b in BASELINES:
        out["point"][b] = pair_delta(ps, pm, mi["ru"], mi[b], w)
        out["boot"][b] = np.array([pair_delta(bsz[..., k], bm[..., k], mi["ru"], mi[b], window(bsz[mi["base"], :, k]))
                                   for k in range(bm.shape[-1])])
    la = np.log(ALPHAS)
    out["op"] = {m: {"cov": float(np.interp(np.log(0.1), la, d["point_cov"][mi[m], :, 0])),
                     "size": float(np.interp(np.log(0.1), la, d["point_size"][mi[m], :, 0]))} for m in METHODS}
    return out


def h5(models=("in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"), resplits=range(5)) -> dict[str, Any]:
    """REPORT ONLY (pre-registered, no pass/fail): RU-CP vs baselines on ImageNet-V2/-Sketch/-R/-A."""
    out: dict[str, Any] = {"report_only": True, "models": {}}
    for m in models:
        for name in NATURALS:
            deltas = {}
            for r in resplits:
                try:
                    deltas[(m, r)] = natural_deltas(m, r, name)
                except FileNotFoundError:
                    pass
            if not deltas:
                continue
            res = {b: hierarchical(deltas, None, b) for b in BASELINES}
            op = {mm: {k: float(np.mean([v["op"][mm][k] for v in deltas.values()])) for k in ("cov", "size")}
                  for mm in METHODS}
            out["models"].setdefault(m, {})[name] = {"comparisons": res, "op_alpha0.1": op}
    return out


def write(results: dict[str, Any]) -> Path:
    p = get_paths().project_root / "audits" / "gates_v2_results.json"
    prev = read_json(p) if p.exists() else {}
    prev.update(results)
    p.write_text(json.dumps(prev, indent=1, default=float))
    return p


def pair_runs(tags_groups: dict, a: str, b: str, cond_names=None) -> dict[str, float]:
    """SECONDARY: Delta AUC(a) - AUC(b) for any two methods, same window / bootstrap / hierarchy as H1-H2.
    tags_groups: {(tag, resplit): group_key}."""
    mi = {m: i for i, m in enumerate(METHODS)}
    deltas = {}
    for (tag, r) in tags_groups:
        meta, d = load(tag, r)
        conds = meta["conditions"]
        idx = None if cond_names is None else np.array([conds.index(c) for c in cond_names])
        pm, ps = curves(d["point_cov"], d["point_size"], idx)
        bm, bsz = curves(d["boot_cov"], d["boot_size"], idx)
        w = window(ps[mi["base"]])
        deltas[(tag, r)] = {
            "point": {"x": pair_delta(ps, pm, mi[a], mi[b], w)},
            "boot": {"x": np.array([pair_delta(bsz[..., k], bm[..., k], mi[a], mi[b], window(bsz[mi["base"], :, k]))
                                    for k in range(bm.shape[-1])])}}
    return hierarchical(deltas, tags_groups, "x")
