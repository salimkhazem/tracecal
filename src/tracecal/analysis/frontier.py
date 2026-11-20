"""Confirmatory evaluation for pre-registration v2 (audits/gates_v2.yaml): coverage / set-size frontiers.

For each method m and calibration level alpha' (25 log-spaced values in [0.001, 0.30]) we record per-condition
coverage and mean set size (point values and B bootstrap draws over test images, the draws shared by all methods,
conditions and runs of the same dataset). Methods (all LAC-type unless stated):
  base         1 - p_y(x)                              threshold from base score, clean cal
  tta_avg      1 - softmax(mean_k z_k)_y               own-score threshold, clean cal (Shanmugam'25 aggregation)
  tta_learned  1 - softmax(sum_k theta_k z_k)_y        theta by CE on the clean tune split; own threshold, clean cal
  union_all    min over all 13 views of 1 - p_y(T_k x)  threshold from BASE score, clean cal
  ru           min over identity + 6 restorers          threshold from BASE score, clean cal   (RU-CP, ours)
  base_pprime  base score, threshold from P' cal (p0 = 0.35; Tomani-style synthetic-shift calibration)
  aps_base     APS on the base view, clean cal
Also computed (H3/H4): restorability gaps KS+ on the fit split and RU-CP coverage on held-out tune-split programs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import families, view_names
from ..conformal.core import aps_all, conformal_quantile
from ..paths import get_paths
from ..shifts.programs import program
from .bundles import load_bundle, softmax
from .methods import TTALearned
from .pprime import severity
from .run import PoolData, test_conditions

ALPHAS = np.geomspace(0.001, 0.30, 25)
RESTORERS = ["bright_up", "bright_down", "contrast_up", "sat_up", "sharpen", "blur"]
METHODS = ["base", "tta_avg", "tta_learned", "union_all", "ru", "base_pprime", "aps_base"]
P0 = 0.35
TARGET = 0.90
N_BOOT = 500


def _softmax3(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def boot_weights(n: int, dataset: str, n_boot: int = N_BOOT) -> np.ndarray:
    """(n, B) multinomial counts, identical for every run of the same dataset (shared image resampling)."""
    seed = {"cifar10": 101, "cifar100": 102, "imagenet": 103}.get(dataset, 199)
    rng = np.random.default_rng(seed)
    w = np.empty((n, n_boot), dtype=np.float32)
    for b in range(n_boot):
        w[:, b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return w


class Scorer:
    """Builds per-label score matrices for every method from a (n, V+1, C) view-logit array."""

    def __init__(self, family: str, theta: np.ndarray):
        vn = ["base"] + view_names(family)
        self.ru_idx = [0] + [vn.index(v) for v in RESTORERS]
        self.theta = theta

    def matrices(self, vw: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
        P = _softmax3(vw)
        lac = 1.0 - P
        return {
            "base": lac[:, 0],
            "tta_avg": 1.0 - softmax(vw.mean(1)),
            "tta_learned": 1.0 - softmax(np.einsum("nmc,m->nc", vw, self.theta)),
            "union_all": lac.min(1),
            "ru": lac[:, self.ru_idx].min(1),
            "base_pprime": lac[:, 0],
            "aps_base": aps_all(P[:, 0], u),
        }


# methods whose threshold comes from the BASE score
_BASE_THRESHOLD = {"union_all": "base", "ru": "base"}


def thresholds(scorer: Scorer, clean_views: np.ndarray, clean_y: np.ndarray, pp_views: np.ndarray,
               pp_y: np.ndarray) -> dict[str, np.ndarray]:
    u_c = np.random.default_rng(0).random(len(clean_y))
    mats = scorer.matrices(clean_views, u_c)
    q: dict[str, np.ndarray] = {}
    for m in METHODS:
        if m == "base_pprime":
            u_p = np.random.default_rng(0).random(len(pp_y))
            t = scorer.matrices(pp_views, u_p)["base"][np.arange(len(pp_y)), pp_y]
        else:
            src = _BASE_THRESHOLD.get(m, m)
            t = mats[src][np.arange(len(clean_y)), clean_y]
        q[m] = np.array([conformal_quantile(t, a) for a in ALPHAS])
    return q


def cov_size(U: np.ndarray, y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-example coverage (n, A) bool and set size (n, A) of {y : U[i, y] <= q[a]} (q = +inf -> full set)."""
    n, c = U.shape
    covered = U[np.arange(n), y][:, None] <= q[None, :]
    srt = np.sort(U.astype(np.float64), axis=1)
    fq = q[np.isfinite(q)]
    lo = min(float(srt.min()), float(fq.min()) if fq.size else 0.0)
    hi = max(float(srt.max()), float(fq.max()) if fq.size else 0.0)
    span = (hi - lo) + 1.0
    base = (np.arange(n, dtype=np.float64) * span)[:, None]
    flat = ((srt - lo) + base).ravel()
    qq = np.where(np.isfinite(q), q - lo, span - 0.5)
    idx = np.searchsorted(flat, (base + qq[None, :]).ravel(), side="right").reshape(n, len(q))
    return covered, (idx - (np.arange(n) * c)[:, None]).astype(np.int32)


def evaluate_conditions(tag: str, conds: list[tuple[str, str]], scorer: Scorer, q: dict[str, np.ndarray],
                        dataset: str, subset=None) -> dict[str, np.ndarray]:
    A, M, K = len(ALPHAS), len(METHODS), len(conds)
    pc = np.zeros((M, A, K), np.float32)
    ps = np.zeros((M, A, K), np.float32)
    bc = bs = None
    w = None
    for ci, (cname, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        if subset is not None:
            b = b.restrict(subset)
        vw = b.all_view_logits()
        y = b.labels
        n = len(y)
        if w is None or w.shape[0] != n:
            w = boot_weights(n, dataset if subset is None else f"{dataset}-{cname}")
            if bc is None:
                bc = np.zeros((M, A, K, w.shape[1]), np.float32)
                bs = np.zeros((M, A, K, w.shape[1]), np.float32)
        u = np.random.default_rng(1).random(n)
        mats = scorer.matrices(vw, u)
        for mi, m in enumerate(METHODS):
            cov, size = cov_size(mats[m], y, q[m])
            pc[mi, :, ci] = cov.mean(0)
            ps[mi, :, ci] = size.mean(0)
            bc[mi, :, ci] = (cov.T.astype(np.float32) @ w) / n
            bs[mi, :, ci] = (size.T.astype(np.float32) @ w) / n
    return {"point_cov": pc, "point_size": ps, "boot_cov": bc, "boot_size": bs}


def ks_plus(clean_scores: np.ndarray, other_scores: np.ndarray) -> float:
    """sup_t [F_clean(t) - F_other(t)]_+ (other stochastically larger -> coverage loss)."""
    a, b = np.sort(clean_scores), np.sort(other_scores)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return max(0.0, float(np.max(fa - fb)))


def restorability(pd: PoolData, scorer: Scorer, q10: float) -> dict[str, Any]:
    """KS+ gaps of each bank member on the FIT split (per family and per family x severity quintile), and RU-CP
    coverage on the TUNE split programs at alpha' = 0.1 with the clean base threshold q10 (H3 bound, H4)."""
    fams = families()
    out: dict[str, Any] = {"families": {}, "q_base_alpha0.1": q10}
    fit, tune = pd.pos["fit"], pd.pos["tune"]
    clean_fit = pd.versions[0].take(fit)
    s_clean = 1.0 - _softmax3(clean_fit.all_view_logits())[np.arange(len(fit)), 0, clean_fit.labels]
    names = ["identity"] + RESTORERS
    for f in fams:
        v = fams.index(f) + 1
        bf = pd.versions[v].take(fit)
        Pf = _softmax3(bf.all_view_logits())
        yf = bf.labels
        sf = {nm: 1.0 - Pf[np.arange(len(yf)), k, yf] for nm, k in zip(names, scorer.ru_idx)}
        gaps_all = {nm: ks_plus(s_clean, sc) for nm, sc in sf.items()}
        sev_fit = np.array([severity(program(int(i), v, pd.family)[0]) for i in pd.ids[fit]])
        bt = pd.versions[v].take(tune)
        Pt = _softmax3(bt.all_view_logits())
        yt = bt.labels
        cov_t = (1.0 - Pt[:, scorer.ru_idx]).min(1)[np.arange(len(yt)), yt] <= q10
        sev_tune = np.array([severity(program(int(i), v, pd.family)[0]) for i in pd.ids[tune]])
        edges = np.quantile(sev_fit, np.linspace(0, 1, 6))
        qbins = []
        for kq in range(5):
            mf = (sev_fit >= edges[kq]) & (sev_fit <= edges[kq + 1])
            mt = (sev_tune >= edges[kq]) & (sev_tune <= edges[kq + 1])
            gap = min(ks_plus(s_clean, sc[mf]) for sc in sf.values()) if mf.sum() > 20 else float("nan")
            qbins.append({"quintile": kq, "gap": gap, "n_fit": int(mf.sum()), "n_tune": int(mt.sum()),
                          "ru_cov_tune": float(cov_t[mt].mean()) if mt.sum() else None})
        out["families"][f] = {"gap_min": min(gaps_all.values()), "gaps": gaps_all,
                              "ru_cov_tune_all": float(cov_t.mean()), "quintiles": qbins}
    return out


def run_frontier(tag: str, dataset: str, resplit: int, naturals: bool = False, out_root: Path | None = None,
                 conditions: list[tuple[str, str]] | None = None, allow_scaled: bool = False) -> Path:
    t0 = time.time()
    out = (out_root or get_paths().raw_results_root / "frontier") / tag / f"r{resplit}"
    if (out / ".ok").exists():
        print(f"skip (done): {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    fams = families()
    pd = PoolData(tag, dataset, resplit, with_views=True, allow_scaled=allow_scaled)
    ctx = pd.ctx(P0, fams, 0.1, "lac", None, views=True)
    theta = TTALearned("clean", True).fit(ctx).theta
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    scorer = Scorer(fam, theta)
    q = thresholds(scorer, ctx.clean_cal_views, ctx.clean_cal.labels, ctx.pcal_views, ctx.pcal.labels)
    conds = conditions if conditions is not None else test_conditions(dataset)
    res = evaluate_conditions(tag, conds, scorer, q, dataset)
    np.savez_compressed(out / "frontier.npz", **res, alphas=ALPHAS)
    meta: dict[str, Any] = {"tag": tag, "dataset": dataset, "resplit": resplit, "methods": METHODS,
                            "conditions": [c for c, _ in conds], "alphas": ALPHAS.tolist(), "theta": theta.tolist(),
                            "thresholds": {m: v.tolist() for m, v in q.items()}, "restorers": RESTORERS,
                            "p0_pprime": P0, "n_boot": N_BOOT}
    t10 = 1.0 - softmax(ctx.clean_cal_views[:, 0])[np.arange(len(ctx.clean_cal.labels)), ctx.clean_cal.labels]
    q10 = conformal_quantile(t10, 0.1)
    meta["q_base_alpha0.1"] = q10
    meta["restorability"] = restorability(pd, scorer, q10)
    if naturals and dataset == "imagenet":
        from ..data.imagenet import natural_index

        meta["naturals"] = {}
        for name in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
            subset = natural_index(name)["class_subset"]
            ctx_s = pd.ctx(P0, fams, 0.1, "lac", None, subset=subset, views=True)
            q_s = thresholds(scorer, ctx_s.clean_cal_views, ctx_s.clean_cal.labels, ctx_s.pcal_views,
                             ctx_s.pcal.labels)
            r = evaluate_conditions(tag, [(name, name)], scorer, q_s, dataset, subset=subset)
            np.savez_compressed(out / f"frontier_{name}.npz", **r, alphas=ALPHAS)
            meta["naturals"][name] = {"n_classes": None if subset is None else len(subset)}
    meta["seconds"] = time.time() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / ".ok").write_text("ok")
    print(f"done {out} ({meta['seconds']:.0f}s)", flush=True)
    return out


def run_naturals(tag: str, resplit: int, out_root: Path | None = None) -> Path:
    """H5 (report only): natural-shift frontiers for an existing ImageNet frontier run (same theta, same methods;
    thresholds recomputed on the class subset for ImageNet-R/-A). Writes frontier_<name>.npz and naturals.json."""
    t0 = time.time()
    out = (out_root or get_paths().raw_results_root / "frontier") / tag / f"r{resplit}"
    if (out / ".ok_naturals").exists():
        return out
    meta = json.loads((out / "meta.json").read_text())
    from ..data.imagenet import natural_index

    fams = families()
    pd = PoolData(tag, "imagenet", resplit, with_views=True)
    scorer = Scorer("imagenet", np.asarray(meta["theta"], dtype=np.float64))
    info: dict[str, Any] = {}
    for name in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
        subset = natural_index(name)["class_subset"]
        ctx_s = pd.ctx(P0, fams, 0.1, "lac", None, subset=subset, views=True)
        q_s = thresholds(scorer, ctx_s.clean_cal_views, ctx_s.clean_cal.labels, ctx_s.pcal_views, ctx_s.pcal.labels)
        r = evaluate_conditions(tag, [(name, name)], scorer, q_s, "imagenet", subset=subset)
        np.savez_compressed(out / f"frontier_{name}.npz", **r, alphas=ALPHAS)
        info[name] = {"n_classes": None if subset is None else len(subset),
                      "thresholds": {m: v.tolist() for m, v in q_s.items()}}
    info["seconds"] = time.time() - t0
    (out / "naturals.json").write_text(json.dumps(info, indent=1))
    (out / ".ok_naturals").write_text("ok")
    print(f"done naturals {out} ({info['seconds']:.0f}s)", flush=True)
    return out


def run_frontier_multi(tag: str, dataset: str, resplits, out_root: Path | None = None) -> list[Path]:
    """Same outputs as run_frontier for several re-splits of one model, reading each test condition ONCE.

    Per re-split: own calibration split, TTA weights, thresholds, restorability (exactly as run_frontier). Per test
    condition: one load, score matrices shared across re-splits (only TTA-learned depends on the re-split), coverage /
    size via the binned computation (identical to cov_size, tests/test_dense_check.py)."""
    from .dense_check import cov_size_fast

    t0 = time.time()
    root = out_root or get_paths().raw_results_root / "frontier"
    todo = [int(r) for r in resplits if not (root / tag / f"r{int(r)}" / ".ok").exists()]
    if not todo:
        print(f"skip (done): {tag} {list(resplits)}")
        return []
    fams = families()
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    setups: dict[int, dict[str, Any]] = {}
    for r in todo:
        pd = PoolData(tag, dataset, r, with_views=True)
        ctx = pd.ctx(P0, fams, 0.1, "lac", None, views=True)
        theta = TTALearned("clean", True).fit(ctx).theta
        scorer = Scorer(fam, theta)
        q = thresholds(scorer, ctx.clean_cal_views, ctx.clean_cal.labels, ctx.pcal_views, ctx.pcal.labels)
        t10 = 1.0 - softmax(ctx.clean_cal_views[:, 0])[np.arange(len(ctx.clean_cal.labels)), ctx.clean_cal.labels]
        q10 = conformal_quantile(t10, 0.1)
        setups[r] = {"scorer": scorer, "q": q, "theta": theta, "q10": q10,
                     "restorability": restorability(pd, scorer, q10), "t_setup": time.time() - t0}
        del ctx, pd
        print(f"[{tag}] setup r{r} done ({time.time() - t0:.0f}s)", flush=True)
    conds = test_conditions(dataset)
    A, M, K = len(ALPHAS), len(METHODS), len(conds)
    acc = {r: {"point_cov": np.zeros((M, A, K), np.float32), "point_size": np.zeros((M, A, K), np.float32),
               "boot_cov": None, "boot_size": None} for r in todo}
    w = None
    for ci, (cname, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        vw = b.all_view_logits()
        y = b.labels
        n = len(y)
        if w is None or w.shape[0] != n:
            w = boot_weights(n, dataset)
            for r in todo:
                if acc[r]["boot_cov"] is None:
                    acc[r]["boot_cov"] = np.zeros((M, A, K, w.shape[1]), np.float32)
                    acc[r]["boot_size"] = np.zeros((M, A, K, w.shape[1]), np.float32)
        u = np.random.default_rng(1).random(n)
        mats0 = setups[todo[0]]["scorer"].matrices(vw, u)
        for r in todo:
            mats = dict(mats0)
            if r != todo[0]:
                mats["tta_learned"] = 1.0 - softmax(np.einsum("nmc,m->nc", vw, setups[r]["theta"]))
            q = setups[r]["q"]
            a = acc[r]
            for mi, m in enumerate(METHODS):
                cov, size = cov_size_fast(mats[m], y, q[m])
                a["point_cov"][mi, :, ci] = cov.mean(0)
                a["point_size"][mi, :, ci] = size.mean(0)
                a["boot_cov"][mi, :, ci] = (cov.T.astype(np.float32) @ w) / n
                a["boot_size"][mi, :, ci] = (size.T.astype(np.float32) @ w) / n
        if ci % 16 == 0:
            print(f"[{tag}] condition {ci + 1}/{K} ({time.time() - t0:.0f}s)", flush=True)
    outs = []
    for r in todo:
        out = root / tag / f"r{r}"
        out.mkdir(parents=True, exist_ok=True)
        s = setups[r]
        np.savez_compressed(out / "frontier.npz", **acc[r], alphas=ALPHAS)
        meta: dict[str, Any] = {"tag": tag, "dataset": dataset, "resplit": r, "methods": METHODS,
                                "conditions": [c for c, _ in conds], "alphas": ALPHAS.tolist(),
                                "theta": s["theta"].tolist(), "thresholds": {m: v.tolist() for m, v in s["q"].items()},
                                "restorers": RESTORERS, "p0_pprime": P0, "n_boot": N_BOOT, "q_base_alpha0.1": s["q10"],
                                "restorability": s["restorability"], "seconds": time.time() - t0,
                                "runner": "run_frontier_multi", "resplits_in_run": todo}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        (out / ".ok").write_text("ok")
        outs.append(out)
        print(f"done {out} ({meta['seconds']:.0f}s)", flush=True)
    return outs
