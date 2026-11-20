"""Pre-registration v3 (audits/gates_v3.yaml): restoration-boosted ensemble conformal prediction (RBE-CP).

Score from the 13 extracted views z_k(x) (identity, 11 probes, hflip) and the six restorers R of v2:
    pbar(x)   = softmax(mean_k z_k(x))                              (test-time ensemble)
    rbe_lam   : 1 - [ pbar_y(x) + lam * ( max_{T in R} p_y(Tx) - pbar_y(x) )_+ ],   lam in LAMBDAS
lam = 0 is the ensemble; lam = 1 the union of the ensemble with the restored views. RBE-CP selects lam on the
labelled synthetic corruptions of the calibration pool (tune split, program versions v0..v8; no test data) by the
efficiency-matched area, then calibrates on clean data (split conformal, own score). Baselines as in v2.

Outputs per run: results/raw/frontier_v3/<tag>/r<k>/{curves.npz, meta.json}. Dense alpha' grid (400 levels), image
bootstrap draws shared with v2 (boot_weights). Every test condition is read once for all pending re-splits.
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
from .bundles import load_bundle, softmax
from .dense_check import DENSE, curves_boot, window
from .frontier import P0, RESTORERS, TARGET, _softmax3, boot_weights
from .gates_v2_eval import auc
from .methods import TTALearned
from .run import PoolData, test_conditions

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BASELINES = ["base", "tta_avg", "tta_learned", "union_all", "ru", "aps_base"]
GATED = BASELINES + ["rbe_sel"]


def lam_name(lam: float) -> str:
    return f"rbe{int(round(100 * lam))}"


class ScorerV3:
    def __init__(self, family: str, theta: np.ndarray):
        vn = ["base"] + view_names(family)
        self.ru_idx = [0] + [vn.index(v) for v in RESTORERS]
        self.r_only = self.ru_idx[1:]
        self.theta = theta

    def matrices(self, vw: np.ndarray, u: np.ndarray, lams=LAMBDAS) -> dict[str, np.ndarray]:
        P = _softmax3(vw)
        pbar = softmax(vw.mean(1))
        rmax = P[:, self.r_only].max(1)
        out = {"base": 1.0 - P[:, 0], "tta_avg": 1.0 - pbar,
               "tta_learned": 1.0 - softmax(np.einsum("nmc,m->nc", vw, self.theta)),
               "union_all": 1.0 - P.max(1), "ru": 1.0 - P[:, self.ru_idx].max(1), "aps_base": aps_all(P[:, 0], u)}
        excess = np.maximum(rmax - pbar, 0.0)
        for lam in lams:
            out[lam_name(lam)] = 1.0 - (pbar + lam * excess)
        return out


def _area(size: np.ndarray, mcs: np.ndarray, w: tuple[float, float]) -> float:
    lo, hi = max(w[0], size.min()), min(w[1], size.max())
    return float("nan") if hi - lo < 0.1 * (w[1] - w[0]) else auc(size, mcs, lo, hi)


def _setup(tag: str, dataset: str, r: int) -> dict[str, Any]:
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    pd = PoolData(tag, dataset, r, with_views=True)
    ctx = pd.ctx(P0, families(), 0.1, "lac", None, views=True)
    theta = TTALearned("clean", True).fit(ctx).theta
    sc = ScorerV3(fam, theta)
    yc = ctx.clean_cal.labels
    mats_c = sc.matrices(ctx.clean_cal_views, np.random.default_rng(0).random(len(yc)))
    q = {m: np.array([conformal_quantile(s[np.arange(len(yc)), yc], a) for a in DENSE]) for m, s in mats_c.items()}
    # lam selection on the tune split's synthetic versions (labelled, no test data)
    tune = pd.pos["tune"]
    K = len(pd.versions)
    names = ["base"] + [lam_name(l) for l in LAMBDAS]
    mcs = {m: np.zeros(len(DENSE)) for m in names}
    size = {m: np.zeros(len(DENSE)) for m in names}
    for v in range(K):
        b = pd.versions[v].take(tune)
        mats = sc.matrices(b.all_view_logits(), np.random.default_rng(2).random(len(b.labels)))
        for m in names:
            cp, sz, _, _ = curves_boot(mats[m], b.labels, q[m])
            mcs[m] += np.maximum(0.0, TARGET - cp) / K
            size[m] += sz / K
    w = window(size["base"])
    sel_area = {lam_name(l): _area(size[lam_name(l)], mcs[lam_name(l)], w) for l in LAMBDAS}
    finite = {k: v for k, v in sel_area.items() if np.isfinite(v)}
    lam_star = min(LAMBDAS, key=lambda l: finite.get(lam_name(l), np.inf))
    # guaranteed level under exact restoration (Prop. RBE): P_clean(p_Y(X) >= (1 - qhat) / lam), at alpha' = 0.1
    i10 = int(np.argmin(np.abs(np.log(DENSE) - np.log(0.1))))
    p_true = 1.0 - mats_c["base"][np.arange(len(yc)), yc]
    q10 = q[lam_name(lam_star)][i10]
    restore_level = float(np.mean(p_true >= (1.0 - q10) / lam_star)) if lam_star > 0 else 0.0
    return {"scorer": sc, "q": q, "theta": theta, "lam_star": lam_star, "sel_area": sel_area, "sel_window": w,
            "restore_level_a01": restore_level, "q_sel_a01": float(q10)}


def run_multi(tag: str, dataset: str, resplits, out_root: Path | None = None) -> list[Path]:
    t0 = time.time()
    root = out_root or get_paths().raw_results_root / "frontier_v3"
    todo = [int(r) for r in resplits if not (root / tag / f"r{int(r)}" / ".ok").exists()]
    if not todo:
        return [root / tag / f"r{int(r)}" for r in resplits]
    st = {r: _setup(tag, dataset, r) for r in todo}
    for r in todo:
        print(f"[{tag}] r{r}: lam* = {st[r]['lam_star']} ({time.time() - t0:.0f}s)", flush=True)
    conds = test_conditions(dataset)
    K, A = len(conds), len(DENSE)
    all_names = BASELINES + [lam_name(l) for l in LAMBDAS] + ["rbe_sel"]
    acc = {r: {"pm": {m: np.zeros(A) for m in all_names}, "ps": {m: np.zeros(A) for m in all_names},
               "bm": None, "bs": None, "clean_cov": {}} for r in todo}
    w = None
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        vw = b.all_view_logits()
        y = b.labels
        n = len(y)
        if w is None:
            w = boot_weights(n, dataset)
            for r in todo:
                acc[r]["bm"] = {m: np.zeros((A, w.shape[1])) for m in GATED}
                acc[r]["bs"] = {m: np.zeros((A, w.shape[1])) for m in GATED}
        u = np.random.default_rng(1).random(n)
        mats0 = st[todo[0]]["scorer"].matrices(vw, u)
        for r in todo:
            s = st[r]
            mats = dict(mats0)
            if r != todo[0]:
                mats["tta_learned"] = 1.0 - softmax(np.einsum("nmc,m->nc", vw, s["theta"]))
            mats["rbe_sel"] = mats[lam_name(s["lam_star"])]
            qs = dict(s["q"])
            qs["rbe_sel"] = s["q"][lam_name(s["lam_star"])]
            a = acc[r]
            for m in all_names:
                boot = m in GATED
                cp, sz, cb, sb = curves_boot(mats[m], y, qs[m], w if boot else None)
                a["pm"][m] += np.maximum(0.0, TARGET - cp) / K
                a["ps"][m] += sz / K
                if boot:
                    a["bm"][m] += np.maximum(0.0, TARGET - cb) / K
                    a["bs"][m] += sb / K
                if cn == "clean":
                    a["clean_cov"][m] = float(np.interp(np.log(0.1), np.log(DENSE), cp))
        if ci % 16 == 0:
            print(f"[{tag}] condition {ci + 1}/{K} ({time.time() - t0:.0f}s)", flush=True)
    outs = []
    for r in todo:
        a, s = acc[r], st[r]
        out = root / tag / f"r{r}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "curves.npz", alphas=DENSE,
                            **{f"pm_{m}": v for m, v in a["pm"].items()}, **{f"ps_{m}": v for m, v in a["ps"].items()},
                            **{f"bm_{m}": v.astype(np.float32) for m, v in a["bm"].items()},
                            **{f"bs_{m}": v.astype(np.float32) for m, v in a["bs"].items()})
        meta = {"tag": tag, "dataset": dataset, "resplit": r, "lambdas": list(LAMBDAS), "lam_star": s["lam_star"],
                "selection_area": s["sel_area"], "selection_window": s["sel_window"], "theta": s["theta"].tolist(),
                "restore_level_a01": s["restore_level_a01"], "q_sel_a01": s["q_sel_a01"],
                "clean_cov_a01": a["clean_cov"], "conditions": [c for c, _ in conds], "seconds": time.time() - t0}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        (out / ".ok").write_text("ok")
        outs.append(out)
        print(f"done {out} ({meta['seconds']:.0f}s)", flush=True)
    return outs


def run_naturals_v3(tag: str, resplits, out_root: Path | None = None) -> list[Path]:
    """K5 (report only): natural-shift curves on the fresh ImageNet models with the lam* of the main v3 run.
    Thresholds are recomputed on the clean calibration split restricted to each natural set's classes."""
    from ..data.imagenet import natural_index

    root = out_root or get_paths().raw_results_root / "frontier_v3"
    outs = []
    for r in resplits:
        out = root / tag / f"r{int(r)}"
        if (out / ".ok_naturals").exists() or not (out / ".ok").exists():
            continue
        meta = json.loads((out / "meta.json").read_text())
        lam = float(meta["lam_star"])
        pd = PoolData(tag, "imagenet", int(r), with_views=True)
        sc = ScorerV3("imagenet", np.asarray(meta["theta"], dtype=np.float64))
        names = ["base", "tta_avg", "ru", "rbe_sel"]
        res: dict[str, Any] = {}
        for nat in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
            subset = natural_index(nat)["class_subset"]
            ctx = pd.ctx(P0, families(), 0.1, "lac", None, subset=subset, views=True)
            yc = ctx.clean_cal.labels
            mc = sc.matrices(ctx.clean_cal_views, np.random.default_rng(0).random(len(yc)), lams=(lam,))
            mc["rbe_sel"] = mc[lam_name(lam)]
            q = {m: np.array([conformal_quantile(mc[m][np.arange(len(yc)), yc], a) for a in DENSE]) for m in names}
            b = load_bundle(tag, nat, 0, cache=False).restrict(subset)
            vw = b.all_view_logits()
            mt = sc.matrices(vw, np.random.default_rng(1).random(len(b.labels)), lams=(lam,))
            mt["rbe_sel"] = mt[lam_name(lam)]
            w = boot_weights(len(b.labels), f"imagenet-{nat}")
            curves = {}
            for m in names:
                cp, sz, cb, sb = curves_boot(mt[m], b.labels, q[m], w)
                curves[m] = (np.maximum(0.0, TARGET - cp), sz, np.maximum(0.0, TARGET - cb), sb)
            res[nat] = {f"{k}_{m}": v.tolist() for m, (pm, ps, bm, bs) in curves.items()
                        for k, v in (("pm", pm), ("ps", ps))}
            np.savez_compressed(out / f"naturals_{nat}.npz", **{f"pm_{m}": c[0] for m, c in curves.items()},
                                **{f"ps_{m}": c[1] for m, c in curves.items()},
                                **{f"bm_{m}": c[2].astype(np.float32) for m, c in curves.items()},
                                **{f"bs_{m}": c[3].astype(np.float32) for m, c in curves.items()})
        (out / ".ok_naturals").write_text("ok")
        outs.append(out)
        print(f"done naturals {out}", flush=True)
    return outs
