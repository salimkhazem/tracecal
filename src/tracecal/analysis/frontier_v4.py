"""Pre-registration v4 (audits/gates_v4.yaml): RBE-CP with a symmetric anchor.

v4 score from 8 views (identity, horizontal flip, the six restorers R of v2):
    psym(x)  = softmax( (z(x) + z(flip x)) / 2 )                       (average over the symmetry only)
    rbe4_lam : 1 - [ psym_y(x) + lam * ( max_{T in R} p_y(Tx) - psym_y(x) )_+ ],   lam in LAMBDAS4 = {.25,.5,.75,1}
lam is selected on the labelled synthetic corruptions of the calibration pool (tune split, versions v0..v8) by the
efficiency-matched area (ties -> smallest lam), exactly as in v3; the score is calibrated on clean data.

Computed alongside, on the same data and grid: the v3 baselines (split CP, 13-view TTA average, TTA learned, 13-view
union, RU-CP, APS), the flip-only average (tta_flip), and RBE-CP v3 with its own selection (rbe3_sel).

Outputs per run: results/raw/frontier_v4/<tag>/r<k>/{curves.npz, meta.json}; naturals_<set>.npz for ImageNet models.
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
from .frontier_v3 import _area
from .methods import TTALearned
from .run import PoolData, test_conditions

LAMBDAS3 = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDAS4 = (0.25, 0.5, 0.75, 1.0)
BASELINES = ["base", "tta_avg", "tta_learned", "union_all", "ru", "aps_base", "tta_flip", "rbe3_sel"]
GATED = BASELINES + ["rbe4_sel"]


def n3(lam: float) -> str:
    return f"rbe3_{int(round(100 * lam))}"


def n4(lam: float) -> str:
    return f"rbe4_{int(round(100 * lam))}"


class ScorerV4:
    def __init__(self, family: str, theta: np.ndarray):
        vn = ["base"] + view_names(family)
        self.ru_idx = [0] + [vn.index(v) for v in RESTORERS]
        self.r_only = self.ru_idx[1:]
        self.sym_idx = [0, vn.index("hflip")]
        self.theta = theta

    def matrices(self, vw: np.ndarray, u: np.ndarray, lams3=LAMBDAS3, lams4=LAMBDAS4) -> dict[str, np.ndarray]:
        P = _softmax3(vw)
        pbar = softmax(vw.mean(1))
        psym = softmax(vw[:, self.sym_idx].mean(1))
        rmax = P[:, self.r_only].max(1)
        out = {"base": 1.0 - P[:, 0], "tta_avg": 1.0 - pbar,
               "tta_learned": 1.0 - softmax(np.einsum("nmc,m->nc", vw, self.theta)),
               "union_all": 1.0 - P.max(1), "ru": 1.0 - P[:, self.ru_idx].max(1), "aps_base": aps_all(P[:, 0], u),
               "tta_flip": 1.0 - psym}
        ex3 = np.maximum(rmax - pbar, 0.0)
        for lam in lams3:
            out[n3(lam)] = 1.0 - (pbar + lam * ex3)
        ex4 = np.maximum(rmax - psym, 0.0)
        for lam in lams4:
            out[n4(lam)] = 1.0 - (psym + lam * ex4)
        return out


def _select(size: dict, mcs: dict, names: list[str], lams) -> tuple[float, dict[str, float], tuple[float, float]]:
    w = window(size["base"])
    areas = {nm: _area(size[nm], mcs[nm], w) for nm in names}
    fin = {lam: areas[nm] for lam, nm in zip(lams, names) if np.isfinite(areas[nm])}
    return min(lams, key=lambda l: fin.get(l, np.inf)), areas, w


def _setup(tag: str, dataset: str, r: int) -> dict[str, Any]:
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    pd = PoolData(tag, dataset, r, with_views=True)
    ctx = pd.ctx(P0, families(), 0.1, "lac", None, views=True)
    theta = TTALearned("clean", True).fit(ctx).theta
    sc = ScorerV4(fam, theta)
    yc = ctx.clean_cal.labels
    mats_c = sc.matrices(ctx.clean_cal_views, np.random.default_rng(0).random(len(yc)))
    q = {m: np.array([conformal_quantile(s[np.arange(len(yc)), yc], a) for a in DENSE]) for m, s in mats_c.items()}
    tune = pd.pos["tune"]
    K = len(pd.versions)
    names = ["base"] + [n3(l) for l in LAMBDAS3] + [n4(l) for l in LAMBDAS4]
    mcs = {m: np.zeros(len(DENSE)) for m in names}
    size = {m: np.zeros(len(DENSE)) for m in names}
    for v in range(K):
        b = pd.versions[v].take(tune)
        mats = sc.matrices(b.all_view_logits(), np.random.default_rng(2).random(len(b.labels)))
        for m in names:
            cp, sz, _, _ = curves_boot(mats[m], b.labels, q[m])
            mcs[m] += np.maximum(0.0, TARGET - cp) / K
            size[m] += sz / K
    lam3, areas3, w = _select(size, mcs, [n3(l) for l in LAMBDAS3], LAMBDAS3)
    lam4, areas4, _ = _select(size, mcs, [n4(l) for l in LAMBDAS4], LAMBDAS4)
    i10 = int(np.argmin(np.abs(np.log(DENSE) - np.log(0.1))))
    p_true = 1.0 - mats_c["base"][np.arange(len(yc)), yc]
    q10 = q[n4(lam4)][i10]
    restore_level = float(np.mean(p_true >= (1.0 - q10) / lam4))
    return {"scorer": sc, "q": q, "theta": theta, "lam3": lam3, "lam4": lam4, "areas3": areas3, "areas4": areas4,
            "sel_window": w, "restore_level_a01": restore_level, "q_sel_a01": float(q10)}


def run_multi(tag: str, dataset: str, resplits, out_root: Path | None = None,
              conds: list[tuple[str, str]] | None = None) -> list[Path]:
    t0 = time.time()
    root = out_root or get_paths().raw_results_root / "frontier_v4"
    todo = [int(r) for r in resplits if not (root / tag / f"r{int(r)}" / ".ok").exists()]
    if not todo:
        return [root / tag / f"r{int(r)}" for r in resplits]
    st = {r: _setup(tag, dataset, r) for r in todo}
    for r in todo:
        print(f"[{tag}] r{r}: lam3* = {st[r]['lam3']}, lam4* = {st[r]['lam4']} ({time.time() - t0:.0f}s)", flush=True)
    conds = conds if conds is not None else test_conditions(dataset)   # override: ImageNet-C-bar (v5b)
    K, A = len(conds), len(DENSE)
    all_names = ["base", "tta_avg", "tta_learned", "union_all", "ru", "aps_base", "tta_flip"] + \
        [n3(l) for l in LAMBDAS3] + [n4(l) for l in LAMBDAS4] + ["rbe3_sel", "rbe4_sel"]
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
            mats["rbe3_sel"] = mats[n3(s["lam3"])]
            mats["rbe4_sel"] = mats[n4(s["lam4"])]
            qs = dict(s["q"])
            qs["rbe3_sel"] = s["q"][n3(s["lam3"])]
            qs["rbe4_sel"] = s["q"][n4(s["lam4"])]
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
        meta = {"tag": tag, "dataset": dataset, "resplit": r, "lambdas3": list(LAMBDAS3), "lambdas4": list(LAMBDAS4),
                "lam3_star": s["lam3"], "lam4_star": s["lam4"], "selection_areas3": s["areas3"],
                "selection_areas4": s["areas4"], "selection_window": s["sel_window"], "theta": s["theta"].tolist(),
                "restore_level_a01": s["restore_level_a01"], "q_sel_a01": s["q_sel_a01"],
                "clean_cov_a01": a["clean_cov"], "conditions": [c for c, _ in conds], "seconds": time.time() - t0}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        (out / ".ok").write_text("ok")
        outs.append(out)
        print(f"done {out} ({meta['seconds']:.0f}s)", flush=True)
    return outs


NAT_METHODS = ["base", "tta_avg", "tta_flip", "ru", "rbe3_sel", "rbe4_sel"]


def run_naturals_v4(tag: str, resplits, out_root: Path | None = None) -> list[Path]:
    """Natural shifts with the lam3* / lam4* of the main v4 run; thresholds on the clean calibration split restricted
    to each natural set's classes (as K5 of v3)."""
    from ..data.imagenet import natural_index

    root = out_root or get_paths().raw_results_root / "frontier_v4"
    outs = []
    for r in resplits:
        out = root / tag / f"r{int(r)}"
        if (out / ".ok_naturals").exists() or not (out / ".ok").exists():
            continue
        meta = json.loads((out / "meta.json").read_text())
        lam3, lam4 = float(meta["lam3_star"]), float(meta["lam4_star"])
        pd = PoolData(tag, "imagenet", int(r), with_views=True)
        sc = ScorerV4("imagenet", np.asarray(meta["theta"], dtype=np.float64))
        for nat in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
            subset = natural_index(nat)["class_subset"]
            ctx = pd.ctx(P0, families(), 0.1, "lac", None, subset=subset, views=True)
            yc = ctx.clean_cal.labels
            mc = sc.matrices(ctx.clean_cal_views, np.random.default_rng(0).random(len(yc)), lams3=(lam3,), lams4=(lam4,))
            mc["rbe3_sel"], mc["rbe4_sel"] = mc[n3(lam3)], mc[n4(lam4)]
            q = {m: np.array([conformal_quantile(mc[m][np.arange(len(yc)), yc], a) for a in DENSE]) for m in NAT_METHODS}
            b = load_bundle(tag, nat, 0, cache=False).restrict(subset)
            mt = sc.matrices(b.all_view_logits(), np.random.default_rng(1).random(len(b.labels)), lams3=(lam3,),
                             lams4=(lam4,))
            mt["rbe3_sel"], mt["rbe4_sel"] = mt[n3(lam3)], mt[n4(lam4)]
            wts = boot_weights(len(b.labels), f"imagenet-{nat}")
            curves = {}
            for m in NAT_METHODS:
                cp, sz, cb, sb = curves_boot(mt[m], b.labels, q[m], wts)
                curves[m] = (np.maximum(0.0, TARGET - cp), sz, np.maximum(0.0, TARGET - cb), sb)
            np.savez_compressed(out / f"naturals_{nat}.npz", **{f"pm_{m}": c[0] for m, c in curves.items()},
                                **{f"ps_{m}": c[1] for m, c in curves.items()},
                                **{f"bm_{m}": c[2].astype(np.float32) for m, c in curves.items()},
                                **{f"bs_{m}": c[3].astype(np.float32) for m, c in curves.items()})
        (out / ".ok_naturals").write_text("ok")
        outs.append(out)
        print(f"done naturals {out}", flush=True)
    return outs


ES_METHODS = ["base", "tta_avg", "tta_learned", "union_all", "ru", "aps_base", "tta_flip", "rbe3_sel", "rbe4_sel"]


def run_es_v4(tag: str, resplits, conds: list[tuple[str, str]], subset: list[int], source_ids: np.ndarray,
              out_root: Path | None = None) -> list[Path]:
    """Pre-registration v5d (audits/gates_v5d.yaml): ImageNet-ES es-test, a real camera acquisition shift.

    Frozen lam3*, lam4* and TTA-CP weights of the main v4 run (nothing re-selected); thresholds on the clean calibration
    split restricted to the 200 classes MINUS the es-test source images (``source_ids``, ILSVRC2012 val ids); MCS
    averaged over ``conds``. Bootstrap: one multinomial weight per source image, shared by every condition and method.
    Writes curves.npz (aggregate curves + bootstrap, and per-condition point curves) and meta.json per re-split."""
    from ..data.splits import imagenet_val_split  # noqa: F401  (import check: same split code as the main runs)

    root = out_root or get_paths().raw_results_root / "frontier_v4_es"
    main = get_paths().raw_results_root / "frontier_v4"
    source_ids = np.sort(np.asarray(source_ids, dtype=np.int64))
    W = boot_weights(len(source_ids), "imagenet-es")
    K, A = len(conds), len(DENSE)
    tests = []
    for cn, src in conds:
        b = load_bundle(tag, src, 0, cache=False).restrict(subset)
        pos = np.searchsorted(source_ids, b.ids)
        assert np.all(source_ids[np.minimum(pos, len(source_ids) - 1)] == b.ids), f"{src}: id outside es-test sources"
        tests.append((cn, b, b.all_view_logits(), pos))
    outs = []
    for r in resplits:
        out = root / tag / f"r{int(r)}"
        if (out / ".ok").exists():
            outs.append(out)
            continue
        t0 = time.time()
        meta = json.loads((main / tag / f"r{int(r)}" / "meta.json").read_text())
        lam3, lam4 = float(meta["lam3_star"]), float(meta["lam4_star"])
        sc = ScorerV4("imagenet", np.asarray(meta["theta"], dtype=np.float64))
        pd = PoolData(tag, "imagenet", int(r), with_views=True)
        ctx = pd.ctx(P0, families(), 0.1, "lac", None, subset=subset, views=True)
        keep = ~np.isin(ctx.clean_cal.ids, source_ids)
        yc = ctx.clean_cal.labels[keep]
        mc = sc.matrices(ctx.clean_cal_views[keep], np.random.default_rng(0).random(len(yc)), lams3=(lam3,),
                         lams4=(lam4,))
        mc["rbe3_sel"], mc["rbe4_sel"] = mc[n3(lam3)], mc[n4(lam4)]
        q = {m: np.array([conformal_quantile(mc[m][np.arange(len(yc)), yc], a) for a in DENSE]) for m in ES_METHODS}
        pm = {m: np.zeros(A) for m in ES_METHODS}
        ps = {m: np.zeros(A) for m in ES_METHODS}
        bm = {m: np.zeros((A, W.shape[1])) for m in ES_METHODS}
        bs = {m: np.zeros((A, W.shape[1])) for m in ES_METHODS}
        cov_c = {m: np.zeros((K, A), dtype=np.float32) for m in ES_METHODS}
        size_c = {m: np.zeros((K, A), dtype=np.float32) for m in ES_METHODS}
        cov_np = {m: np.zeros((K, A), dtype=np.float32) for m in ES_METHODS}     # images in no split of our pool
        size_np = {m: np.zeros((K, A), dtype=np.float32) for m in ES_METHODS}
        top1, n_img = {}, {}
        for ci, (cn, b, vw, pos) in enumerate(tests):
            y = b.labels
            notpool = ~np.isin(b.ids, pd.ids)
            mt = sc.matrices(vw, np.random.default_rng(1).random(len(y)), lams3=(lam3,), lams4=(lam4,))
            mt["rbe3_sel"], mt["rbe4_sel"] = mt[n3(lam3)], mt[n4(lam4)]
            for m in ES_METHODS:
                cp, sz, cb, sb = curves_boot(mt[m], y, q[m], W[pos])
                pm[m] += np.maximum(0.0, TARGET - cp) / K
                ps[m] += sz / K
                bm[m] += np.maximum(0.0, TARGET - cb) / K
                bs[m] += sb / K
                cov_c[m][ci], size_c[m][ci] = cp, sz
                cp2, sz2, _, _ = curves_boot(mt[m][notpool], y[notpool], q[m])
                cov_np[m][ci], size_np[m][ci] = cp2, sz2
            top1[cn] = {m: float((mt[m].argmin(1) == y).mean()) for m in ("base", "tta_avg", "rbe3_sel")}
            n_img[cn] = int(len(y))
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "curves.npz", alphas=DENSE,
                            **{f"pm_{m}": v for m, v in pm.items()}, **{f"ps_{m}": v for m, v in ps.items()},
                            **{f"bm_{m}": v.astype(np.float32) for m, v in bm.items()},
                            **{f"bs_{m}": v.astype(np.float32) for m, v in bs.items()},
                            **{f"cov_{m}": v for m, v in cov_c.items()}, **{f"size_{m}": v for m, v in size_c.items()},
                            **{f"covnp_{m}": v for m, v in cov_np.items()}, **{f"sizenp_{m}": v for m, v in size_np.items()})
        info = {"tag": tag, "resplit": int(r), "lam3_star": lam3, "lam4_star": lam4, "conditions": [c for c, _ in conds],
                "n_images": n_img, "n_cal": int(keep.sum()), "n_cal_removed_es_sources": int((~keep).sum()),
                "n_sources_in_pool": int(np.isin(source_ids, pd.ids).sum()),
                "top1": top1, "seconds": time.time() - t0}
        (out / "meta.json").write_text(json.dumps(info, indent=1))
        (out / ".ok").write_text("ok")
        outs.append(out)
        print(f"done es {out} ({info['seconds']:.0f}s; removed {info['n_cal_removed_es_sources']} cal images)", flush=True)
    return outs
