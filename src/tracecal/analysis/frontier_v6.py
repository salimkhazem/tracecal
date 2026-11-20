"""Pre-registration v6 (audits/gates_v6.yaml): RBE-CP with a camera-aware bank.

Score (designed on the v2 development models and ImageNet-ES es-val only, scripts/explore_v6.py):
    psym(x) = softmax((z(x) + z(flip x)) / 2)                                  anchor: identity + horizontal flip
    rmax(x) = max over the 11 restorers {6 default (frozen bank)} U {5 camera: auto_gamma, stretch, equalize,
              grayworld, median3} of p_y(Tx)
    v6_lam  = 1 - [psym_y + lam * (rmax_y - psym_y)_+],   lam in {0, .25, .5, .75, 1}
lam is selected per (model, re-split) with the v3 rule on the tune split's synthetic pool versions (full label space);
the score is calibrated on clean data. 13 forward passes (identity, flip, 11 restorers).

Computed in the same pass, with the same bootstrap weights: split CP, test-time averaging over the 13 default views,
TTA-CP (frozen weights of the main v4 run), the 13-view union, RU-CP, APS and RBE-CP v3 (frozen lam3* of the main run).
Benchmarks: ImageNet-C (96 conditions; bootstrap over the 10k test images) and ImageNet-ES es-test (65 conditions;
200 classes; calibration without the es-test source images; bootstrap over the 1,000 source images) as in v5d.
Outputs: results/raw/frontier_v6/<tag>/r<k>/{curves_inc.npz, curves_es.npz, meta.json}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import families, probe_names, view_names
from ..conformal.core import aps_all, conformal_quantile
from ..paths import get_paths
from .bundles import load_bundle, softmax
from .dense_check import DENSE, curves_boot, window
from .frontier import P0, RESTORERS, TARGET, _softmax3, boot_weights
from .frontier_v3 import _area
from .run import PoolData, test_conditions

LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ["base", "tta_avg", "tta_learned", "union_all", "ru", "aps_base", "rbe3_sel", "v6_sel"]


class ScorerV6:
    def __init__(self, theta: np.ndarray, lam3: float):
        vn = ["base"] + view_names("imagenet")
        self.n_def = len(vn)
        self.r_def = [vn.index(r) for r in RESTORERS]
        self.flip = vn.index("hflip")
        self.n_cam = len(probe_names("imagenet", "camera"))
        self.theta, self.lam3 = theta, lam3

    def join(self, z_def: np.ndarray, z_cam: np.ndarray) -> np.ndarray:
        return np.concatenate([z_def, z_cam[:, 1:1 + self.n_cam]], axis=1)      # (n, 13 + 5, C)

    def v6(self, z: np.ndarray, P: np.ndarray, lam: float) -> np.ndarray:
        psym = softmax(z[:, [0, self.flip]].mean(1))
        if lam == 0:
            return 1.0 - psym
        rmax = P[:, self.r_def + list(range(self.n_def, self.n_def + self.n_cam))].max(1)
        return 1.0 - (psym + lam * np.maximum(rmax - psym, 0.0))

    def matrices(self, z: np.ndarray, u: np.ndarray, lam6: float) -> dict[str, np.ndarray]:
        P = _softmax3(z)
        zd, Pd = z[:, : self.n_def], P[:, : self.n_def]
        pbar = softmax(zd.mean(1))
        rmax3 = Pd[:, self.r_def].max(1)
        return {"base": 1.0 - Pd[:, 0], "tta_avg": 1.0 - pbar,
                "tta_learned": 1.0 - softmax(np.einsum("nmc,m->nc", zd, self.theta)),
                "union_all": 1.0 - Pd.max(1), "ru": 1.0 - Pd[:, [0] + self.r_def].max(1),
                "aps_base": aps_all(Pd[:, 0], u),
                "rbe3_sel": 1.0 - (pbar + self.lam3 * np.maximum(rmax3 - pbar, 0.0)),
                "v6_sel": self.v6(z, P, lam6)}


def _bundles(tag: str, src: str, version: int = 0, rows=None, subset=None):
    d = load_bundle(tag, src, version, cache=False)
    c = load_bundle(tag, src, version, bank="camera", cache=False)
    if not np.array_equal(d.ids, c.ids):
        raise RuntimeError(f"{tag} {src} v{version}: default and camera bundles are not aligned")
    if rows is not None:
        d, c = d.take(rows), c.take(rows)
    if subset is not None:
        d, c = d.restrict(subset), c.restrict(subset)
    return d, c


def select_lambda(tag: str, pd: PoolData, sc: ScorerV6) -> tuple[float, dict]:
    """The v3 rule: argmin efficiency-matched area on the tune split's synthetic pool versions (ties -> smallest)."""
    d, c = _bundles(tag, "imagenet:pool", 0, rows=pd.pos["cal"])
    z = sc.join(d.all_view_logits(), c.all_view_logits())
    P, y = _softmax3(z), d.labels
    q = {l: np.array([conformal_quantile(sc.v6(z, P, l)[np.arange(len(y)), y], a) for a in DENSE]) for l in LAMS}
    qb = np.array([conformal_quantile(1.0 - P[:, 0][np.arange(len(y)), y], a) for a in DENSE])
    K = len(pd.versions)
    mcs = {l: np.zeros(len(DENSE)) for l in LAMS}
    size = {l: np.zeros(len(DENSE)) for l in LAMS}
    sb = np.zeros(len(DENSE))
    for v in range(K):
        d, c = _bundles(tag, "imagenet:pool", v, rows=pd.pos["tune"])
        z = sc.join(d.all_view_logits(), c.all_view_logits())
        P, y = _softmax3(z), d.labels
        for l in LAMS:
            cp, sz, _, _ = curves_boot(sc.v6(z, P, l), y, q[l])
            mcs[l] += np.maximum(0.0, TARGET - cp) / K
            size[l] += sz / K
        _, sz0, _, _ = curves_boot(1.0 - P[:, 0], y, qb)
        sb += sz0 / K
    w = window(sb)
    areas = {l: _area(size[l], mcs[l], w) for l in LAMS}
    fin = {l: a for l, a in areas.items() if np.isfinite(a)}
    return min(LAMS, key=lambda l: fin.get(l, np.inf)), {str(l): float(a) for l, a in areas.items()}


class _Acc:
    """Streaming accumulator of point / bootstrap curves over test conditions for one (re-split, benchmark)."""

    def __init__(self, sc: ScorerV6, lam6: float, cal: tuple, n_cond: int, nb: int):
        zc, yc = cal
        mc = sc.matrices(zc, np.random.default_rng(0).random(len(yc)), lam6)
        self.q = {m: np.array([conformal_quantile(mc[m][np.arange(len(yc)), yc], a) for a in DENSE]) for m in METHODS}
        self.sc, self.lam6, self.K = sc, lam6, n_cond
        A = len(DENSE)
        self.acc = {f"{k}_{m}": (np.zeros(A) if k in ("pm", "ps") else np.zeros((A, nb)))
                    for m in METHODS for k in ("pm", "ps", "bm", "bs")}
        self.cov = {m: np.zeros((n_cond, A), dtype=np.float32) for m in METHODS}
        self.siz = {m: np.zeros((n_cond, A), dtype=np.float32) for m in METHODS}

    def add(self, ci: int, z: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        mt = self.sc.matrices(z, np.random.default_rng(1).random(len(y)), self.lam6)
        for m in METHODS:
            cp, sz, cb, sb = curves_boot(mt[m], y, self.q[m], w)
            self.acc[f"pm_{m}"] += np.maximum(0.0, TARGET - cp) / self.K
            self.acc[f"ps_{m}"] += sz / self.K
            self.acc[f"bm_{m}"] += np.maximum(0.0, TARGET - cb) / self.K
            self.acc[f"bs_{m}"] += sb / self.K
            self.cov[m][ci], self.siz[m][ci] = cp, sz

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: (v.astype(np.float32) if k[:2] in ("bm", "bs") else v) for k, v in self.acc.items()}
        out.update({f"cov_{m}": v for m, v in self.cov.items()})
        out.update({f"size_{m}": v for m, v in self.siz.items()})
        return out


def run_v6(tag: str, resplits, es_conds: list[tuple[str, str]], es_subset, es_source_ids, out_root: Path | None = None,
           inc_conds: list[tuple[str, str]] | None = None, main_root: Path | None = None) -> list[Path]:
    """main_root: where the frozen v4 meta (TTA-CP weights, lam3*) is read; only overridden by the smoke test."""
    root = out_root or get_paths().raw_results_root / "frontier_v6"
    main = main_root or get_paths().raw_results_root / "frontier_v4"
    todo = [int(r) for r in resplits if not (root / tag / f"r{int(r)}" / ".ok").exists()]
    if not todo:
        return [root / tag / f"r{int(r)}" for r in resplits]
    t0 = time.time()
    inc_conds = inc_conds if inc_conds is not None else test_conditions("imagenet")
    es_ids = np.sort(np.asarray(es_source_ids, dtype=np.int64))
    W_es = boot_weights(len(es_ids), "imagenet-es")
    W_inc = None
    st = {}
    for r in todo:   # per re-split: frozen v4 meta, lambda6 selection, both calibration sets
        meta4 = json.loads((main / tag / f"r{r}" / "meta.json").read_text())
        sc = ScorerV6(np.asarray(meta4["theta"], dtype=np.float64), float(meta4["lam3_star"]))
        pd = PoolData(tag, "imagenet", r, with_views=True)
        lam6, sel_areas = select_lambda(tag, pd, sc)
        d, c = _bundles(tag, "imagenet:pool", 0, rows=pd.pos["cal"])
        cal_inc = (sc.join(d.all_view_logits(), c.all_view_logits()), d.labels)
        keep = np.flatnonzero(~np.isin(d.ids, es_ids))
        de, ce = d.take(keep).restrict(es_subset), c.take(keep).restrict(es_subset)
        cal_es = (sc.join(de.all_view_logits(), ce.all_view_logits()), de.labels)
        st[r] = {"sc": sc, "lam6": lam6, "areas": sel_areas, "cal_inc": cal_inc, "cal_es": cal_es,
                 "n_removed": int(len(d.ids) - len(keep))}
        print(f"[{tag}] r{r}: lam6* = {lam6}, lam3* = {sc.lam3} ({time.time() - t0:.0f}s)", flush=True)
    tmp = st[todo[0]]["sc"]
    accs_inc, accs_es = {}, {r: _Acc(st[r]["sc"], st[r]["lam6"], st[r]["cal_es"], len(es_conds), W_es.shape[1])
                             for r in todo}
    for ci, (_, src) in enumerate(inc_conds):   # one ImageNet-C condition in memory at a time
        d, c = _bundles(tag, src)
        z, y = tmp.join(d.all_view_logits(), c.all_view_logits()), d.labels
        if W_inc is None:
            W_inc = boot_weights(len(y), "imagenet")
            accs_inc = {r: _Acc(st[r]["sc"], st[r]["lam6"], st[r]["cal_inc"], len(inc_conds), W_inc.shape[1])
                        for r in todo}
        for r in todo:
            accs_inc[r].add(ci, z, y, W_inc)
        if ci % 16 == 0:
            print(f"[{tag}] ImageNet-C {ci + 1}/{len(inc_conds)} ({time.time() - t0:.0f}s)", flush=True)
    for ci, (_, src) in enumerate(es_conds):
        d, c = _bundles(tag, src, subset=es_subset)
        z, y, pos = tmp.join(d.all_view_logits(), c.all_view_logits()), d.labels, np.searchsorted(es_ids, d.ids)
        for r in todo:
            accs_es[r].add(ci, z, y, W_es[pos])
    outs = []
    for r in todo:
        out = root / tag / f"r{r}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "curves_inc.npz", alphas=DENSE, **accs_inc[r].arrays())
        np.savez_compressed(out / "curves_es.npz", alphas=DENSE, **accs_es[r].arrays())
        meta = {"tag": tag, "resplit": r, "lam6_star": st[r]["lam6"], "lam3_star": st[r]["sc"].lam3,
                "selection_areas6": st[r]["areas"], "conditions_inc": [cn for cn, _ in inc_conds],
                "conditions_es": [cn for cn, _ in es_conds], "n_cal_es": int(len(st[r]["cal_es"][1])),
                "n_cal_es_removed": st[r]["n_removed"], "seconds": time.time() - t0}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        (out / ".ok").write_text("ok")
        outs.append(out)
        print(f"done v6 {out} lam6*={st[r]['lam6']}", flush=True)
    return outs


NAT_METHODS6 = ["base", "tta_avg", "rbe3_sel", "rbe4_sel", "v6_sel"]


def run_naturals_v6(tag: str, resplits) -> list[Path]:
    """REPORT ONLY (not pre-registered, added 2026-11-06): the frozen v6 score on ImageNet-V2, -Sketch, -R and -A, with
    lam6* of the v6 run and lam3*, lam4*, theta of the main v4 run; thresholds on the clean calibration split restricted
    to each set's classes (as the pre-registered natural-shift analyses). rbe4_sel = the v4 identity+flip variant."""
    from ..data.imagenet import natural_index

    root = get_paths().raw_results_root / "frontier_v6"
    main = get_paths().raw_results_root / "frontier_v4"
    outs = []
    for r in resplits:
        out = root / tag / f"r{int(r)}"
        if (out / ".ok_naturals").exists() or not (out / ".ok").exists():
            continue
        m6 = json.loads((out / "meta.json").read_text())
        m4 = json.loads((main / tag / f"r{int(r)}" / "meta.json").read_text())
        sc = ScorerV6(np.asarray(m4["theta"], dtype=np.float64), float(m4["lam3_star"]))
        lam4, lam6 = float(m4["lam4_star"]), float(m6["lam6_star"])
        pd = PoolData(tag, "imagenet", int(r), with_views=True)

        def mats(z: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
            mt = sc.matrices(z, u, lam6)
            P = _softmax3(z[:, : sc.n_def])
            psym = softmax(z[:, [0, sc.flip]].mean(1))
            mt["rbe4_sel"] = 1.0 - (psym + lam4 * np.maximum(P[:, sc.r_def].max(1) - psym, 0.0))
            return mt

        for nat in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
            subset = natural_index(nat)["class_subset"]
            d, c = _bundles(tag, "imagenet:pool", 0, rows=pd.pos["cal"], subset=subset)
            zc, yc = sc.join(d.all_view_logits(), c.all_view_logits()), d.labels
            mc = mats(zc, np.random.default_rng(0).random(len(yc)))
            q = {m: np.array([conformal_quantile(mc[m][np.arange(len(yc)), yc], a) for a in DENSE]) for m in NAT_METHODS6}
            d, c = _bundles(tag, nat, subset=subset)
            z, y = sc.join(d.all_view_logits(), c.all_view_logits()), d.labels
            mt = mats(z, np.random.default_rng(1).random(len(y)))
            w = boot_weights(len(y), f"imagenet-{nat}")
            arr = {}
            for m in NAT_METHODS6:
                cp, sz, cb, sb = curves_boot(mt[m], y, q[m], w)
                arr.update({f"pm_{m}": np.maximum(0.0, TARGET - cp), f"ps_{m}": sz,
                            f"bm_{m}": np.maximum(0.0, TARGET - cb).astype(np.float32), f"bs_{m}": sb.astype(np.float32)})
            np.savez_compressed(out / f"naturals_{nat}.npz", **arr)
        (out / ".ok_naturals").write_text("ok")
        outs.append(out)
        print(f"done v6 naturals {out}", flush=True)
    return outs
