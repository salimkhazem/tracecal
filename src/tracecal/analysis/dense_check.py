"""ROBUSTNESS CHECK (not pre-registered): the efficiency-matched statistic on a dense alpha' grid.

The pre-registered statistic samples each method's coverage/size curve at 25 alpha' values and interpolates linearly
between them. Two methods that threshold the SAME score (e.g. split CP and synthetic-shift calibration, both on the
base LAC score) trace the same curve, so any difference between them is interpolation error of the sparse grid. Here we
recompute the curves on a dense grid (N_DENSE log-spaced alpha' in the same range) with the same window, area and
bootstrap definitions, for: the 7 frontier methods (all H1/H2 pairs and the secondary baseline pairs, with image
bootstrap draws shared with the frontier runs), the bank-ablation variants and the per-condition QTC / QTC + RU pair
(point estimates). Only scalar deltas and point curves are stored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import families
from ..conformal.core import conformal_quantile
from ..paths import get_paths
from .ablation_bank import variants
from .bundles import load_bundle
from .complementary import qtc_beta
from .bundles import softmax
from .frontier import METHODS, P0, TARGET, Scorer, _softmax3, boot_weights
from .gates_v2_eval import BASELINES, auc, pair_delta
from .run import PoolData, test_conditions

N_DENSE = 400
DENSE = np.geomspace(0.001, 0.30, N_DENSE)
SECONDARY = [("base_pprime", "base"), ("tta_avg", "base"), ("tta_learned", "base"), ("union_all", "base")]


def cov_size_fast(U: np.ndarray, y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same output as frontier.cov_size, in O(nC log A + nA): bin every score among the sorted thresholds, then
    cumulate per row (q may be in any order and contain +inf)."""
    n, c = U.shape
    A = len(q)
    order = np.argsort(q, kind="stable")
    k = np.searchsorted(q[order], U, side="left")                     # thresholds strictly below each score
    cnt = np.bincount((np.arange(n)[:, None] * (A + 1) + k).ravel(), minlength=n * (A + 1)).reshape(n, A + 1)
    srt = np.cumsum(cnt, axis=1)[:, :A]                                # size at q[order][a]
    size = np.empty((n, A), np.int32)
    size[:, order] = srt
    covered = U[np.arange(n), y][:, None] <= q[None, :]
    return covered, size


def curves_boot(U: np.ndarray, y: np.ndarray, q: np.ndarray, w: np.ndarray | None = None):
    """Mean coverage and mean set size at every threshold in q (point, shape (A,)) and, if w (n, B) bootstrap counts
    is given, their bootstrap versions (A, B). Uses that coverage and size are monotone in the threshold: bin each
    score among the sorted thresholds, aggregate bootstrap weights per bin (sparse), then cumulate."""
    import scipy.sparse as sp

    n, c = U.shape
    A = len(q)
    order = np.argsort(q, kind="stable")
    qs = q[order]
    k = np.searchsorted(qs, U, side="left").ravel()
    keep = k < A                                             # scores above every threshold never enter a set
    S = sp.csr_matrix((np.ones(int(keep.sum())), (np.repeat(np.arange(n), c)[keep], k[keep])), shape=(n, A + 1))
    kc = np.searchsorted(qs, U[np.arange(n), y], side="left")
    Cm = sp.csr_matrix((np.ones(n), (np.arange(n), kc)), shape=(n, A + 1))

    def unsort(x):
        o = np.empty_like(x)
        o[order] = x
        return o

    cov_pt = unsort(np.cumsum(np.asarray(Cm.sum(0)).ravel())[:A] / n)
    size_pt = unsort(np.cumsum(np.asarray(S.sum(0)).ravel())[:A] / n)
    if w is None:
        return cov_pt, size_pt, None, None
    wd = w.astype(np.float64)
    cov_b = unsort(np.cumsum(np.asarray(Cm.T @ wd), axis=0)[:A] / n)
    size_b = unsort(np.cumsum(np.asarray(S.T @ wd), axis=0)[:A] / n)
    return cov_pt, size_pt, cov_b, size_b


def window(base_size: np.ndarray, alphas: np.ndarray = DENSE) -> tuple[float, float]:
    la = np.log(alphas)
    return float(np.interp(np.log(0.10), la, base_size)), float(np.interp(np.log(0.01), la, base_size))


def _setup(tag: str, dataset: str, resplit: int) -> dict[str, Any]:
    fmeta = json.loads((get_paths().raw_results_root / "frontier" / tag / f"r{resplit}" / "meta.json").read_text())
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    scorer = Scorer(fam, np.asarray(fmeta["theta"], dtype=np.float64))
    pd = PoolData(tag, dataset, resplit, with_views=True)
    ctx = pd.ctx(P0, families(), 0.1, "lac", None, views=True)
    # thresholds (dense), exactly as frontier.thresholds but on DENSE
    yc = ctx.clean_cal.labels
    mats_c = scorer.matrices(ctx.clean_cal_views, np.random.default_rng(0).random(len(yc)))
    t_src = {m: mats_c[{"union_all": "base", "ru": "base"}.get(m, m)][np.arange(len(yc)), yc] for m in METHODS}
    yp = ctx.pcal.labels
    t_src["base_pprime"] = scorer.matrices(ctx.pcal_views, np.random.default_rng(0).random(len(yp)))["base"][
        np.arange(len(yp)), yp]
    q = {m: np.array([conformal_quantile(t_src[m], a) for a in DENSE]) for m in METHODS}
    # sanity: the dense thresholds at the 25 pre-registered alphas equal the stored frontier thresholds
    from .frontier import ALPHAS
    for m in METHODS:
        q25 = np.array([conformal_quantile(t_src[m], a) for a in ALPHAS])
        stored = np.asarray(fmeta["thresholds"][m], dtype=np.float64)
        assert np.allclose(q25, stored, rtol=1e-5, atol=1e-6, equal_nan=True), (m, q25[:3], stored[:3])
    V = variants(fam)
    names_pt = list(METHODS) + [v for v in V if v not in ("ru", "base_only")] + ["qtc", "qtc_ru"]
    return {"scorer": scorer, "q": q, "V": V, "t_base": t_src["base"],
            "src_conf": _softmax3(ctx.clean_cal_views[:, 0]).max(1), "names_pt": names_pt,
            "shortfall": {k: np.zeros(N_DENSE) for k in names_pt}, "size": {k: np.zeros(N_DENSE) for k in names_pt},
            "boot_mcs": None, "boot_size": None}


def _update(st: dict[str, Any], mats: dict[str, np.ndarray], lac: np.ndarray, tgt_conf: np.ndarray, y: np.ndarray,
            w: np.ndarray, K: int) -> None:
    q, V = st["q"], st["V"]
    shortfall, size = st["shortfall"], st["size"]
    for mi, m in enumerate(METHODS):
        cp, sz, bc, bs = curves_boot(mats[m], y, q[m], w)
        shortfall[m] += np.maximum(0.0, TARGET - cp) / K
        size[m] += sz / K
        st["boot_mcs"][mi] += np.maximum(0.0, TARGET - bc) / K
        st["boot_size"][mi] += bs / K
    for v, idx in V.items():
        if v in ("ru", "base_only"):
            continue
        cp, sz, _, _ = curves_boot(lac[:, idx].min(1), y, q["base"])
        shortfall[v] += np.maximum(0.0, TARGET - cp) / K
        size[v] += sz / K
    t_base, src_conf = st["t_base"], st["src_conf"]
    qq = np.array([conformal_quantile(t_base, be) if (be := qtc_beta(src_conf, tgt_conf, a)) > 0 else np.inf
                   for a in DENSE])
    for nm, U in (("qtc", lac[:, 0]), ("qtc_ru", lac[:, V["ru"]].min(1))):
        cp, sz, _, _ = curves_boot(U, y, qq)
        shortfall[nm] += np.maximum(0.0, TARGET - cp) / K
        size[nm] += sz / K


def _finalize(st: dict[str, Any], out: Path, t0: float) -> Path:
    shortfall, size, names_pt = st["shortfall"], st["size"], st["names_pt"]
    boot_mcs, boot_size = st["boot_mcs"], st["boot_size"]
    M = len(METHODS)
    mi = {m: i for i, m in enumerate(METHODS)}
    ps = np.stack([size[m] for m in METHODS])
    pm = np.stack([shortfall[m] for m in METHODS])
    w0 = window(ps[mi["base"]])
    res: dict[str, Any] = {"window": w0, "point": {}, "boot": {}}
    pairs = [("ru", b_) for b_ in BASELINES] + SECONDARY
    for a_, b_ in pairs:
        key = f"{a_}_vs_{b_}"
        res["point"][key] = pair_delta(ps, pm, mi[a_], mi[b_], w0)
        res["boot"][key] = [pair_delta(boot_size[..., k], boot_mcs[..., k], mi[a_], mi[b_],
                                       window(boot_size[mi["base"], :, k])) for k in range(boot_mcs.shape[-1])]
    # ablation variants and QTC pair: point deltas
    abl = {}
    for v in names_pt[M:]:
        if v.startswith("qtc"):
            continue
        sv, mv = np.stack([size["ru"], size[v]]), np.stack([shortfall["ru"], shortfall[v]])
        abl[v] = pair_delta(sv, mv, 1, 0, w0)        # variant minus RU (positive = variant worse)
    res["ablation_vs_ru"] = abl
    s1, m1, s2, m2 = size["qtc"], shortfall["qtc"], size["qtc_ru"], shortfall["qtc_ru"]
    lo, hi = max(s1.min(), s2.min()), min(s1.max(), s2.max())
    res["qtcru_vs_qtc"] = float("nan") if hi <= lo else auc(s2, m2, lo, hi) - auc(s1, m1, lo, hi)
    res["seconds"] = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curves.npz", alphas=DENSE, **{f"mcs_{k}": v for k, v in shortfall.items()},
                        **{f"size_{k}": v for k, v in size.items()})
    (out / "deltas.json").write_text(json.dumps(res))
    (out / ".ok").write_text("ok")
    print(f"done {out} ({res['seconds']:.0f}s)", flush=True)
    return out


def run_multi(tag: str, dataset: str, resplits, out_root: Path | None = None) -> list[Path]:
    """All pending re-splits of one model, reading each test condition once (identical outputs to `run`)."""
    t0 = time.time()
    root = out_root or get_paths().raw_results_root / "dense_check"
    todo = [int(r) for r in resplits if not (root / tag / f"r{int(r)}" / ".ok").exists()]
    if not todo:
        return [root / tag / f"r{int(r)}" for r in resplits]
    st = {r: _setup(tag, dataset, r) for r in todo}
    conds = test_conditions(dataset)
    K = len(conds)
    w = None
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        vw = b.all_view_logits()
        y = b.labels
        n = len(y)
        if w is None:
            w = boot_weights(n, dataset)
            for r in todo:
                st[r]["boot_mcs"] = np.zeros((len(METHODS), N_DENSE, w.shape[1]))
                st[r]["boot_size"] = np.zeros((len(METHODS), N_DENSE, w.shape[1]))
        mats0 = st[todo[0]]["scorer"].matrices(vw, np.random.default_rng(1).random(n))
        lac = 1.0 - _softmax3(vw)
        tgt_conf = _softmax3(vw[:, 0]).max(1)
        for r in todo:
            mats = dict(mats0)
            if r != todo[0]:
                mats["tta_learned"] = 1.0 - softmax(np.einsum("nmc,m->nc", vw, st[r]["scorer"].theta))
            _update(st[r], mats, lac, tgt_conf, y, w, K)
    return [_finalize(st[r], root / tag / f"r{r}", t0) for r in todo]


def run(tag: str, dataset: str, resplit: int, out_root: Path | None = None) -> Path:
    out = (out_root or get_paths().raw_results_root / "dense_check") / tag / f"r{resplit}"
    if (out / ".ok").exists():
        return out
    return run_multi(tag, dataset, [resplit], out_root)[0]
