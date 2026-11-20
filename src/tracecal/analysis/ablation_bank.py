"""SECONDARY (not pre-registered) bank ablations for RU-CP, on the confirmatory runs.

Variants (all: LAC, threshold from the base view on clean calibration, union over the listed views):
  ru            identity + 6 restorers (the frozen method)
  ru_minus_<r>  drop one restorer r
  ru_plus_deg   add the 5 degrading probes (contrast_down, sat_down, noise, jpeg, downup)
  ru_plus_flip  add the horizontal flip
  restorers_only_union_score  (diagnostic) union-score restorability gap vs best single restorer (bound tightness)
Output: results/raw/ablation_bank/<tag>/r<k>/{curves.npz, meta.json}; Delta AUC vs ru computed by the reporter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import families, view_names
from ..conformal.core import conformal_quantile
from ..paths import get_paths
from .bundles import load_bundle
from .frontier import ALPHAS, RESTORERS, TARGET, _softmax3, cov_size, ks_plus
from .run import PoolData, test_conditions

DEGRADERS = ["contrast_down", "sat_down", "noise", "jpeg", "downup"]


def variants(family: str) -> dict[str, list[int]]:
    vn = ["base"] + view_names(family)
    ru = [0] + [vn.index(v) for v in RESTORERS]
    out = {"ru": ru}
    for r in RESTORERS:
        out[f"ru_minus_{r}"] = [i for i in ru if i != vn.index(r)]
    out["ru_plus_deg"] = ru + [vn.index(v) for v in DEGRADERS]
    out["ru_plus_flip"] = ru + [vn.index("hflip")]
    out["base_only"] = [0]
    return out


def run(tag: str, dataset: str, resplit: int, out_root: Path | None = None) -> Path:
    t0 = time.time()
    out = (out_root or get_paths().raw_results_root / "ablation_bank") / tag / f"r{resplit}"
    if (out / ".ok").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    pd = PoolData(tag, dataset, resplit, with_views=True)
    clean_cal = pd.versions[0].take(pd.pos["cal"])
    vw = clean_cal.all_view_logits()
    y = clean_cal.labels
    t = 1.0 - _softmax3(vw)[np.arange(len(y)), 0, y]
    q = np.array([conformal_quantile(t, a) for a in ALPHAS])
    V = variants(fam)
    conds = test_conditions(dataset)
    cov = np.zeros((len(V), len(ALPHAS), len(conds)), np.float32)
    size = np.zeros_like(cov)
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        P = _softmax3(b.all_view_logits())
        lac = 1.0 - P
        for vi, (name, idx) in enumerate(V.items()):
            c, s = cov_size(lac[:, idx].min(1), b.labels, q)
            cov[vi, :, ci] = c.mean(0)
            size[vi, :, ci] = s.mean(0)
    # bound tightness on the fit split: union-score gap vs best single-member gap, per family
    fit = pd.pos["fit"]
    cf = pd.versions[0].take(fit)
    s_clean = 1.0 - _softmax3(cf.all_view_logits())[np.arange(len(fit)), 0, cf.labels]
    tight = {}
    for f in families():
        v = families().index(f) + 1
        bf = pd.versions[v].take(fit)
        Pf = _softmax3(bf.all_view_logits())
        yf = bf.labels
        singles = [ks_plus(s_clean, 1.0 - Pf[np.arange(len(yf)), k, yf]) for k in V["ru"]]
        union = ks_plus(s_clean, (1.0 - Pf[:, V["ru"]]).min(1)[np.arange(len(yf)), yf])
        tight[f] = {"min_single_gap": min(singles), "union_gap": union}
    np.savez_compressed(out / "curves.npz", cov=cov, size=size, alphas=ALPHAS)
    meta = {"tag": tag, "resplit": resplit, "variants": list(V), "conditions": [c for c, _ in conds],
            "bound_tightness": tight, "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / ".ok").write_text("ok")
    return out


def delta_auc(tag: str, resplit: int) -> dict[str, float]:
    """Point Delta AUC (variant - ru) over the pre-registered window (from the base_only curve)."""
    from .gates_v2_eval import auc, window

    d = get_paths().raw_results_root / "ablation_bank" / tag / f"r{resplit}"
    meta = json.loads((d / "meta.json").read_text())
    z = np.load(d / "curves.npz")
    mcs = np.maximum(0.0, TARGET - z["cov"]).mean(2)
    msz = z["size"].mean(2)
    names = meta["variants"]
    w = window(msz[names.index("base_only")])
    iru = names.index("ru")
    out = {}
    for i, n in enumerate(names):
        if n == "ru":
            continue
        lo = max(w[0], msz[iru].min(), msz[i].min())
        hi = min(w[1], msz[iru].max(), msz[i].max())
        out[n] = float("nan") if hi - lo < 0.1 * (w[1] - w[0]) else \
            auc(msz[i], mcs[i], lo, hi) - auc(msz[iru], mcs[iru], lo, hi)
    return out
