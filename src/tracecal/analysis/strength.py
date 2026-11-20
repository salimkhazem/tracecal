"""SECONDARY (post hoc, not pre-registered): sensitivity of RU-CP to the restorer magnitudes.

The six restorers were fixed a priori (protocol.yaml). We re-extract the test conditions with two other banks
(configs/core/banks_secondary.yaml): "weak" (every magnitude halved) and "strong" (the pre-defined M = 2 magnitudes),
and compare on the dense alpha' grid with the frontier window / area / image bootstrap:
  ru         identity + 6 default restorers (RU-CP)      ru_weak    identity + 6 weak restorers
  ru_strong  identity + 6 strong restorers               ru_all     identity + all 18 restorers
  base       split CP
Thresholds: base-view LAC quantiles on clean calibration (default-bank base logits for every variant).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import view_names
from ..conformal.core import conformal_quantile
from ..paths import get_paths
from .bundles import load_bundle
from .dense_check import DENSE, curves_boot, window
from .frontier import RESTORERS, TARGET, _softmax3, boot_weights
from .gates_v2_eval import pair_delta
from .run import PoolData, test_conditions

METHODS = ["base", "ru", "ru_weak", "ru_strong", "ru_all"]
PAIRS = [("ru_weak", "ru"), ("ru_strong", "ru"), ("ru_all", "ru"), ("ru_weak", "base"), ("ru_strong", "base"),
         ("ru", "base")]


def run(tag: str, dataset: str, resplit: int, out_root: Path | None = None) -> Path:
    t0 = time.time()
    out = (out_root or get_paths().raw_results_root / "strength") / tag / f"r{resplit}"
    if (out / ".ok").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    vn = ["base"] + view_names(fam)
    ru_idx = [vn.index(v) for v in RESTORERS]
    n_r = len(RESTORERS)
    pd = PoolData(tag, dataset, resplit, with_views=False)
    cal = pd.versions[0].take(pd.pos["cal"])
    t_base = 1.0 - cal.probs[np.arange(len(cal)), cal.labels]
    q = np.array([conformal_quantile(t_base, a) for a in DENSE])
    conds = test_conditions(dataset)
    K, A, M = len(conds), len(DENSE), len(METHODS)
    pm, ps = np.zeros((M, A)), np.zeros((M, A))
    bm = bs = None
    w = None
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        bw = load_bundle(tag, src, 0, bank="weak", cache=False)
        bsg = load_bundle(tag, src, 0, bank="strong", cache=False)
        if not (np.array_equal(b.ids, bw.ids) and np.array_equal(b.ids, bsg.ids)):
            raise RuntimeError(f"row mismatch between banks for {tag} {src}")
        V = b.all_view_logits()
        P0 = _softmax3(V[:, 0])
        Pd = _softmax3(V[:, ru_idx])
        Pw = _softmax3(bw.all_view_logits()[:, 1:1 + n_r])
        Ps = _softmax3(bsg.all_view_logits()[:, 1:1 + n_r])
        y = b.labels
        n = len(y)
        if w is None:
            w = boot_weights(n, dataset)
            bm, bs = np.zeros((M, A, w.shape[1])), np.zeros((M, A, w.shape[1]))
        base = 1.0 - P0
        sd, sw, ss = (1.0 - Pd).min(1), (1.0 - Pw).min(1), (1.0 - Ps).min(1)
        scores = {"base": base, "ru": np.minimum(base, sd), "ru_weak": np.minimum(base, sw),
                  "ru_strong": np.minimum(base, ss), "ru_all": np.minimum(np.minimum(base, sd), np.minimum(sw, ss))}
        for mi, m in enumerate(METHODS):
            cp, sz, cb, sb = curves_boot(scores[m], y, q, w)
            pm[mi] += np.maximum(0.0, TARGET - cp) / K
            ps[mi] += sz / K
            bm[mi] += np.maximum(0.0, TARGET - cb) / K
            bs[mi] += sb / K
    mi = {m: i for i, m in enumerate(METHODS)}
    i10 = int(np.argmin(np.abs(np.log(DENSE) - np.log(0.1))))
    w0 = window(ps[mi["base"]])
    res: dict[str, Any] = {"window": w0, "point": {}, "boot": {},
                           "op_a01": {m: {"mcs": float(pm[mi[m], i10]), "size": float(ps[mi[m], i10])} for m in METHODS}}
    for a_, b_ in PAIRS:
        key = f"{a_}_vs_{b_}"
        res["point"][key] = pair_delta(ps, pm, mi[a_], mi[b_], w0)
        res["boot"][key] = [pair_delta(bs[..., k], bm[..., k], mi[a_], mi[b_], window(bs[mi["base"], :, k]))
                            for k in range(bm.shape[-1])]
    res["seconds"] = time.time() - t0
    np.savez_compressed(out / "curves.npz", alphas=DENSE, mcs=pm, size=ps)
    (out / "deltas.json").write_text(json.dumps(res))
    (out / ".ok").write_text("ok")
    print(f"done {out} ({res['seconds']:.0f}s)", flush=True)
    return out
