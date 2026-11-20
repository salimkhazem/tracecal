"""SECONDARY (not pre-registered): is RU-CP's gain explained by never returning an empty set?

LAC sets {y : 1 - p_y(x) <= q} are often EMPTY under corruption (the top probability falls below 1 - q). A common
conservative fix always includes the top-1 label ("non-empty"). We compare, on the dense alpha' grid with the same
window / area / image bootstrap as the frontier evaluation:
  base     LAC on the input                       ru     RU-CP (union over identity + 6 restorers)
  base_ne  LAC + top-1 label of the input         ru_ne  RU-CP + top-1 label of the input
All thresholds: base-view LAC quantiles on clean calibration (a threshold family only moves along a curve, so the
choice does not change the curves).
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

METHODS = ["base", "base_ne", "ru", "ru_ne"]
PAIRS = [("ru", "base_ne"), ("ru_ne", "base_ne"), ("ru_ne", "ru"), ("base_ne", "base"), ("ru", "base")]


def run_multi(tag: str, dataset: str, resplits, out_root: Path | None = None) -> list[Path]:
    """All pending re-splits of one model, reading each test condition once (identical outputs to `run`)."""
    t0 = time.time()
    root = out_root or get_paths().raw_results_root / "nonempty"
    todo = [int(r) for r in resplits if not (root / tag / f"r{int(r)}" / ".ok").exists()]
    if not todo:
        return [root / tag / f"r{int(r)}" for r in resplits]
    fam = "cifar" if dataset.startswith("cifar") else "imagenet"
    vn = ["base"] + view_names(fam)
    ru_idx = [0] + [vn.index(v) for v in RESTORERS]
    conds = test_conditions(dataset)
    K, A, M = len(conds), len(DENSE), len(METHODS)
    i10 = int(np.argmin(np.abs(np.log(DENSE) - np.log(0.1))))
    st = {}
    for r in todo:
        pd = PoolData(tag, dataset, r, with_views=False)
        cal = pd.versions[0].take(pd.pos["cal"])
        t_base = 1.0 - cal.probs[np.arange(len(cal)), cal.labels]
        st[r] = {"q": np.array([conformal_quantile(t_base, a) for a in DENSE]), "pm": np.zeros((M, A)),
                 "ps": np.zeros((M, A)), "bm": None, "bs": None, "empty": []}
    w = None
    for ci, (cn, src) in enumerate(conds):
        b = load_bundle(tag, src, 0, cache=False)
        P = _softmax3(b.all_view_logits()[:, ru_idx])
        y = b.labels
        n = len(y)
        if w is None:
            w = boot_weights(n, dataset)
            for r in todo:
                st[r]["bm"], st[r]["bs"] = np.zeros((M, A, w.shape[1])), np.zeros((M, A, w.shape[1]))
        base = 1.0 - P[:, 0]
        ru = (1.0 - P).min(1)
        top = P[:, 0].argmax(1)
        base_ne, ru_ne = base.copy(), ru.copy()
        base_ne[np.arange(n), top] = -1.0          # always below every threshold (scores and thresholds are >= 0)
        ru_ne[np.arange(n), top] = -1.0
        for r in todo:
            s_ = st[r]
            s_["empty"].append(float((base.min(1) > s_["q"][i10]).mean()))
            for mi, U in enumerate((base, base_ne, ru, ru_ne)):
                cp, sz, cb, sb = curves_boot(U, y, s_["q"], w)
                s_["pm"][mi] += np.maximum(0.0, TARGET - cp) / K
                s_["ps"][mi] += sz / K
                s_["bm"][mi] += np.maximum(0.0, TARGET - cb) / K
                s_["bs"][mi] += sb / K
    outs = []
    mi = {m: i for i, m in enumerate(METHODS)}
    for r in todo:
        s_ = st[r]
        pm, ps, bm, bs = s_["pm"], s_["ps"], s_["bm"], s_["bs"]
        out = root / tag / f"r{r}"
        out.mkdir(parents=True, exist_ok=True)
        w0 = window(ps[mi["base"]])
        res: dict[str, Any] = {"window": w0, "point": {}, "boot": {}, "empty_frac_base_a01": float(np.mean(s_["empty"])),
                               "op_a01": {m: {"mcs": float(pm[mi[m], i10]), "size": float(ps[mi[m], i10])}
                                          for m in METHODS}}
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
        outs.append(out)
    return outs


def run(tag: str, dataset: str, resplit: int, out_root: Path | None = None) -> Path:
    out = (out_root or get_paths().raw_results_root / "nonempty") / tag / f"r{resplit}"
    if (out / ".ok").exists():
        return out
    return run_multi(tag, dataset, [resplit], out_root)[0]
