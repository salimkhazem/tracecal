"""G1 (signal) diagnostics: does R carry information beyond scalar disagreement and confidence?

(a) corruption-name decoding on CIFAR-C severities 3-5 (diagnostic only), 5-fold CV multinomial logistic regression,
    1000 fixed test images per condition; paired bootstrap over rows of the CV predictions.
(b) correctness AUROC: logistic regression fit on the P' tune split, evaluated per CIFAR-C condition; bootstrap over
    conditions of the mean AUROC difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import families, protocol
from .bundles import load_bundle
from .featurize import features
from .pprime import assemble, pprime_versions
from .run import PoolData

SETS = ("trace", "scalar_s", "conf")


def _boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        d.append(float(a[i].mean() - b[i].mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(a.mean() - b.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def g1_for_model(model_tag: str, dataset: str) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    corr = protocol()["corruptions"]["all"]
    rng = np.random.default_rng(0)
    rows_idx = np.sort(rng.choice(10000, size=1000, replace=False))
    # (a) decoding
    X = {k: [] for k in SETS}
    y = []
    for ci, c in enumerate(corr):
        for s in (3, 4, 5):
            b = load_bundle(model_tag, f"{dataset}c:{c}:{s}", 0, cache=False).take(rows_idx)
            for k in SETS:
                X[k].append(features(b, k))
            y.append(np.full(len(rows_idx), ci))
    y = np.concatenate(y)
    correct = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for k in SETS:
        Z = np.concatenate(X[k])
        pred = np.empty(len(y), dtype=np.int64)
        for tr, te in skf.split(Z, y):
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
            clf = LogisticRegression(max_iter=3000).fit((Z[tr] - mu) / sd, y[tr])
            pred[te] = clf.predict((Z[te] - mu) / sd)
        correct[k] = (pred == y).astype(np.float64)
    dec = {k: float(v.mean()) for k, v in correct.items()}
    dec_diff = {f"trace_vs_{k}": _boot_diff(correct["trace"], correct[k]) for k in ("scalar_s", "conf")}
    # (b) correctness AUROC: train on P' tune split (re-split 0, p0 = 0.35, all families)
    pd = PoolData(model_tag, dataset, 0, with_views=False)
    pos = pd.pos["tune"]
    ver = pprime_versions(pd.ids[pos], 0.35, families(), pd.family)
    tune = assemble(pd.versions, pos, ver)
    ytr = (tune.probs.argmax(1) == tune.labels).astype(int)
    models = {}
    for k in SETS:
        Z = features(tune, k)
        mu, sd = Z.mean(0), Z.std(0) + 1e-8
        models[k] = (LogisticRegression(max_iter=3000).fit((Z - mu) / sd, ytr), mu, sd)
    auc = {k: [] for k in SETS}
    for c in corr:
        for s in protocol()["corruptions"]["severities"]:
            b = load_bundle(model_tag, f"{dataset}c:{c}:{s}", 0, cache=False)
            yc = (b.probs.argmax(1) == b.labels).astype(int)
            if yc.min() == yc.max():
                continue
            for k in SETS:
                clf, mu, sd = models[k]
                auc[k].append(roc_auc_score(yc, clf.decision_function((features(b, k) - mu) / sd)))
    auc_arr = {k: np.array(v) for k, v in auc.items()}
    auc_diff = {f"trace_vs_{k}": _boot_diff(auc_arr["trace"], auc_arr[k]) for k in ("scalar_s", "conf")}
    passed = all(d["ci_lo"] > 0 for d in list(dec_diff.values()) + list(auc_diff.values()))
    return {"decoding_acc": dec, "decoding_diff": dec_diff, "auroc_mean": {k: float(v.mean()) for k, v in auc_arr.items()},
            "auroc_diff": auc_diff, "pass": bool(passed)}


def g1() -> dict[str, Any]:
    out = {}
    for ds in ("cifar10", "cifar100"):
        for arch in ("resnet18", "vits16"):
            tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s0"
            out[f"{ds}/{arch}"] = g1_for_model(tag, ds)
    out["verdict"] = "PASS" if all(v["pass"] for v in out.values() if isinstance(v, dict)) else "FAIL"
    return out
