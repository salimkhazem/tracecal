"""Evaluate pre-registration v3 (audits/gates_v3.yaml) from results/raw/frontier_v3/<tag>/r<k>/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.stats import holm
from ..paths import get_paths
from .dense_check import window
from .frontier_v3 import BASELINES, LAMBDAS, lam_name
from .gates_v2_eval import hierarchical, pair_delta

EXTERNAL = ["base", "tta_avg", "tta_learned", "union_all", "aps_base"]
CIFAR_TAGS = {f"{d}/{a}": [f"{'c10' if d == 'cifar10' else 'c100'}_{a}_s{s}" for s in (5, 6, 7, 8)]
              for d in ("cifar10", "cifar100") for a in ("resnet18", "vits16")}
IMAGENET = ["in_deit_s", "in_effnet_b0", "in_regnety16", "in_convnext_s", "in_vit_s16"]
MARGIN = 0.10 / 100           # in MCS (fraction) units


def run_dir(tag: str, r: int) -> Path:
    return get_paths().raw_results_root / "frontier_v3" / tag / f"r{r}"


def run_deltas(tag: str, r: int) -> dict[str, Any]:
    d = dict(np.load(run_dir(tag, r) / "curves.npz"))   # decompress each array once
    meta = json.loads((run_dir(tag, r) / "meta.json").read_text())
    refs = BASELINES
    w = window(d["ps_base"])
    out: dict[str, Any] = {"point": {}, "boot": {}, "lam_star": meta["lam_star"],
                           "clean_cov": meta["clean_cov_a01"].get("rbe_sel"), "restore_level": meta["restore_level_a01"]}
    nb = d["bm_base"].shape[1]
    for b in refs:
        size = np.stack([d["ps_rbe_sel"], d[f"ps_{b}"]])
        mcs = np.stack([d["pm_rbe_sel"], d[f"pm_{b}"]])
        out["point"][b] = pair_delta(size, mcs, 0, 1, w)
        bsz = np.stack([d["bs_rbe_sel"], d[f"bs_{b}"]])
        bmc = np.stack([d["bm_rbe_sel"], d[f"bm_{b}"]])
        out["boot"][b] = np.array([pair_delta(bsz[..., k], bmc[..., k], 0, 1, window(d["bs_base"][:, k]))
                                   for k in range(nb)])
    # fixed-lam variants (point only, report)
    out["fixed"] = {}
    for lam in LAMBDAS:
        m = lam_name(lam)
        for b in ("tta_avg", "ru"):
            size = np.stack([d[f"ps_{m}"], d[f"ps_{b}"]])
            mcs = np.stack([d[f"pm_{m}"], d[f"pm_{b}"]])
            out["fixed"][f"{m}_vs_{b}"] = pair_delta(size, mcs, 0, 1, w)
    return out


def _collect(tags: list[str], resplits=range(5)):
    deltas, groups = {}, {}
    for t in tags:
        for r in resplits:
            if (run_dir(t, r) / ".ok").exists():
                deltas[(t, r)] = run_deltas(t, r)
                groups[(t, r)] = t
    return deltas, groups


def _summ(deltas, groups, holm_set: list[str]) -> dict[str, Any]:
    res = {b: hierarchical(deltas, groups, b) for b in BASELINES}
    hl = holm({b: res[b]["p_two_sided"] for b in holm_set}, 0.05)
    for b in BASELINES:
        res[b]["holm_reject"] = bool(hl[b]["reject"]) if b in hl else None
    lam = [v["lam_star"] for v in deltas.values()]
    fixed = {k: float(np.nanmean([v["fixed"][k] for v in deltas.values()]))
             for k in next(iter(deltas.values()))["fixed"]}
    return {"comparisons": res, "lam_star": lam, "fixed_point_mean": fixed,
            "clean_cov_mean": float(np.mean([v["clean_cov"] for v in deltas.values()])),
            "restore_level_mean": float(np.mean([v["restore_level"] for v in deltas.values()])),
            "n_runs": len(deltas)}


def evaluate() -> dict[str, Any]:
    out: dict[str, Any] = {"K1_cifar": {"settings": {}}, "K2_imagenet": {"models": {}}}
    for g, tags in CIFAR_TAGS.items():
        deltas, groups = _collect(tags)
        if not deltas:
            continue
        s = _summ(deltas, groups, EXTERNAL)
        c = s["comparisons"]
        s["pass"] = bool(all(c[b]["holm_reject"] and c[b]["delta"] < 0 for b in EXTERNAL) and c["ru"]["ci_hi"] < MARGIN)
        out["K1_cifar"]["settings"][g] = s
    k2_set = ["base", "tta_learned", "union_all", "aps_base", "ru"]
    for m in IMAGENET:
        deltas, groups = _collect([m])
        if not deltas:
            continue
        s = _summ(deltas, None, k2_set)
        c = s["comparisons"]
        s["pass"] = bool(all(c[b]["holm_reject"] and c[b]["delta"] < 0 for b in k2_set) and c["tta_avg"]["ci_hi"] < MARGIN)
        s["improves_on_v2"] = bool(c["ru"]["ci_hi"] < 0)
        out["K2_imagenet"]["models"][m] = s
    n1 = sum(v["pass"] for v in out["K1_cifar"]["settings"].values())
    n2 = sum(v["pass"] for v in out["K2_imagenet"]["models"].values())
    n3 = sum(v["improves_on_v2"] for v in out["K2_imagenet"]["models"].values())
    full1, full2 = len(out["K1_cifar"]["settings"]) == 4, len(out["K2_imagenet"]["models"]) == 5
    out["K1_cifar"].update(n_pass=n1, verdict=("PASS" if n1 >= 3 else "FAIL") if full1 else "INCOMPLETE")
    out["K2_imagenet"].update(n_pass=n2, verdict=("PASS" if n2 >= 3 else "FAIL") if full2 else "INCOMPLETE")
    out["K3_improvement_over_v2_imagenet"] = {"n_improves": n3,
                                              "verdict": ("PASS" if n3 >= 3 else "FAIL") if full2 else "INCOMPLETE"}
    covs = [v["clean_cov_mean"] for v in out["K1_cifar"]["settings"].values()] + \
           [v["clean_cov_mean"] for v in out["K2_imagenet"]["models"].values()]
    out["K4_guarantee_sanity"] = {"clean_cov_min": float(min(covs)) if covs else None,
                                  "pass": bool(covs and min(covs) >= 0.89)}
    return out


def write(res: dict[str, Any]) -> Path:
    p = get_paths().project_root / "audits" / "gates_v3_results.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    return p


NATS = ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")


def k5_naturals(resplits=range(5)) -> dict[str, Any]:
    """K5 (report only): RBE-CP vs split CP / TTA average / RU-CP on the natural-shift sets, fresh ImageNet models."""
    out: dict[str, Any] = {"report_only": True, "models": {}}
    for m in IMAGENET:
        for nat in NATS:
            deltas = {}
            for r in resplits:
                p = run_dir(m, r) / f"naturals_{nat}.npz"
                if not p.exists():
                    continue
                d = dict(np.load(p))   # decompress each array once
                w = window(d["ps_base"])
                nb = d["bm_base"].shape[1]
                dd = {"point": {}, "boot": {}}
                for b in ("base", "tta_avg", "ru"):
                    size = np.stack([d["ps_rbe_sel"], d[f"ps_{b}"]])
                    mcs = np.stack([d["pm_rbe_sel"], d[f"pm_{b}"]])
                    dd["point"][b] = pair_delta(size, mcs, 0, 1, w)
                    bsz = np.stack([d["bs_rbe_sel"], d[f"bs_{b}"]])
                    bmc = np.stack([d["bm_rbe_sel"], d[f"bm_{b}"]])
                    dd["boot"][b] = np.array([pair_delta(bsz[..., k], bmc[..., k], 0, 1, window(d["bs_base"][:, k]))
                                              for k in range(nb)])
                deltas[(m, r)] = dd
            if deltas:
                out["models"].setdefault(m, {})[nat] = {b: hierarchical(deltas, None, b) for b in ("base", "tta_avg", "ru")}
    return out
