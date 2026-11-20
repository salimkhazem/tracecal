"""Evaluate pre-registration v4 (audits/gates_v4.yaml) from results/raw/frontier_v4/<tag>/r<k>/.

Primary set: the fresh v4 ImageNet models (configs/core/models_v4.yaml). Secondary sets (held out from v4's design,
used in the v3 confirmation): the v3 ImageNet models and CIFAR seeds 5-8.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.stats import holm
from ..paths import get_paths
from .dense_check import window
from .frontier_v4 import BASELINES
from .gates_v2_eval import hierarchical, pair_delta

PRIMARY = ["in_densenet121", "in_mnv3_l", "in_deit3_s", "in_maxvit_t", "in_mixer_b16"]
SECONDARY_IN = ["in_deit_s", "in_effnet_b0", "in_regnety16", "in_convnext_s", "in_vit_s16"]
CIFAR_TAGS = {f"{d}/{a}": [f"{'c10' if d == 'cifar10' else 'c100'}_{a}_s{s}" for s in (5, 6, 7, 8)]
              for d in ("cifar10", "cifar100") for a in ("resnet18", "vits16")}
N1_HOLM = ["base", "tta_learned", "union_all", "aps_base"]
CIFAR_HOLM = ["base", "tta_avg", "tta_learned", "union_all", "aps_base"]
NATS = ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")
NAT_BASELINES = ("base", "tta_avg", "tta_flip", "ru", "rbe3_sel")
NI_MARGIN = 0.10 / 100          # non-inferiority margin, MCS fraction units
A_MARGIN = 1.00 / 100           # ImageNet-A margin vs split CP (N2b)


def run_dir(tag: str, r: int) -> Path:
    return get_paths().raw_results_root / "frontier_v4" / tag / f"r{r}"


def _deltas(d: dict, refs) -> dict[str, Any]:
    w = window(d["ps_base"])
    nb = d["bm_base"].shape[1]
    out: dict[str, Any] = {"point": {}, "boot": {}}
    for b in refs:
        size = np.stack([d["ps_rbe4_sel"], d[f"ps_{b}"]])
        mcs = np.stack([d["pm_rbe4_sel"], d[f"pm_{b}"]])
        out["point"][b] = pair_delta(size, mcs, 0, 1, w)
        bsz = np.stack([d["bs_rbe4_sel"], d[f"bs_{b}"]])
        bmc = np.stack([d["bm_rbe4_sel"], d[f"bm_{b}"]])
        out["boot"][b] = np.array([pair_delta(bsz[..., k], bmc[..., k], 0, 1, window(d["bs_base"][:, k]))
                                   for k in range(nb)])
    return out


def _collect(tags: list[str], resplits=range(5)):
    deltas, groups, metas = {}, {}, {}
    for t in tags:
        for r in resplits:
            if (run_dir(t, r) / ".ok").exists():
                d = dict(np.load(run_dir(t, r) / "curves.npz"))      # decompress each array once
                deltas[(t, r)] = _deltas(d, BASELINES)
                groups[(t, r)] = t
                metas[(t, r)] = json.loads((run_dir(t, r) / "meta.json").read_text())
    return deltas, groups, metas


def _summary(deltas, groups, metas, holm_set) -> dict[str, Any]:
    res = {b: hierarchical(deltas, groups, b) for b in BASELINES}
    hl = holm({b: res[b]["p_two_sided"] for b in holm_set}, 0.05)
    for b in BASELINES:
        res[b]["holm_reject"] = bool(hl[b]["reject"]) if b in hl else None
    ms = list(metas.values())
    return {"comparisons": res, "lam4_star": [m["lam4_star"] for m in ms], "lam3_star": [m["lam3_star"] for m in ms],
            "clean_cov_mean": float(np.mean([m["clean_cov_a01"]["rbe4_sel"] for m in ms])),
            "restore_level_mean": float(np.mean([m["restore_level_a01"] for m in ms])), "n_runs": len(ms)}


def naturals(models: list[str], resplits=range(5)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in models:
        for nat in NATS:
            deltas = {}
            for r in resplits:
                p = run_dir(m, r) / f"naturals_{nat}.npz"
                if not p.exists():
                    continue
                d = dict(np.load(p))
                deltas[(m, r)] = _deltas(d, NAT_BASELINES)
            if deltas:
                out.setdefault(m, {})[nat] = {b: hierarchical(deltas, None, b) for b in NAT_BASELINES}
    return out


def evaluate() -> dict[str, Any]:
    out: dict[str, Any] = {"primary_imagenet_c": {}, "secondary_imagenet_c": {}, "cifar": {}}
    for key, models in (("primary_imagenet_c", PRIMARY), ("secondary_imagenet_c", SECONDARY_IN)):
        for m in models:
            deltas, _, metas = _collect([m])
            if deltas:
                out[key][m] = _summary(deltas, None, metas, N1_HOLM)
    for g, tags in CIFAR_TAGS.items():
        deltas, groups, metas = _collect(tags)
        if deltas:
            out["cifar"][g] = _summary(deltas, groups, metas, CIFAR_HOLM)
    out["primary_naturals"] = naturals(PRIMARY)
    out["secondary_naturals"] = naturals(SECONDARY_IN)
    P = out["primary_imagenet_c"]
    nat = out["primary_naturals"]
    full_p = len(P) == len(PRIMARY) and all(len(nat.get(m, {})) == len(NATS) for m in PRIMARY)

    def cnt(pred, models):
        return int(sum(1 for m in models if pred(m)))

    # N1: ImageNet-C, primary
    n1a = cnt(lambda m: all(P[m]["comparisons"][b]["holm_reject"] and P[m]["comparisons"][b]["delta"] < 0
                            for b in N1_HOLM), P)
    n1b = cnt(lambda m: P[m]["comparisons"]["tta_avg"]["ci_hi"] < NI_MARGIN, P)
    out["N1_imagenet_c"] = {"n_beats_4": n1a, "n_noninferior_tta_avg": n1b,
                            "verdict": ("PASS" if n1a >= 4 and n1b >= 3 else "FAIL") if full_p else "INCOMPLETE"}
    # N2: natural shifts, primary
    n2a = cnt(lambda m: nat[m]["imagenet_a"]["tta_avg"]["ci_hi"] < 0, nat)
    n2b = cnt(lambda m: nat[m]["imagenet_a"]["base"]["ci_hi"] < A_MARGIN, nat)
    n2c_s = cnt(lambda m: nat[m]["imagenet_sketch"]["base"]["ci_hi"] < 0, nat)
    n2c_r = cnt(lambda m: nat[m]["imagenet_r"]["base"]["ci_hi"] < 0, nat)
    ok2 = n2a >= 4 and n2b >= 4 and n2c_s >= 4 and n2c_r >= 4
    out["N2_naturals"] = {"a_vs_tta_avg_on_A": n2a, "b_A_within_margin_of_split": n2b, "c_sketch_vs_split": n2c_s,
                          "c_r_vs_split": n2c_r, "verdict": ("PASS" if ok2 else "FAIL") if full_p else "INCOMPLETE"}
    # N3: v4 vs v3, primary
    n3a = cnt(lambda m: nat[m]["imagenet_a"]["rbe3_sel"]["ci_hi"] < 0, nat)
    n3b = cnt(lambda m: P[m]["comparisons"]["rbe3_sel"]["ci_hi"] < NI_MARGIN, P)
    out["N3_vs_v3"] = {"a_better_on_A": n3a, "b_noninferior_on_C": n3b,
                       "verdict": ("PASS" if n3a >= 4 and n3b >= 3 else "FAIL") if full_p else "INCOMPLETE"}
    # N4: CIFAR (secondary set)
    C = out["cifar"]
    n4 = cnt(lambda g: all(C[g]["comparisons"][b]["holm_reject"] and C[g]["comparisons"][b]["delta"] < 0
                           for b in CIFAR_HOLM), C)
    out["N4_cifar"] = {"n_pass": n4, "verdict": ("PASS" if n4 >= 3 else "FAIL") if len(C) == 4 else "INCOMPLETE"}
    # N5: clean coverage
    covs = [v["clean_cov_mean"] for k in ("primary_imagenet_c", "secondary_imagenet_c", "cifar") for v in out[k].values()]
    out["N5_clean"] = {"clean_cov_min": float(min(covs)) if covs else None, "pass": bool(covs and min(covs) >= 0.89)}
    out["decision"] = ("v4 replaces v3" if all(out[k]["verdict"] == "PASS" for k in ("N1_imagenet_c", "N2_naturals",
                                                                                     "N3_vs_v3"))
                       else "v3 stays the method" if full_p else "INCOMPLETE")
    return out


def write(res: dict[str, Any]) -> Path:
    p = get_paths().project_root / "audits" / "gates_v4_results.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    return p
