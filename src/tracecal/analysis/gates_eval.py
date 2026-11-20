"""Evaluate the pre-registered gates (audits/gates.yaml) from raw run outputs. Results go to
audits/gates_results.json; nothing here may change a threshold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import protocol
from ..paths import get_paths
from ..utils.io import atomic_write_json, read_json
from ..evaluation.stats import hierarchical_delta_mcs, holm, spearman_boot_ci

G2_BASELINES = ["split_clean", "pprime_marginal", "mondrian_conf", "mondrian_scalar_s"]
CIFAR_PAIRS = [("cifar10", "resnet18"), ("cifar10", "vits16"), ("cifar100", "resnet18"), ("cifar100", "vits16")]
IMAGENET_MODELS = ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]


def run_dir(model_tag: str, resplit: int, proto: str = "A", alpha: float = 0.1, score: str = "lac") -> Path:
    return get_paths().raw_results_root / "runs" / model_tag / f"r{resplit}" / f"{proto.replace(':', '-')}_a{alpha:g}_{score}"


def load_run(model_tag: str, resplit: int, proto: str = "A") -> tuple[dict, Any]:
    d = run_dir(model_tag, resplit, proto)
    if not (d / ".ok").exists():
        raise FileNotFoundError(d)
    return read_json(d / "summary.json"), np.load(d / "arrays.npz")


def cov_matrix(arrays, method: str, conditions: list[str]) -> np.ndarray:
    return np.stack([arrays[f"{method}|{c}|cov"] for c in conditions])


def size_mean(arrays, method: str, conditions: list[str]) -> float:
    return float(np.mean([arrays[f"{method}|{c}|size"].mean() for c in conditions]))


def decisive(replicates: list[tuple[str, int, Any]], conditions: list[str], alpha: float, groups: dict | None,
             method: str = "mondrian_trace") -> dict[str, Any]:
    """G2/G3 criterion for one (dataset, architecture) pair or one ImageNet model."""
    res: dict[str, Any] = {"comparisons": {}}
    cov_t = {f"{t}|{r}": cov_matrix(a, method, conditions) for t, r, a in replicates}
    for base in G2_BASELINES:
        cov_b = {f"{t}|{r}": cov_matrix(a, base, conditions) for t, r, a in replicates}
        res["comparisons"][base] = hierarchical_delta_mcs(cov_t, cov_b, alpha, groups=groups)
    hl = holm({b: v["p_two_sided"] for b, v in res["comparisons"].items()}, 0.05)
    for b in G2_BASELINES:
        res["comparisons"][b]["holm_reject"] = bool(hl[b]["reject"])
        res["comparisons"][b]["holm_threshold"] = hl[b]["threshold"]
    s_t = np.mean([size_mean(a, method, conditions) for _, _, a in replicates])
    s_p = np.mean([size_mean(a, "pprime_marginal", conditions) for _, _, a in replicates])
    res["size_ratio_vs_pprime"] = float(s_t / s_p)
    res["pass"] = bool(all(res["comparisons"][b]["holm_reject"] and res["comparisons"][b]["delta"] < 0
                           for b in G2_BASELINES) and res["size_ratio_vs_pprime"] <= 1.15)
    return res


def g2(alpha: float = 0.1) -> dict[str, Any]:
    out: dict[str, Any] = {"pairs": {}}
    resplits = protocol()["splits"]["cifar"]["pool_resplits"]
    seeds = protocol()["models"]["cifar"]["resnet18"]["seeds"]
    for ds, arch in CIFAR_PAIRS:
        reps, groups = [], {}
        conds = None
        for s in seeds:
            tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{s}"
            for r in resplits:
                summ, arr = load_run(tag, r)
                conds = conds or summ["conditions"]
                reps.append((tag, r, arr))
                groups[f"{tag}|{r}"] = tag
        out["pairs"][f"{ds}/{arch}"] = decisive(reps, conds, alpha, groups)
    n_pass = sum(v["pass"] for v in out["pairs"].values())
    out["n_pass"] = n_pass
    out["verdict"] = "PASS" if n_pass >= 3 else "FAIL"
    narrow = []
    for k, v in out["pairs"].items():
        c = v["comparisons"]
        if (c["split_clean"]["holm_reject"] and c["pprime_marginal"]["holm_reject"]
                and not (c["mondrian_conf"]["holm_reject"] and c["mondrian_scalar_s"]["holm_reject"])):
            narrow.append(k)
    out["narrow_condition_pairs"] = narrow
    return out


def g3(alpha: float = 0.1) -> dict[str, Any]:
    out: dict[str, Any] = {"models": {}}
    resplits = protocol()["splits"]["imagenet"]["pool_resplits"]
    for m in IMAGENET_MODELS:
        reps, conds = [], None
        for r in resplits:
            summ, arr = load_run(m, r)
            conds = conds or summ["conditions"]
            reps.append((m, r, arr))
        out["models"][m] = decisive(reps, conds, alpha, None)
    n_pass = sum(v["pass"] for v in out["models"].values())
    out["n_pass"], out["verdict"] = n_pass, "PASS" if n_pass >= 3 else "FAIL"
    return out


def g4(beta: float = 0.05, alpha: float = 0.1, runs: list[tuple[str, int, str]] | None = None) -> dict[str, Any]:
    """Spearman(abstention rate, TRACE shortfall) across (run, condition) pairs; false abstention on clean."""
    out: dict[str, Any] = {}
    rows = {g: [] for g in ("trace_knn", "maxsoftmax", "energy", "embed_maha")}
    clean_abst = []
    for tag, r, proto in runs or []:
        summ, arr = load_run(tag, r, proto)
        for c in summ["conditions"] + [n for n in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")
                                       if f"mondrian_trace|{n}|cov" in arr]:
            sf = max(0.0, 1 - alpha - float(arr[f"mondrian_trace|{c}|cov"].mean()))
            for g in rows:
                rows[g].append((float((arr[f"gate_{g}|{c}|p"] < beta).mean()), sf))
            if c == "clean":
                clean_abst.append(float((arr[f"gate_trace_knn|{c}|p"] < beta).mean()))
    for g, v in rows.items():
        if len(v) >= 5:
            x, y = np.array(v).T
            out[g] = spearman_boot_ci(x, y)
    out["false_abstention_clean"] = float(np.mean(clean_abst)) if clean_abst else None
    if "trace_knn" in out:
        t = out["trace_knn"]
        out["pass"] = bool(t["rho"] >= 0.5 and t["ci_lo"] > 0 and out["false_abstention_clean"] is not None
                           and out["false_abstention_clean"] <= beta + 0.01)
    return out


def write(results: dict[str, Any]) -> Path:
    p = get_paths().project_root / "audits" / "gates_results.json"
    prev = read_json(p) if p.exists() else {}
    prev.update(results)
    atomic_write_json(p, prev)
    return p
