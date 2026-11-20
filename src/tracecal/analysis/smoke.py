"""End-to-end smoke test on a small CIFAR subset with a RANDOM-INIT model (tag *_s99, never a real seed).

Exercises: probe ops, shift programs, extraction (all 9 pool versions, test, 2 corruptions), P' assembly, selection,
every method and gate, and the summary writer. Numbers produced here are meaningless by construction.
"""

from __future__ import annotations

import json
import time

import torch

from ..models.zoo import ResNet18Cifar, checkpoint_path
from ..paths import get_paths


def run_smoke(dataset: str = "cifar10", n: int = 2000) -> None:
    from ..extract import run as extract
    from ..shifts.programs import n_versions
    from .run import run as analyze

    t0 = time.time()
    tag = f"{'c10' if dataset == 'cifar10' else 'c100'}_resnet18_s99"
    ck = checkpoint_path(tag)
    if not ck.exists():
        torch.manual_seed(99)
        net = ResNet18Cifar(10 if dataset == "cifar10" else 100)
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(), "config": {"smoke": True}}, ck)
    for v in range(n_versions()):
        extract(tag, f"{dataset}:pool", v, limit=n, overwrite=True, batch_size=500)
    conds = [("clean", f"{dataset}:test"), ("gaussian_noise:3", f"{dataset}c:gaussian_noise:3"),
             ("fog:3", f"{dataset}c:fog:3")]
    for _, src in conds:
        extract(tag, src, 0, limit=n, overwrite=True, batch_size=500)
    out_root = get_paths().results_root / "smoke"
    out = analyze(tag, dataset, 0, "A", conditions=conds, allow_scaled=True, out_root=out_root)
    summ = json.loads((out / "summary.json").read_text())
    for m, d in summ["methods"].items():
        print(f"{m:28s} " + "  ".join(f"{c}: cov={v['coverage']:.3f} size={v['size']:.2f}" for c, v in d.items()))
    print(f"smoke OK in {time.time() - t0:.0f}s -> {out}")
