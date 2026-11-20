#!/usr/bin/env python
"""Run analysis jobs (CPU) in parallel processes with a fixed worker count; skips finished runs.

  uv run python scripts/run_cpu_jobs.py cifar --workers 8
  uv run python scripts/run_cpu_jobs.py imagenet --workers 4 --naturals
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SEEDS = [0, 1, 2, 3, 4]
RESPLITS = [0, 1, 2, 3, 4]


def _run(cmd: str, log: str) -> tuple[str, int]:
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               CUDA_VISIBLE_DEVICES="")
    with open(log, "w") as f:
        rc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return cmd, rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["cifar", "imagenet", "imagenet-lfo", "cifar-lfo", "frontier-cifar",
                                      "frontier-imagenet", "ablation-cifar", "ablation-imagenet",
                                      "complementary-cifar", "complementary-imagenet", "mixed-cifar", "mixed-imagenet", "dense-cifar", "dense-imagenet", "nonempty-cifar", "nonempty-imagenet", "strength-cifar", "v3"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--naturals", action="store_true")
    ap.add_argument("--extra", default="")
    ap.add_argument("--tags", default="", help="comma-separated model tags to restrict to (default: all)")
    ap.add_argument("--only-ready", action="store_true", help="skip models whose extraction is not complete")
    a = ap.parse_args()
    keep = set(t for t in a.tags.split(",") if t)
    cmds = []
    if a.which == "frontier-cifar":
        for seed in SEEDS:
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds += [f"uv run tracecal frontier {tag} {ds} {r}" for r in RESPLITS]
    elif a.which == "ablation-cifar":
        for seed in (1, 2, 3, 4):
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds += [f"uv run tracecal ablation-bank {tag} {ds} {r}" for r in RESPLITS]
    elif a.which == "ablation-imagenet":
        for m in ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]:
            cmds += [f"uv run tracecal ablation-bank {m} imagenet {r}" for r in RESPLITS]
    elif a.which == "complementary-cifar":
        for seed in (1, 2, 3, 4):
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds += [f"uv run tracecal complementary {tag} {ds} {r}" for r in RESPLITS]
    elif a.which == "mixed-cifar":
        for seed in (1, 2, 3, 4):
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds += [f"uv run tracecal mixed-stream {tag} {ds} {r}" for r in RESPLITS]
    elif a.which == "dense-cifar":
        for seed in (1, 2, 3, 4):
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds += [f"uv run tracecal dense-check {tag} {ds} {r}" for r in RESPLITS]
    elif a.which == "nonempty-cifar":
        for seed in (1, 2, 3, 4):
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds += [f"uv run tracecal nonempty {tag} {ds} {r}" for r in RESPLITS]
    elif a.which == "v3":
        for m in ["in_deit_s", "in_effnet_b0", "in_regnety16", "in_convnext_s", "in_vit_s16"]:
            cmds.append(f"uv run tracecal frontier-v3 {m} imagenet")
        for seed in (5, 6, 7, 8):
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds.append(f"uv run tracecal frontier-v3 {tag} {ds}")
    elif a.which == "strength-cifar":
        for seed in (1, 2, 3, 4):
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    cmds += [f"uv run tracecal strength {tag} {ds} {r}" for r in RESPLITS]
    elif a.which == "nonempty-imagenet":
        for m in ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]:
            cmds += [f"uv run tracecal nonempty {m} imagenet {r}" for r in RESPLITS]
    elif a.which == "dense-imagenet":
        for m in ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]:
            cmds += [f"uv run tracecal dense-check {m} imagenet {r}" for r in RESPLITS]
    elif a.which == "mixed-imagenet":
        for m in ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]:
            cmds += [f"uv run tracecal mixed-stream {m} imagenet {r}" for r in RESPLITS]
    elif a.which == "complementary-imagenet":
        for m in ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]:
            cmds += [f"uv run tracecal complementary {m} imagenet {r}" for r in RESPLITS]
    elif a.which == "frontier-imagenet":
        for m in ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]:
            cmds += [f"uv run tracecal frontier {m} imagenet {r}" + (" --naturals" if a.naturals else "")
                     for r in RESPLITS]
    elif a.which.startswith("cifar"):
        protos = ["A"] if a.which == "cifar" else ["B:noise", "B:blur", "B:weather", "B:digital"]
        for seed in SEEDS:
            for ds in ("cifar10", "cifar100"):
                for arch in ("resnet18", "vits16"):
                    tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                    for r in RESPLITS:
                        for p in protos:
                            cmds.append(f"uv run tracecal analyze {tag} {ds} {r} --proto {p} {a.extra}")
    else:
        protos = ["A"] if a.which == "imagenet" else ["B:noise", "B:blur", "B:weather", "B:digital"]
        for m in ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]:
            for r in RESPLITS:
                for p in protos:
                    nat = " --naturals" if (a.naturals and p == "A") else ""
                    cmds.append(f"uv run tracecal analyze {m} imagenet {r} --proto {p}{nat} {a.extra}")
    if keep:
        cmds = [c for c in cmds if c.split()[4] in keep]
    if a.only_ready:
        sys.path.insert(0, "src")
        from tracecal.extract import required_sources  # noqa: E402
        from tracecal.fsio import is_done  # noqa: E402

        ready = {}
        for c in cmds:
            tag, ds = c.split()[4:6]
            if tag not in ready:
                ready[tag] = all(is_done(tag, s, v) for s, v in required_sources(ds))
        cmds = [c for c in cmds if ready[c.split()[4]]]
    os.makedirs("results/logs/analysis", exist_ok=True)
    t0 = time.time()
    fails = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run, "source configs/ws/env.sh && export CUDA_VISIBLE_DEVICES= && " + c,
                          f"results/logs/analysis/{'_'.join(c.split()[2:]).replace(':', '-').replace('--', '')}.log"): c
                for c in cmds}
        for f in as_completed(futs):
            cmd, rc = f.result()
            fails += rc != 0
            print(f"[{time.time() - t0:7.0f}s] rc={rc} {futs[f]}", flush=True)
    print(f"done: {len(cmds)} jobs, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
