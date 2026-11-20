#!/usr/bin/env python
"""Write job files for scripts/gpu_queue.py (one job per GPU at a time; phases run sequentially).

  uv run python scripts/make_jobs.py train   > configs/core/jobs_train.json
  uv run python scripts/make_jobs.py extract-cifar > configs/core/jobs_extract_cifar.json
  uv run python scripts/make_jobs.py extract-imagenet > configs/core/jobs_extract_imagenet.json
Analysis (CPU) jobs are launched with scripts/run_cpu_jobs.py instead.
"""

from __future__ import annotations

import json
import sys

PRE = "source configs/ws/env.sh && "
SEEDS = [0, 1, 2, 3, 4]


def train() -> dict:
    # ViT fine-tunes are short; ResNet-18 runs are long -> interleave so both GPUs stay busy
    jobs = []
    for seed in SEEDS:
        for ds in ("cifar10", "cifar100"):
            for arch in ("resnet18", "vits16"):
                tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                jobs.append({"name": f"train_{tag}", "cmd": PRE + f"uv run tracecal train-cifar {ds} {arch} {seed}"})
    return {"phases": [jobs]}


def extract_cifar() -> dict:
    jobs = []
    for seed in SEEDS:
        for ds in ("cifar10", "cifar100"):
            for arch in ("resnet18", "vits16"):
                tag = f"{'c10' if ds == 'cifar10' else 'c100'}_{arch}_s{seed}"
                bs = 1000 if arch == "resnet18" else 256
                jobs.append({"name": f"extract_{tag}",
                             "cmd": PRE + f"uv run tracecal extract-all {tag} {ds} --batch-size {bs} --workers 0"})
    return {"phases": [jobs]}


def extract_imagenet() -> dict:
    models = ["in_rn50", "in_convnext_t", "in_swin_t", "in_vitb16", "in_clip_b16"]
    pool = [{"name": f"extract_pool_{m}", "cmd": PRE + f"uv run tracecal extract-all {m} imagenet --only pool "
             f"--batch-size 256 --workers 12"} for m in models]
    test = [{"name": f"extract_test_{m}", "cmd": PRE + f"uv run tracecal extract-all {m} imagenet --only test "
             f"--batch-size 256 --workers 12"} for m in models]
    nat = [{"name": f"extract_nat_{m}", "cmd": PRE + f"uv run tracecal extract-all {m} imagenet --only naturals "
            f"--batch-size 256 --workers 12"} for m in models]
    return {"phases": [pool + test + nat]}


if __name__ == "__main__":
    what = sys.argv[1]
    fn = {"train": train, "extract-cifar": extract_cifar, "extract-imagenet": extract_imagenet}[what]
    print(json.dumps(fn(), indent=1))
