#!/usr/bin/env python
"""Wall-clock cost of RBE-CP's views (added 2026-10-23): per-image latency of one forward pass vs the 13 views of
RBE-CP (view generation included: GPU ops, JPEG via CPU libjpeg as in extraction), the 8 views of the symmetric
variant and the 7 views identity + restorers, on one GPU, fp16 autocast, batch of 64 real ImageNet test images.
The conformal step itself (a max, an average and a threshold) is negligible and not timed.

  CUDA_VISIBLE_DEVICES=0 uv run python scripts/measure_latency.py      (on ws; results/tables/latency.json)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracecal.analysis.frontier import RESTORERS  # noqa: E402
from tracecal.data.sources import get_source  # noqa: E402
from tracecal.models.zoo import load  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402
from tracecal.probes.bank import ProbeBank  # noqa: E402
from tracecal.probes.ops import apply_op  # noqa: E402

MODELS = ["in_rn50_tv1", "in_vit_s16", "in_convnext_s", "in_vit_l16"]
BATCH, REPS, WARM = 64, 20, 5
WAIT_MAX_H = 12   # wait for the shared GPU to be free of other users' processes before timing


def other_procs() -> list[str]:
    """Compute processes on the physical GPU we time on, other than ours (nvidia-smi index = PCI order)."""
    import os

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout
    return [ln.strip() for ln in q.splitlines() if ln.strip() and ln.split(",")[0].strip() != str(os.getpid())]


@torch.inference_mode()
def timed(fn) -> float:
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) / BATCH * 1000.0   # ms per image


def main() -> None:
    t0 = time.time()
    while other_procs() and time.time() - t0 < WAIT_MAX_H * 3600:
        time.sleep(300)
    before = other_procs()
    print("other processes on the GPU at start:", before or "none", flush=True)
    src = get_source("imagenet:test10k")
    x = torch.stack([src.dataset[i][0] for i in range(BATCH)]).float().div(255).cuda()
    bank = ProbeBank("imagenet")
    spec = {s["name"]: s for s in bank.specs}
    out = {"gpu": torch.cuda.get_device_name(0), "batch": BATCH, "models": {}, "other_procs_start": before}
    for m in MODELS:
        w = load(m, "cuda")

        def run(names):
            def f():
                xs = [x]
                for nm in names:
                    if nm == "hflip":
                        xs.append(apply_op(x, {"op": "hflip"}))
                    else:
                        xs.append(apply_op(x, spec[nm], noise=bank.field(224, 224, x.device) if spec[nm]["op"] == "noise" else None))
                w(torch.cat(xs))
            return f

        all12 = bank.view_names                                  # 11 probes + hflip (+ identity = 13 views)
        res = {"1_view": timed(run([])), "7_views_id_restorers": timed(run(RESTORERS)),
               "8_views_symmetric": timed(run(RESTORERS + ["hflip"])), "13_views_rbe": timed(run(all12))}
        res["ratio_13_vs_1"] = res["13_views_rbe"] / res["1_view"]
        out["models"][m] = res
        print(m, {k: round(v, 3) for k, v in res.items()}, flush=True)
        del w
        torch.cuda.empty_cache()
    out["other_procs_end"] = other_procs()
    p = get_paths().results_root / "tables" / "latency.json"
    p.write_text(json.dumps(out, indent=1))
    print("written", p)


if __name__ == "__main__":
    main()
