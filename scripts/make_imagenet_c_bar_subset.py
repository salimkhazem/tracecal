#!/usr/bin/env python
"""ImageNet-C-bar (Mintun et al., NeurIPS 2021) on our fixed 10k ImageNet test subset (pre-registration v5b).

Faithful to the official generator (facebookresearch/augmentation-corruption, imagenet_c_bar/make_imagenet_c_bar.py, MIT):
PIL bilinear Resize(256) -> CenterCrop(224) -> uint8 numpy -> official transform (build_transform(name, severity,
'imagenet')) -> Image.fromarray(np.uint8(out)).save(quality=85, optimize=True). Severity values from the official
imagenet_c_bar.csv. Deviation (documented): a per-image seed hash(val_id, corruption, severity index) instead of one
global seed, so any subset is reproducible.

Writes DATA_ROOT/imagenet_c_bar/<corruption>/<severity 1..5>/<val_id>.JPEG
  uv run --with scikit-image python scripts/make_imagenet_c_bar_subset.py [--workers 20]      (on ws)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CBAR = REPO / ".cache" / "src" / "augmentation-corruption" / "imagenet_c_bar"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(CBAR))


def severities() -> dict[str, list[float]]:
    out = {}
    for line in (CBAR / "imagenet_c_bar.csv").read_text().splitlines():
        vals = line.strip().split(",")
        if len(vals) == 6:
            out[vals[0]] = [float(v) for v in vals[1:]]
    return out


def out_root() -> Path:
    return Path(os.environ["DATA_ROOT"]) / "imagenet_c_bar"


def _work(job: tuple[str, int, float, list[int], list[int]]) -> int:
    from PIL import Image

    from tracecal.data.imagenet import clean_preprocess
    from tracecal.data.sources import imagenet_val
    from transform_finder import build_transform

    name, sev_idx, sev, positions, vids = job
    val = imagenet_val()
    t = build_transform(name=name, severity=sev, dataset_type="imagenet")
    d = out_root() / name / str(sev_idx)
    d.mkdir(parents=True, exist_ok=True)
    done = 0
    for pos, vid in zip(positions, vids):
        dst = d / f"{vid}.JPEG"
        if dst.exists():
            continue
        seed = int(hashlib.sha256(f"{vid}:{name}:{sev_idx}".encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        img = clean_preprocess(val.image(int(pos)))
        out = t(img)
        tmp = d / f".{vid}.tmp.JPEG"
        Image.fromarray(np.uint8(out)).save(tmp, quality=85, optimize=True)
        tmp.replace(dst)
        done += 1
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None, help="images per condition (smoke test)")
    args = ap.parse_args()
    from tracecal.data.sources import imagenet_split_ids

    ids = imagenet_split_ids()
    pos, vids = ids["csub_pos"], ids["csub_ids"]
    if args.limit:
        pos, vids = pos[: args.limit], vids[: args.limit]
    jobs = []
    for name, sevs in severities().items():
        for k, sev in enumerate(sevs, start=1):
            for i in range(0, len(pos), args.chunk):
                jobs.append((name, k, sev, pos[i : i + args.chunk].tolist(), vids[i : i + args.chunk].tolist()))
    t0 = time.time()
    n = 0
    with Pool(args.workers) as p:
        for j, c in enumerate(p.imap_unordered(_work, jobs), 1):
            n += c
            if j % 50 == 0:
                print(f"{j}/{len(jobs)} chunks, {n} images, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new images in {time.time() - t0:.0f}s -> {out_root()}")


if __name__ == "__main__":
    main()
