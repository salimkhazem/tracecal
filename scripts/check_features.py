#!/usr/bin/env python
"""Integrity check of every extracted feature directory: a directory marked done (.ok) must contain every array with
consistent shapes and finite logits. Prints problems; exit code 1 if any."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracecal.paths import get_paths  # noqa: E402

ARRAYS = ["base_logits", "view_logits", "R", "KL", "flip", "feats", "labels", "ids"]


def check(d: Path, deep: bool) -> list[str]:
    probs = []
    missing = [a for a in ARRAYS if not (d / f"{a}.npy").exists()]
    if missing:
        return [f"missing {missing}"]
    arr = {a: np.load(d / f"{a}.npy", mmap_mode="r") for a in ARRAYS}
    n = arr["labels"].shape[0]
    for a, x in arr.items():
        if x.shape[0] != n:
            probs.append(f"{a} has {x.shape[0]} rows, labels {n}")
    if arr["view_logits"].shape[2] != arr["base_logits"].shape[1]:
        probs.append("class dim mismatch")
    if deep:
        for a in ("base_logits", "view_logits"):
            x = np.asarray(arr[a][:: max(1, n // 512)], dtype=np.float32)
            if not np.isfinite(x).all():
                probs.append(f"{a} non-finite")
            if np.all(x == 0):
                probs.append(f"{a} all zero (sampled rows)")
        if len(np.unique(np.asarray(arr["ids"]))) != n:
            probs.append("duplicate ids")
    return probs


def main() -> int:
    deep = "--deep" in sys.argv
    root = get_paths().features_root
    bad = 0
    n_ok = 0
    for ok in sorted(root.glob("*/*/*/.ok")):
        d = ok.parent
        p = check(d, deep)
        n_ok += 1
        if p:
            bad += 1
            print(f"BAD {d.relative_to(root)}: {'; '.join(p)}")
    stray = sorted(str(p.relative_to(root)) for p in root.glob("*/*/*.tmp*"))
    print(f"checked {n_ok} done dirs, {bad} bad; stray tmp dirs: {stray}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
