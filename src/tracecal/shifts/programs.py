"""Synthetic shift programs (Pi) for the orbit-augmented calibration set P'.

Every calibration-pool image is materialised in ``n_versions = len(families) + 2`` versions:
  v0 = clean, v1..v7 = one family each (random severity), v8 = two distinct random families composed.
Parameters are a deterministic function of (base_seed, image_id, version) so any machine reproduces them.
P' itself (which version each image contributes) is chosen later, at analysis time (``analysis/pprime.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import families, program_ranges, protocol
from ..utils.seed import derive_seed, rng_for


def n_versions() -> int:
    return len(families()) + 2


def base_seed() -> int:
    return int(protocol()["programs"]["base_seed"])


def sample_params(family: str, ranges: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    r = ranges[family]
    if family == "noise":
        return {"op": "noise", "sigma": float(rng.uniform(*r["sigma"]))}
    if family == "blur":
        return {"op": "blur", "sigma": float(rng.uniform(*r["sigma"]))}
    if family == "brightness":
        a = float(rng.uniform(*r["abs_delta"]))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return {"op": "brightness", "delta": sign * a}
    if family == "contrast":
        return {"op": "contrast", "factor": float(rng.uniform(*r["factor"]))}
    if family == "jpeg":
        lo, hi = r["quality"]
        return {"op": "jpeg", "quality": int(rng.integers(int(lo), int(hi) + 1))}
    if family == "downup":
        return {"op": "downup", "factor": float(rng.uniform(*r["factor"]))}
    if family == "saturation":
        lo, hi = r["low"] if rng.random() < 0.5 else r["high"]
        return {"op": "saturation", "factor": float(rng.uniform(lo, hi))}
    raise ValueError(f"unknown family {family!r}")


def program(image_id: int, version: int, res_family: str) -> list[dict[str, Any]]:
    """The list of op specs applied to image ``image_id`` for pool version ``version``."""
    fams = families()
    ranges = program_ranges(res_family)
    if version == 0:
        return []
    rng = rng_for(base_seed(), "program", int(image_id), int(version))
    if 1 <= version <= len(fams):
        return [sample_params(fams[version - 1], ranges, rng)]
    if version == len(fams) + 1:
        i, j = rng.choice(len(fams), size=2, replace=False)
        return [sample_params(fams[int(i)], ranges, rng), sample_params(fams[int(j)], ranges, rng)]
    raise ValueError(f"version must be in [0, {len(fams) + 1}]")


def program_families(prog: list[dict[str, Any]]) -> list[str]:
    return [step["op"] for step in prog]


def apply_programs(x, image_ids, version: int, res_family: str):
    """Apply each image's program (per-image loop; P' extraction is a small fraction of compute)."""
    import torch

    from ..probes.ops import apply_op, noise_field

    if version == 0:
        return x.clone(), [[] for _ in range(x.shape[0])]
    h, w = x.shape[-2:]
    out = torch.empty_like(x)
    progs = []
    for i, iid in enumerate(image_ids):
        prog = program(int(iid), version, res_family)
        y = x[i : i + 1]
        for step_idx, step in enumerate(prog):
            noise = None
            if step["op"] == "noise":
                noise = noise_field((3, h, w), derive_seed(base_seed(), "pnoise", int(iid), version, step_idx))
            y = apply_op(y, step, noise=noise)
        out[i] = y[0]
        progs.append(prog)
    return out, progs
