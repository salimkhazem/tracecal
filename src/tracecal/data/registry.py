"""Dataset locations (all under SHARED_DATASETS_ROOT on ws) and a verification report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import get_paths

CIFAR_C_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
    "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate",
    "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
]


def shared_root() -> Path:
    root = get_paths().shared_datasets_root
    if root is None:
        raise RuntimeError("SHARED_DATASETS_ROOT is not set (source configs/ws/env.sh)")
    return root


def roots() -> dict[str, Path]:
    s = shared_root()
    return {
        "cifar10": s / "cifar-10-batches-py",
        "cifar100": s / "cifar-100-python",
        "cifar10c": s / "CIFAR-10-C",
        "cifar100c": s / "CIFAR-100-C",
        "imagenet_hf": s / "Imagenet",
        "imagenet_c": s / "ImageNet-C",
        "imagenet_r": s / "imagenet-r",
        "imagenet_a": s / "imagenet-a",
        "imagenet_v2": s / "clip_benchmark" / "imagenetv2",
        "imagenet_sketch": s / "clip_benchmark" / "imagenet_sketch",
    }


def verify() -> dict[str, Any]:
    """Cheap existence/shape checks. Heavy checks (ID mapping, label agreement) live in ``tracecal data idmap``."""
    import numpy as np

    out: dict[str, Any] = {}
    r = roots()
    for name in ("cifar10", "cifar100"):
        out[name] = {"exists": r[name].is_dir(), "files": sorted(p.name for p in r[name].glob("*"))[:10]}
    for name in ("cifar10c", "cifar100c"):
        info: dict[str, Any] = {"exists": r[name].is_dir()}
        if r[name].is_dir():
            missing = [c for c in CIFAR_C_CORRUPTIONS if not (r[name] / f"{c}.npy").exists()]
            info["missing"] = missing
            lab = r[name] / "labels.npy"
            info["labels_shape"] = list(np.load(lab, mmap_mode="r").shape) if lab.exists() else None
            if not missing:
                info["shape_gaussian_noise"] = list(np.load(r[name] / "gaussian_noise.npy", mmap_mode="r").shape)
        out[name] = info
    imc = r["imagenet_c"]
    out["imagenet_c"] = {
        "exists": imc.is_dir(),
        "corruptions": sorted(p.name for p in imc.iterdir() if p.is_dir()) if imc.is_dir() else [],
    }
    for name in ("imagenet_r", "imagenet_a"):
        p = r[name]
        out[name] = {"exists": p.is_dir(), "n_classes": len([d for d in p.iterdir() if d.is_dir()]) if p.is_dir() else 0}
    for name in ("imagenet_v2", "imagenet_sketch", "imagenet_hf"):
        p = r[name]
        listing = sorted(str(q.relative_to(p)) for q in p.rglob("*") if q.is_file())[:20] if p.is_dir() else []
        out[name] = {"exists": p.is_dir(), "first_files": listing}
    return out
