from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch RNGs. Deterministic algorithms are requested where torch supports them."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ImportError:  # analysis-only environments
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def derive_seed(base: int, *keys: object) -> int:
    """Stable per-item seed derived from a base seed and arbitrary keys (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256(("|".join([str(base), *map(str, keys)])).encode()).digest()
    return int.from_bytes(h[:4], "little")


def rng_for(base: int, *keys: object) -> np.random.Generator:
    """numpy Generator seeded by ``derive_seed(base, *keys)``."""
    return np.random.default_rng(derive_seed(base, *keys))
