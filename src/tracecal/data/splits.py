"""Seeded, disjoint splits. All functions return sorted int64 index arrays."""

from __future__ import annotations

import numpy as np

from ..config import protocol


def stratified_subset(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` indices with exactly n / n_classes per class (requires divisibility and enough items per class)."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    per, rem = divmod(n, classes.size)
    if rem:
        raise ValueError(f"n={n} not divisible by {classes.size} classes")
    rng = np.random.default_rng(seed)
    out = []
    for c in classes:
        cand = np.flatnonzero(labels == c)
        if cand.size < per:
            raise ValueError(f"class {c} has {cand.size} < {per} items")
        out.append(rng.choice(cand, size=per, replace=False))
    return np.sort(np.concatenate(out)).astype(np.int64)


def complement(n_total: int, idx: np.ndarray) -> np.ndarray:
    mask = np.ones(n_total, dtype=bool)
    mask[idx] = False
    return np.flatnonzero(mask).astype(np.int64)


def cifar_holdout(train_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx 40k, pool_idx 10k) of the 50k CIFAR training images; identical for every model seed."""
    cfg = protocol()["splits"]["cifar"]
    pool = stratified_subset(train_labels, int(cfg["holdout_from_train"]), int(cfg["holdout_seed"]))
    return complement(len(train_labels), pool), pool


def imagenet_val_split(val_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (into the val ordering) of (pool 25k, test 25k, C-subsample 10k within test)."""
    cfg = protocol()["splits"]["imagenet"]
    pool = stratified_subset(val_labels, int(cfg["val_pool"]), int(cfg["val_seed"]))
    test = complement(len(val_labels), pool)
    sub_rel = stratified_subset(np.asarray(val_labels)[test], int(cfg["c_subsample"]), int(cfg["val_seed"]) + 1)
    return pool, test, test[sub_rel]


def pool_resplit(n_pool: int, family: str, resplit: int, allow_scaled: bool = False) -> dict[str, np.ndarray]:
    """Disjoint fit / tune / cal positions within the calibration pool for re-split ``resplit``.
    ``allow_scaled`` (smoke tests only) scales the protocol sizes proportionally to a smaller pool."""
    sizes = protocol()["splits"][family]["pool"]
    a, b, c = int(sizes["fit"]), int(sizes["tune"]), int(sizes["cal"])
    if a + b + c != n_pool:
        if not allow_scaled:
            raise ValueError(f"pool sizes {a}+{b}+{c} != {n_pool}")
        tot = a + b + c
        a, b = int(n_pool * a / tot), int(n_pool * b / tot)
        c = n_pool - a - b
    perm = np.random.default_rng(1000 + int(resplit)).permutation(n_pool)
    return {"fit": np.sort(perm[:a]), "tune": np.sort(perm[a : a + b]), "cal": np.sort(perm[a + b :])}
