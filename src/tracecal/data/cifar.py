"""CIFAR-10/100 and CIFAR-10/100-C (Hendrycks & Dietterich) loaders. Images are returned as NHWC uint8."""

from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np

from .registry import CIFAR_C_CORRUPTIONS, roots

N_CLASSES = {"cifar10": 10, "cifar100": 100}


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


@lru_cache(maxsize=4)
def load_cifar(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = roots()[name]
    if name == "cifar10":
        files = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
        xs, ys = [], []
        for fn in files:
            d = _unpickle(root / fn)
            xs.append(d["data"])
            ys.extend(d["labels"])
    elif name == "cifar100":
        d = _unpickle(root / ("train" if split == "train" else "test"))
        xs, ys = [d["data"]], list(d["fine_labels"])
    else:
        raise ValueError(name)
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.asarray(ys, dtype=np.int64)
    expected = 50000 if split == "train" else 10000
    if x.shape[0] != expected:
        raise RuntimeError(f"{name}/{split}: expected {expected} images, found {x.shape[0]}")
    return np.ascontiguousarray(x), y


def load_cifar_c(name: str, corruption: str, severity: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows [(s-1)*10000, s*10000) of ``{corruption}.npy``; row i is the corrupted version of test image i."""
    if corruption not in CIFAR_C_CORRUPTIONS:
        raise ValueError(corruption)
    if not 1 <= severity <= 5:
        raise ValueError(severity)
    root = roots()[name + "c"]
    arr = np.load(root / f"{corruption}.npy", mmap_mode="r")
    lab = np.load(root / "labels.npy", mmap_mode="r")
    sl = slice((severity - 1) * 10000, severity * 10000)
    x = np.ascontiguousarray(arr[sl])
    y = np.asarray(lab[sl], dtype=np.int64)
    if x.shape != (10000, 32, 32, 3):
        raise RuntimeError(f"unexpected CIFAR-C shape {x.shape}")
    return x, y


def check_c_alignment(name: str) -> dict[str, object]:
    """CIFAR-C labels must equal the clean test labels for every severity block (row i <-> test image i)."""
    _, yt = load_cifar(name, "test")
    lab = np.load(roots()[name + "c"] / "labels.npy", mmap_mode="r")
    ok = [bool(np.array_equal(np.asarray(lab[(s - 1) * 10000 : s * 10000]), yt)) for s in range(1, 6)]
    return {"labels_equal_per_severity": ok, "all_ok": all(ok)}
