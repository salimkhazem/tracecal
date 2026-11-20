"""Small, dependency-free artifact I/O helpers with checksums."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_array(path: str | Path, arr: np.ndarray) -> str:
    """Save .npy atomically and return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)
    return sha256_file(path)


def load_array(path: str | Path, mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def take_rows(arr: np.ndarray, idx: np.ndarray, dtype=None, chunk_rows: int = 1 << 18) -> np.ndarray:
    """Rows ``arr[idx]`` read SEQUENTIALLY in large contiguous chunks (the order of ``idx`` is preserved).

    Random row access into a memmap triggers kernel read-ahead per row (observed ~100x read amplification on a
    loaded host). Scanning the spanned range sequentially and keeping the requested rows is far cheaper whenever
    the requested rows are not extremely sparse.
    """
    idx = np.asarray(idx, dtype=np.int64)
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    out = np.empty((idx.size,) + tuple(arr.shape[1:]), dtype=dtype or arr.dtype)
    if idx.size == 0:
        return out
    lo, hi = int(sidx[0]), int(sidx[-1]) + 1
    pos = 0
    for c0 in range(lo, hi, chunk_rows):
        c1 = min(c0 + chunk_rows, hi)
        j = np.searchsorted(sidx, c1, side="left")
        if j > pos:
            block = np.asarray(arr[c0:c1])
            out[order[pos:j]] = block[sidx[pos:j] - c0]
            pos = j
    return out


class RunLock:
    """Exclusive, non-blocking per-output lock (``<dir>.lock`` next to the output directory).

    ``with RunLock(out) as got: if not got: return`` -- a second process asked to produce the same output skips
    instead of writing concurrently into the same files."""

    def __init__(self, out: str | Path):
        self.path = Path(out).with_name(Path(out).name + ".lock")
        self.f = None

    def __enter__(self) -> bool:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            self.f.close()
            self.f = None
            return False

    def __exit__(self, *exc) -> None:
        if self.f is not None:
            self.f.close()
