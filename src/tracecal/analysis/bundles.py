"""Load extracted arrays and derive per-example quantities used by every method.

A Bundle is a view on one extraction directory: ``rows`` are positions in the on-disk arrays, so splits
(``take``) and class restriction (``restrict``) compose without copying the large view-logit arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..fsio import features_dir
from ..utils.io import load_array, read_json


# Calibration-pool view logits are read many times per process (every split and re-split takes rows from the same
# nine pool versions). For directories loaded with cache=True we keep the full fp16 array in RAM after the first read;
# test conditions (cache=False) are still read from disk on demand. Same values, fewer reads.
_CACHEABLE_DIRS: set[str] = set()
_VIEWS_RAM: dict[str, np.ndarray] = {}
_BASE_RAM: dict[str, np.ndarray] = {}


def _view_array(view_dir: str):
    v = load_array(f"{view_dir}/view_logits.npy")
    if view_dir not in _CACHEABLE_DIRS:
        return v
    if view_dir not in _VIEWS_RAM:
        _VIEWS_RAM[view_dir] = np.asarray(v)
    return _VIEWS_RAM[view_dir]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@dataclass
class Bundle:
    logits: np.ndarray           # (N, C') base logits (restricted to `subset` if set)
    labels: np.ndarray           # (N,) labels in the (possibly restricted) label space
    R: np.ndarray                # (N, K) signed log-confidence change per probe (computed in the full label space)
    KL: np.ndarray               # (N, K)
    ids: np.ndarray              # (N,)
    rows: np.ndarray             # (N,) positions in the on-disk arrays
    feats: np.ndarray | None
    view_dir: str
    subset: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    _probs: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def probs(self) -> np.ndarray:
        if self._probs is None:
            self._probs = softmax(self.logits)
        return self._probs

    @property
    def conf(self) -> np.ndarray:
        return self.probs.max(1)

    @property
    def entropy(self) -> np.ndarray:
        p = np.clip(self.probs, 1e-12, 1.0)
        return -(p * np.log(p)).sum(1)

    @property
    def s(self) -> np.ndarray:
        return self.KL.mean(1)

    def take(self, idx: np.ndarray) -> Bundle:
        idx = np.asarray(idx, dtype=np.int64)
        return Bundle(self.logits[idx], self.labels[idx], self.R[idx], self.KL[idx], self.ids[idx], self.rows[idx],
                      None if self.feats is None else self.feats[idx], self.view_dir, self.subset, self.meta,
                      None if self._probs is None else self._probs[idx])

    def restrict(self, subset: list[int] | np.ndarray | None) -> Bundle:
        """Keep examples whose label is in ``subset``; restrict + renormalise to those classes."""
        if subset is None:
            return self
        if self.subset is not None:
            raise ValueError("bundle already restricted")
        sub = np.asarray(subset, dtype=np.int64)
        pos = -np.ones(int(max(sub.max(), self.labels.max())) + 1, dtype=np.int64)
        pos[sub] = np.arange(len(sub))
        keep = np.flatnonzero(pos[self.labels] >= 0)
        b = self.take(keep)
        return Bundle(b.logits[:, sub], pos[b.labels], b.R, b.KL, b.ids, b.rows, b.feats, b.view_dir, sub, b.meta)

    def all_view_logits(self) -> np.ndarray:
        """(N, V+1, C') float32 logits for [base, views...] in this bundle's rows / label space (memoised)."""
        cached = self.meta.get("_views_cache") if isinstance(self.meta, dict) else None
        if cached is not None and cached[0] is self.rows:
            return cached[1]
        out = self._load_views()
        if isinstance(self.meta, dict):
            self.meta = dict(self.meta, _views_cache=(self.rows, out))
        return out

    def _load_views(self) -> np.ndarray:
        v = _view_array(self.view_dir)
        order = np.argsort(self.rows, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        vv = np.asarray(v[self.rows[order]], dtype=np.float32)[inv]
        out = np.concatenate([self.logits_full_base()[:, None, :], vv], axis=1)
        if self.subset is not None:
            out = out[:, :, self.subset]
        return out

    def logits_full_base(self) -> np.ndarray:
        b = load_array(f"{self.view_dir}/base_logits.npy")
        if self.view_dir in _CACHEABLE_DIRS:
            b = _BASE_RAM.setdefault(self.view_dir, np.asarray(b))
        order = np.argsort(self.rows, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return np.asarray(b[self.rows[order]], dtype=np.float32)[inv]


def _load_raw_uncached(model_tag: str, source: str, version: int, bank: str, with_feats: bool):
    d = features_dir(model_tag, source, version, bank)
    if not (d / ".ok").exists():
        raise FileNotFoundError(f"missing extraction: {d}")
    return {
        "logits": np.asarray(load_array(d / "base_logits.npy"), dtype=np.float32),
        "labels": np.asarray(load_array(d / "labels.npy"), dtype=np.int64),
        "R": np.asarray(load_array(d / "R.npy"), dtype=np.float32),
        "KL": np.asarray(load_array(d / "KL.npy"), dtype=np.float32),
        "ids": np.asarray(load_array(d / "ids.npy"), dtype=np.int64),
        "feats": np.asarray(load_array(d / "feats.npy"), dtype=np.float32) if with_feats else None,
        "dir": str(d),
        "meta": read_json(d / "meta.json"),
    }


_load_raw_cached = lru_cache(maxsize=48)(_load_raw_uncached)


def load_bundle(model_tag: str, source: str, version: int = 0, bank: str = "default",
                with_feats: bool = False, cache: bool = True) -> Bundle:
    """``cache=True`` for calibration-pool data (reused across methods); test conditions use ``cache=False``."""
    raw = (_load_raw_cached if cache else _load_raw_uncached)(model_tag, source, version, bank, with_feats)
    if cache:
        _CACHEABLE_DIRS.add(raw["dir"])
    n = len(raw["labels"])
    return Bundle(raw["logits"], raw["labels"], raw["R"], raw["KL"], raw["ids"], np.arange(n, dtype=np.int64),
                  raw["feats"], raw["dir"], None, raw["meta"])


def concat(bundles: list[Bundle]) -> Bundle:
    """Row-wise concatenation (used to assemble P' from different pool versions; view logits not supported)."""
    def cat(name):
        vals = [getattr(b, name) for b in bundles]
        return None if any(v is None for v in vals) else np.concatenate(vals)

    return Bundle(cat("logits"), cat("labels"), cat("R"), cat("KL"), cat("ids"), cat("rows"), cat("feats"),
                  "<concat>", bundles[0].subset, {})
