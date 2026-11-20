"""Orbit-augmented calibration set P'(p0, allowed families): exactly one pool version per image (i.i.d.)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import families, protocol
from ..shifts.programs import base_seed, program, program_families
from ..utils.seed import rng_for
from .bundles import Bundle, concat


@lru_cache(maxsize=200000)
def _composed(iid: int, res_family: str) -> tuple[str, ...]:
    return tuple(program_families(program(int(iid), len(families()) + 1, res_family)))


@lru_cache(maxsize=200000)
def _uniforms(iid: int) -> tuple[float, float, float]:
    u = rng_for(base_seed(), "pprime", int(iid)).random(3)
    return float(u[0]), float(u[1]), float(u[2])


def pprime_versions(image_ids: np.ndarray, p0: float, allowed: list[str], res_family: str) -> np.ndarray:
    """Version (0 = clean, 1..7 single family, 8 composed) contributed by each image to P'.

    The uniforms depend only on the image id, so P'(p0) is nested across p0 (common random numbers)."""
    fams = families()
    q_comp = float(protocol()["programs"]["q_comp"])
    allowed_set = set(allowed)
    ordered = [f for f in fams if f in allowed_set]
    if not ordered:
        raise ValueError("no allowed families")
    out = np.empty(len(image_ids), dtype=np.int64)
    for i, iid in enumerate(np.asarray(image_ids)):
        u0, u1, u2 = _uniforms(int(iid))
        if u0 < p0:
            out[i] = 0
        elif u1 < q_comp and all(f in allowed_set for f in _composed(int(iid), res_family)):
            out[i] = len(fams) + 1
        else:
            f = ordered[min(int(u2 * len(ordered)), len(ordered) - 1)]
            out[i] = fams.index(f) + 1
    return out


def assemble(pool: list[Bundle], positions: np.ndarray, versions: np.ndarray) -> Bundle:
    """Rows ``positions`` of the pool, each taken from its own version's extraction, in ``positions`` order."""
    positions = np.asarray(positions, dtype=np.int64)
    parts, order = [], []
    for v in np.unique(versions):
        m = np.flatnonzero(versions == v)
        parts.append(pool[int(v)].take(positions[m]))
        order.append(m)
    b = concat(parts)
    inv = np.argsort(np.concatenate(order), kind="stable")
    out = b.take(inv)
    out.meta = {"versions": versions}
    return out


def assemble_view_logits(pool: list[Bundle], positions: np.ndarray, versions: np.ndarray) -> np.ndarray:
    """(n, V+1, C') view logits for an assembled P' (rows in ``positions`` order)."""
    positions = np.asarray(positions, dtype=np.int64)
    out = None
    for v in np.unique(versions):
        m = np.flatnonzero(versions == v)
        vl = pool[int(v)].take(positions[m]).all_view_logits()
        if out is None:
            out = np.empty((len(positions),) + vl.shape[1:], dtype=np.float32)
        out[m] = vl
    assert out is not None
    return out


def version_labels(versions: np.ndarray) -> np.ndarray:
    """Shift-type label used by the white-box 'shiftpred' baseline (0 clean, 1..7 family, 8 composed)."""
    return np.asarray(versions, dtype=np.int64)


def severity(step: dict) -> float:
    op = step["op"]
    if op in ("noise", "blur"):
        return float(step["sigma"])
    if op == "brightness":
        return abs(float(step["delta"]))
    if op == "contrast":
        return 1.0 - float(step["factor"])
    if op == "jpeg":
        return 100.0 - float(step["quality"])
    if op == "downup":
        return 1.0 - float(step["factor"])
    if op == "saturation":
        return abs(float(step["factor"]) - 1.0)
    raise ValueError(op)


def single_family_severity(image_ids: np.ndarray, version: int, res_family: str) -> np.ndarray:
    return np.array([severity(program(int(i), version, res_family)[0]) for i in image_ids])
