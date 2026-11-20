"""G0: ImageNet val (HF cache) <-> ImageNet-C ID alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..data.imagenet import clean_preprocess, imagenet_c_id2wnid, imagenet_c_path, imagenet_wnids
from ..data.sources import imagenet_val


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def check(n: int = 500, seed: int = 0) -> dict[str, Any]:
    val = imagenet_val()
    w2i = {w: i for i, w in enumerate(imagenet_wnids())}
    id2w = imagenet_c_id2wnid()
    c_labels = np.array([w2i[id2w[int(i)]] for i in val.ids])
    agree = float((c_labels == val.labels).mean())
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(val), size=n, replace=False)
    other = rng.permutation(pos)
    matched, mismatched = [], []
    for p, q in zip(pos, other):
        clean = clean_preprocess(val.image(int(p)))
        c_img = np.asarray(Image.open(imagenet_c_path("brightness", 1, int(val.ids[p]))).convert("RGB"))
        matched.append(_corr(clean, c_img))
        if q != p:
            c_other = np.asarray(Image.open(imagenet_c_path("brightness", 1, int(val.ids[q]))).convert("RGB"))
            mismatched.append(_corr(clean, c_other))
    matched_arr, mism_arr = np.array(matched), np.array(mismatched)
    ok = (agree == 1.0 and matched_arr.min() > 0.95 and np.median(mism_arr) < 0.5 and mism_arr.max() < 0.95)
    return {"val": val.describe(), "label_agreement": agree, "matched_min": float(matched_arr.min()),
            "matched_median": float(np.median(matched_arr)), "mismatched_median": float(np.median(mism_arr)),
            "mismatched_max": float(mism_arr.max()), "n": n, "pass": bool(ok)}
