"""G0 numerics: is fp16 inference precise enough for the response profile R?

Decision rule (fixed before running): keep fp16 if, for EVERY probe, mean |R_fp16 - R_fp32| < 0.05 * std(R_fp32)
and the base top-1 agrees on >= 99.5% of images; otherwise extract in fp32.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


@torch.inference_mode()
def _profile(model_tag: str, x01: torch.Tensor, family: str, amp: str) -> tuple[np.ndarray, np.ndarray]:
    from ..models.zoo import load
    from ..probes.bank import ProbeBank

    os.environ["TRACECAL_AMP"] = amp
    model = load(model_tag, "cuda")
    pb = ProbeBank(family)
    out_r, out_y = [], []
    for s in range(0, len(x01), 128):
        x = x01[s : s + 128].cuda()
        l0, _ = model(x)
        lp0 = torch.log_softmax(l0, 1)
        y0 = l0.argmax(1, keepdim=True)
        r = []
        for j, (_, xv) in enumerate(pb.views(x)):
            if j >= len(pb.probe_names):
                break
            lv, _ = model(xv)
            r.append((torch.log_softmax(lv, 1).gather(1, y0) - lp0.gather(1, y0)).squeeze(1))
        out_r.append(torch.stack(r, 1).cpu().numpy())
        out_y.append(y0.squeeze(1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out_r), np.concatenate(out_y)


def compare(model_tag: str, source: str, n: int = 512) -> dict[str, Any]:
    from ..data.sources import get_source

    src = get_source(source)
    xs = torch.stack([src.dataset[i][0] for i in range(n)]).float() / 255.0
    prev = os.environ.get("TRACECAL_AMP")
    try:
        r32, y32 = _profile(model_tag, xs, src.family, "fp32")
        r16, y16 = _profile(model_tag, xs, src.family, "fp16")
    finally:
        if prev is None:
            os.environ.pop("TRACECAL_AMP", None)
        else:
            os.environ["TRACECAL_AMP"] = prev
    mad = np.abs(r16 - r32).mean(0)
    sd = r32.std(0) + 1e-12
    ok = bool(np.all(mad < 0.05 * sd) and (y16 == y32).mean() >= 0.995)
    return {"model": model_tag, "source": source, "n": n, "mean_abs_diff": mad.tolist(), "std_fp32": sd.tolist(),
            "ratio": (mad / sd).tolist(), "top1_agreement": float((y16 == y32).mean()), "keep_fp16": ok}
