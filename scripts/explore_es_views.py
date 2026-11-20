#!/usr/bin/env python
"""EXPLORATORY (post hoc, not pre-registered): why test-time views do not help on ImageNet-ES (v5d failed).
Top-1 accuracy of every single view, relative to the identity view, on the 64 captured ImageNet-ES conditions and on
ImageNet-C at severity 3 (the 19 corruptions), for the 10 fresh ImageNet models; logits restricted to the 200
ImageNet-ES classes in both cases (ImageNet-C images of other classes are dropped).
  uv run python scripts/explore_es_views.py      (on the GPU server; results/tables/explore_es_views.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracecal.analysis.bundles import load_bundle  # noqa: E402
from tracecal.analysis.run import test_conditions  # noqa: E402
from tracecal.cli import es_conditions  # noqa: E402
from tracecal.config import view_names  # noqa: E402
from tracecal.data.sources import imagenet_es_index  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402

MODELS = ["in_deit_s", "in_effnet_b0", "in_regnety16", "in_convnext_s", "in_vit_s16",
          "in_densenet121", "in_mnv3_l", "in_deit3_s", "in_maxvit_t", "in_mixer_b16"]


def view_acc(conds: list[str], tag: str, subset) -> np.ndarray:
    acc = []
    for src in conds:
        b = load_bundle(tag, src, 0, cache=False).restrict(subset)
        z = b.all_view_logits()                                   # (n, V+1, C)
        acc.append((z.argmax(2) == b.labels[:, None]).mean(0))
    return np.mean(acc, 0)


def main() -> None:
    subset = imagenet_es_index()["class_subset"]
    names = ["identity"] + view_names("imagenet")
    es = [src for cn, src in es_conditions() if cn != "reference"]
    inc = [src for cn, src in test_conditions("imagenet") if cn.endswith(":3")]
    out = {"views": names, "models": {}}
    for m in MODELS:
        a_es, a_c = view_acc(es, m, subset), view_acc(inc, m, subset)
        out["models"][m] = {"es": a_es.tolist(), "imagenet_c_sev3": a_c.tolist()}
        print(m, "ES  ", " ".join(f"{n}:{100 * (x - a_es[0]):+.1f}" for n, x in zip(names[1:], a_es[1:])), flush=True)
        print(m, "IN-C", " ".join(f"{n}:{100 * (x - a_c[0]):+.1f}" for n, x in zip(names[1:], a_c[1:])), flush=True)
    mean = {k: np.mean([np.array(v[k]) - v[k][0] for v in out["models"].values()], 0) for k in ("es", "imagenet_c_sev3")}
    out["mean_gain_vs_identity"] = {k: dict(zip(names, map(float, v))) for k, v in mean.items()}
    print("MEAN ES  ", {n: round(100 * x, 1) for n, x in zip(names, mean["es"])})
    print("MEAN IN-C", {n: round(100 * x, 1) for n, x in zip(names, mean["imagenet_c_sev3"])})
    p = get_paths().results_root / "tables" / "explore_es_views.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
