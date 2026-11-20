"""G0: reproduce published ImageNet top-1 under each model's default eval transform."""

from __future__ import annotations

import csv
import io
import urllib.request
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import imagenet_model_spec, protocol
from ..data.sources import imagenet_val
from ..models.zoo import default_eval_transform, load
from ..paths import get_paths

TIMM_CSV = "https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/results-imagenet.csv"
OPENCLIP_CSV = "https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv"


class _ValWithTransform(Dataset):
    def __init__(self, tf):
        self.tf = tf

    def __len__(self) -> int:
        return len(imagenet_val())

    def __getitem__(self, i: int):
        val = imagenet_val()
        return self.tf(val.image(i).convert("RGB")), int(val.labels[i])


def _fetch(url: str, name: str) -> str:
    cache = get_paths().results_root / "index" / name
    if cache.exists():
        return cache.read_text()
    txt = urllib.request.urlopen(url, timeout=60).read().decode()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(txt)
    return txt


def published_top1(model_tag: str) -> dict[str, Any]:
    spec = imagenet_model_spec(model_tag[3:])
    if "robustbench" in spec:
        url = ("https://raw.githubusercontent.com/RobustBench/robustbench/master/model_info/imagenet/corruptions/"
               f"{spec['robustbench']}.json")
        return {"source": url, "rows": [{"img_size": 224, "top1": spec["published_top1"],
                                         "note": "RobustBench clean accuracy (5000-image val subset)"}]}
    if "timm" in spec:
        rows = list(csv.DictReader(io.StringIO(_fetch(TIMM_CSV, "timm_results_imagenet.csv"))))
        hits = [r for r in rows if r["model"] == spec["timm"]]
        return {"source": TIMM_CSV, "rows": hits}
    rows = list(csv.DictReader(io.StringIO(_fetch(OPENCLIP_CSV, "openclip_results.csv"))))
    name = spec["open_clip"].replace("-quickgelu", "")  # the results table lists OpenAI weights under the base name
    hits = [r for r in rows if r.get("name") == name and r.get("pretrained") == spec["pretrained"]]
    return {"source": OPENCLIP_CSV, "rows": [{k: r[k] for k in r if "imagenet" in k.lower() or k in ("name", "pretrained")}
                                             for r in hits]}


@torch.inference_mode()
def imagenet_default_accuracy(model_tag: str, batch_size: int = 128, workers: int = 16) -> dict[str, Any]:
    w = load(model_tag, "cuda")
    tf = default_eval_transform(model_tag)
    dl = DataLoader(_ValWithTransform(tf), batch_size=batch_size, num_workers=workers, pin_memory=True)
    correct = 0
    n = 0
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if w.kind == "timm":
                logits = w.net(x).float()
            else:
                f = torch.nn.functional.normalize(w.net.encode_image(x).float(), dim=-1)
                logits = f @ w.zs_weights
        correct += int((logits.argmax(1).cpu() == y).sum())
        n += len(y)
    acc = 100.0 * correct / n
    pub = published_top1(model_tag)
    ref = None
    # compare at the resolution actually evaluated (timm reports each model at its train and test sizes)
    eval_size = None
    for t in getattr(tf, "transforms", []):
        sz = getattr(t, "size", None)
        if t.__class__.__name__ == "CenterCrop" and sz is not None:
            eval_size = int(sz[0] if isinstance(sz, (tuple, list)) else sz)
    for r in pub["rows"]:
        if "img_size" in r and str(r["img_size"]) == str(eval_size or 224):
            ref = float(r["top1"])
        elif "ImageNet 1k" in r and r["ImageNet 1k"]:
            ref = 100.0 * float(r["ImageNet 1k"])
    out = {"model": model_tag, "top1": acc, "n": n, "eval_size": eval_size, "published": pub, "published_top1": ref,
           "abs_diff": None if ref is None else abs(acc - ref),
           "pass": None if ref is None else abs(acc - ref) <= 0.5}
    from ..utils.io import atomic_write_json

    atomic_write_json(get_paths().results_root / "logs" / f"g0_top1_{model_tag}.json", out)
    return out
