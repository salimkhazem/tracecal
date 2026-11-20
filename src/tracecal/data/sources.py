"""Uniform image sources for extraction. Items are (uint8 tensor 3xHxW at native resolution, label, id).

Source names:
  cifar10:test | cifar10:pool | cifar10c:<corruption>:<severity>        (same for cifar100 / cifar100c)
  imagenet:pool | imagenet:test10k | imagenetc:<corruption>:<severity> | imagenetcbar:<corruption>:<severity>
  imagenet_v2 | imagenet_sketch | imagenet_r | imagenet_a
  imagenetes:ref | imagenetes:pc:<l1|l5>:<1..27> | imagenetes:ae:<l1|l5>:<1..5>   (ImageNet-ES es-test, v5d)
CIFAR ids are positions in the official train (pool) or test (test, C) arrays; ImageNet ids are ILSVRC2012 val IDs
(clean and C) or record positions (natural sets).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from . import cifar as cf
from . import imagenet as inet
from .splits import cifar_holdout, imagenet_val_split


@dataclass
class Source:
    name: str
    family: str
    n_classes: int
    class_subset: list[int] | None
    dataset: Dataset
    in_memory: bool
    info: dict[str, Any]


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, ids: np.ndarray):
        assert x.ndim == 4 and x.shape[-1] == 3 and len(x) == len(y) == len(ids)
        self.x, self.y, self.ids = x, np.asarray(y, dtype=np.int64), np.asarray(ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return torch.from_numpy(np.ascontiguousarray(self.x[i].transpose(2, 0, 1))), int(self.y[i]), int(self.ids[i])

    def batches(self, batch_size: int):
        for s in range(0, len(self), batch_size):
            e = min(s + batch_size, len(self))
            xb = torch.from_numpy(np.ascontiguousarray(self.x[s:e].transpose(0, 3, 1, 2)))
            yield xb, torch.from_numpy(self.y[s:e]), torch.from_numpy(self.ids[s:e])


@lru_cache(maxsize=1)
def imagenet_val() -> inet.ImageNetValHF:
    return inet.ImageNetValHF()


@lru_cache(maxsize=1)
def imagenet_split_ids() -> dict[str, np.ndarray]:
    val = imagenet_val()
    pool, test, csub = imagenet_val_split(val.labels)
    return {"pool_pos": pool, "test_pos": test, "csub_pos": csub,
            "pool_ids": val.ids[pool], "test_ids": val.ids[test], "csub_ids": val.ids[csub]}


class ImageNetValDataset(Dataset):
    def __init__(self, positions: np.ndarray):
        self.pos = np.asarray(positions, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, i: int):
        val = imagenet_val()
        p = int(self.pos[i])
        arr = inet.clean_preprocess(val.image(p))
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))), int(val.labels[p]), int(val.ids[p])


class ImageNetShiftDataset(ImageNetValDataset):
    """E1: the clean ImageNet test subset under one exactly-invertible shift (see the cifar*rs sources)."""

    def __init__(self, positions: np.ndarray, shift: str):
        super().__init__(positions)
        self.shift = shift

    def __getitem__(self, i: int):
        from ..probes import ops

        x, y, vid = super().__getitem__(i)
        t = x.float()[None] / 255.0
        t = {"dark": lambda u: ops.brightness(u, -0.20), "lowcontrast": lambda u: ops.contrast(u, 1.0 / 1.5),
             "desat": lambda u: ops.saturation(u, 1.0 / 1.7)}[self.shift](t)
        return (t[0].clamp(0, 1) * 255.0).round().to(torch.uint8), y, vid


class ImageNetCDataset(Dataset):
    def __init__(self, corruption: str, severity: int, val_ids: np.ndarray):
        self.corruption, self.severity = corruption, int(severity)
        self.ids = np.asarray(val_ids, dtype=np.int64)
        w2i = {w: i for i, w in enumerate(inet.imagenet_wnids())}
        id2w = inet.imagenet_c_id2wnid()
        self.labels = np.asarray([w2i[id2w[int(v)]] for v in self.ids], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        from PIL import Image

        path = inet.imagenet_c_path(self.corruption, self.severity, int(self.ids[i]))
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if arr.shape != (224, 224, 3):
            raise RuntimeError(f"unexpected ImageNet-C image shape {arr.shape} at {path}")
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))), int(self.labels[i]), int(self.ids[i])


class ImageNetCBarDataset(ImageNetCDataset):
    """ImageNet-C-bar on the 10k test subset (scripts/make_imagenet_c_bar_subset.py; pre-registration v5b)."""

    def __getitem__(self, i: int):
        import os
        from pathlib import Path

        from PIL import Image

        path = Path(os.environ["DATA_ROOT"]) / "imagenet_c_bar" / self.corruption / str(self.severity) / \
            f"{int(self.ids[i])}.JPEG"
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if arr.shape != (224, 224, 3):
            raise RuntimeError(f"unexpected ImageNet-C-bar image shape {arr.shape} at {path}")
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))), int(self.labels[i]), int(self.ids[i])


def imagenet_es_root(split: str = "es-test"):
    import os
    from pathlib import Path

    return Path(os.environ["DATA_ROOT"]) / "imagenet_es" / "ImageNet-ES" / split


ES_REF = {"es-test": "sampled_tin_no_resize2", "es-val": "sampled_tin_no_resize"}


def _es_vid(f) -> int:
    return int(f.stem.split("_")[-1])          # ILSVRC2012_val_00030992 -> 30992


@lru_cache(maxsize=2)
def imagenet_es_index(split: str = "es-test") -> dict[str, Any]:
    """ImageNet-ES es-test (pre-registration v5d) or es-val (development split of v6): the 1,000 source images
    (ILSVRC2012 val ids, sorted) and the 200 Tiny-ImageNet classes as indices in the ImageNet-1k order."""
    w2i = {w: i for i, w in enumerate(inet.imagenet_wnids())}
    files = sorted((imagenet_es_root(split) / ES_REF[split]).glob("*/*.JPEG"), key=_es_vid)
    if len(files) != 1000:
        raise RuntimeError(f"expected 1000 es-test source images, found {len(files)}")
    return {"source_ids": np.asarray([_es_vid(f) for f in files], dtype=np.int64),
            "class_subset": sorted({w2i[f.parent.name] for f in files})}


class ImageNetESDataset(Dataset):
    """One ImageNet-ES es-test condition: 'ref' (source images), 'pc:<l1|l5>:<1..27>' (manual ISO / shutter / aperture
    setting, light on = l1 / off = l5) or 'ae:<l1|l5>:<1..5>' (auto-exposure shot). Items sorted by val id; images go
    through the same Resize(256) + CenterCrop(224) as the natural-shift sets; labels in the ImageNet-1k order."""

    def __init__(self, cond: str, split: str = "es-test"):
        root = imagenet_es_root(split)
        if cond == "ref":
            d = root / ES_REF[split]
        else:
            mode, light, k = cond.split(":")
            d = root / {"pc": "param_control", "ae": "auto_exposure"}[mode] / light / f"param_{int(k)}"
        w2i = {w: i for i, w in enumerate(inet.imagenet_wnids())}
        self.files = sorted(d.glob("*/*.JPEG"), key=_es_vid)
        if not self.files:
            raise FileNotFoundError(d)
        self.ids = np.asarray([_es_vid(f) for f in self.files], dtype=np.int64)
        self.labels = np.asarray([w2i[f.parent.name] for f in self.files], dtype=np.int64)
        self.class_subset = imagenet_es_index(split)["class_subset"]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        from PIL import Image

        arr = inet.clean_preprocess(Image.open(self.files[i]))
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))), int(self.labels[i]), int(self.ids[i])


class NaturalDataset(Dataset):
    def __init__(self, name: str):
        idx = inet.natural_index(name)
        self.kind, self.recs = idx["kind"], idx["records"]
        self.labels = np.asarray(idx["labels"], dtype=np.int64)
        self.class_subset = idx["class_subset"]

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int):
        arr = inet.clean_preprocess(inet.read_record(self.kind, self.recs[i]))
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))), int(self.labels[i]), int(i)


def get_source(name: str) -> Source:
    parts = name.split(":")
    head = parts[0]
    if head in ("cifar10", "cifar100"):
        nc = cf.N_CLASSES[head]
        if parts[1] == "test":
            x, y = cf.load_cifar(head, "test")
            ds = ArrayDataset(x, y, np.arange(len(y)))
        elif parts[1] == "pool":
            x, y = cf.load_cifar(head, "train")
            _, pool = cifar_holdout(y)
            ds = ArrayDataset(x[pool], y[pool], pool)
        else:
            raise ValueError(name)
        return Source(name, "cifar", nc, None, ds, True, {})
    if head in ("cifar10rs", "cifar100rs"):
        # E1 (internal review 2026-10-23): shifts that one restorer undoes up to clipping and uint8 rounding
        #   dark: brightness -0.20 (undone by bright_up +0.20); lowcontrast: contrast x 1/1.5 (contrast_up x 1.5);
        #   desat: saturation x 1/1.7 (sat_up x 1.7)
        from ..probes import ops

        base = head[:-2]
        x, y = cf.load_cifar(base, "test")
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
        shift = parts[1]
        if shift == "dark":
            t = ops.brightness(t, -0.20)
        elif shift == "lowcontrast":
            t = ops.contrast(t, 1.0 / 1.5)
        elif shift == "desat":
            t = ops.saturation(t, 1.0 / 1.7)
        else:
            raise ValueError(name)
        xs = (t.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        return Source(name, "cifar", cf.N_CLASSES[base], None, ArrayDataset(xs, y, np.arange(len(y))), True, {})
    if head in ("cifar10c", "cifar100c"):
        base = head[:-1]
        corruption, severity = parts[1], int(parts[2])
        x, y = cf.load_cifar_c(base, corruption, severity)
        ds = ArrayDataset(x, y, np.arange(len(y)))
        return Source(name, "cifar", cf.N_CLASSES[base], None, ds, True, {})
    if head == "imagenet":
        ids = imagenet_split_ids()
        key = {"pool": "pool_pos", "test10k": "csub_pos", "test": "test_pos"}[parts[1]]
        return Source(name, "imagenet", 1000, None, ImageNetValDataset(ids[key]), False,
                      {"val": imagenet_val().describe()})
    if head == "imagenetc":
        ids = imagenet_split_ids()["csub_ids"]
        return Source(name, "imagenet", 1000, None, ImageNetCDataset(parts[1], int(parts[2]), ids), False, {})
    if head == "imagenetrs":
        ids = imagenet_split_ids()
        return Source(name, "imagenet", 1000, None, ImageNetShiftDataset(ids["csub_pos"], parts[1]), False, {})
    if head == "imagenetcbar":
        ids = imagenet_split_ids()["csub_ids"]
        return Source(name, "imagenet", 1000, None, ImageNetCBarDataset(parts[1], int(parts[2]), ids), False, {})
    if head in ("imagenetes", "imagenetesval"):
        split = "es-test" if head == "imagenetes" else "es-val"
        ds = ImageNetESDataset(":".join(parts[1:]), split)
        return Source(name, "imagenet", 1000, ds.class_subset, ds, False, {"kind": "imagenet_es", "split": split})
    if head in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
        ds = NaturalDataset(head)
        return Source(name, "imagenet", 1000, ds.class_subset, ds, False, {"kind": ds.kind})
    raise ValueError(f"unknown source {name!r}")
