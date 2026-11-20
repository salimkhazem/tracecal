"""ImageNet val (HF arrow/parquet cache), ImageNet-C, and natural-shift sets (V2, Sketch, R, A).

Clean images are preprocessed exactly like the ImageNet-C generator: PIL bilinear Resize(256) -> CenterCrop(224).
ImageNet-C files are already 224 x 224 crops and are only decoded. Indices (file lists, tar offsets) are cached as
JSON under ``$RESULTS_ROOT/index`` so every run sees the same ordering.
"""

from __future__ import annotations

import io
import re
import tarfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..paths import get_paths
from ..utils.io import atomic_write_json, read_json
from .registry import roots

VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG", ".PNG"}


def index_dir() -> Path:
    d = get_paths().results_root / "index"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_preprocess(img: Image.Image) -> np.ndarray:
    """PIL bilinear Resize(256) (shorter side) + CenterCrop(224), as in the ImageNet-C generation script."""
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    img = img.convert("RGB")
    img = TF.resize(img, 256, interpolation=InterpolationMode.BILINEAR)
    img = TF.center_crop(img, 224)
    return np.asarray(img, dtype=np.uint8)


# ----------------------------------------------------------------------------------------------------------------
# class ordering
# ----------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def imagenet_wnids() -> list[str]:
    """The 1000 ImageNet-1k wnids in sorted order (= standard class index), read from ImageNet-C's folders."""
    d = roots()["imagenet_c"] / "brightness" / "1"
    w = sorted(p.name for p in d.iterdir() if p.is_dir())
    if len(w) != 1000:
        raise RuntimeError(f"expected 1000 wnid folders in {d}, found {len(w)}")
    return w


@lru_cache(maxsize=1)
def imagenet_c_id2wnid() -> dict[int, str]:
    cache = index_dir() / "imagenet_c_id2wnid.json"
    if cache.exists():
        return {int(k): v for k, v in read_json(cache).items()}
    d = roots()["imagenet_c"] / "brightness" / "1"
    m: dict[int, str] = {}
    for wd in sorted(d.iterdir()):
        if not wd.is_dir():
            continue
        for f in wd.iterdir():
            mm = VAL_RE.search(f.name)
            if mm:
                m[int(mm.group(1))] = wd.name
    if len(m) != 50000:
        raise RuntimeError(f"ImageNet-C brightness/1 has {len(m)} images, expected 50000")
    atomic_write_json(cache, {str(k): v for k, v in sorted(m.items())})
    return m


# ----------------------------------------------------------------------------------------------------------------
# ImageNet val from the HF datasets cache
# ----------------------------------------------------------------------------------------------------------------
def _read_arrow(path: Path):
    import pyarrow as pa

    mm = pa.memory_map(str(path), "r")
    try:
        table = pa.ipc.open_stream(mm).read_all()
    except pa.ArrowInvalid:
        table = pa.ipc.open_file(mm).read_all()
    return table, mm


class ImageNetValHF:
    """The 50k ImageNet val images from the HF cache. ``ids`` are ILSVRC2012 val IDs (1..50000)."""

    def __init__(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        root = roots()["imagenet_hf"]
        groups: dict[Path, list[Path]] = defaultdict(list)
        for p in root.rglob("*"):
            if p.suffix in (".arrow", ".parquet") and "validation" in p.name:
                groups[p.parent].append(p)
        chosen = None
        self._mmaps: list[Any] = []
        for parent in sorted(groups):
            files = sorted(groups[parent])
            tables = []
            mms = []
            for f in files:
                if f.suffix == ".arrow":
                    t, mm = _read_arrow(f)
                    mms.append(mm)
                else:
                    t = pq.read_table(f, memory_map=True)
                tables.append(t)
            table = pa.concat_tables(tables)
            if table.num_rows == 50000:
                chosen = (parent, files, table)
                self._mmaps = mms
                break
        if chosen is None:
            raise RuntimeError(f"no ImageNet validation cache with 50000 rows under {root}: {dict(groups)}")
        self.source_dir, self.files, self.table = chosen
        self._img = self.table.column("image")
        self.labels = np.asarray(self.table.column("label").to_numpy(), dtype=np.int64)
        if pa.types.is_struct(self._img.type) and self._img.type.get_field_index("path") >= 0:
            # read only the 'path' child column (does not touch the image bytes)
            paths = pa.chunked_array([ch.field("path") for ch in self._img.chunks]).to_pylist()
        else:
            paths = [None] * len(self)
        parsed = [VAL_RE.search(p) if p else None for p in paths]
        if all(parsed):
            self.ids = np.asarray([int(m.group(1)) for m in parsed], dtype=np.int64)  # type: ignore[union-attr]
            self.id_source = "path"
        else:
            self.ids = np.arange(1, len(self) + 1, dtype=np.int64)
            self.id_source = "order"

    def __len__(self) -> int:
        return self.table.num_rows

    def _raw(self, i: int) -> Any:
        return self._img[int(i)].as_py()

    def _path_of(self, i: int) -> str | None:
        rec = self._raw(i)
        return rec.get("path") if isinstance(rec, dict) else None

    def image(self, i: int) -> Image.Image:
        rec = self._raw(i)
        data = rec["bytes"] if isinstance(rec, dict) else rec
        return Image.open(io.BytesIO(data))

    def describe(self) -> dict[str, Any]:
        return {"source_dir": str(self.source_dir), "n_files": len(self.files), "id_source": self.id_source,
                "first_paths": [self._path_of(i) for i in range(3)], "first_labels": self.labels[:5].tolist()}


def imagenet_c_path(corruption: str, severity: int, val_id: int) -> Path:
    wnid = imagenet_c_id2wnid()[int(val_id)]
    return roots()["imagenet_c"] / corruption / str(int(severity)) / wnid / f"ILSVRC2012_val_{int(val_id):08d}.JPEG"


# ----------------------------------------------------------------------------------------------------------------
# natural shifts
# ----------------------------------------------------------------------------------------------------------------
def _folder_records(root: Path) -> tuple[list[Any], list[int]]:
    wnids = imagenet_wnids()
    w2i = {w: i for i, w in enumerate(wnids)}
    recs, labels = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in w2i:
            lab = w2i[d.name]
        elif d.name.isdigit():
            lab = int(d.name)
        else:
            continue
        for f in sorted(d.iterdir()):
            if f.suffix in IMG_EXT:
                recs.append(str(f))
                labels.append(lab)
    return recs, labels


def _tar_records(root: Path) -> tuple[list[Any], list[int]]:
    tars = sorted(root.rglob("*.tar"))
    recs, labels = [], []
    for tp in tars:
        by_key: dict[str, dict[str, Any]] = defaultdict(dict)
        with tarfile.open(tp) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # webdataset convention: key = path up to the first dot of the basename, ext = the rest
                head, _, base = m.name.rpartition("/")
                if "." not in base:
                    continue
                stem, rest = base.split(".", 1)
                key = f"{head}/{stem}" if head else stem
                ext = "." + rest
                if ext == ".cls":
                    by_key[key]["cls"] = int(tf.extractfile(m).read().decode().strip())  # type: ignore[union-attr]
                elif ext in IMG_EXT:
                    by_key[key]["img"] = (str(tp), int(m.offset_data), int(m.size))
        for key in sorted(by_key):
            r = by_key[key]
            if "img" in r and "cls" in r:
                recs.append(list(r["img"]))
                labels.append(int(r["cls"]))
    return recs, labels


def natural_index(name: str) -> dict[str, Any]:
    """Records + labels (+ class subset for the 200-class sets) of a natural-shift set; cached as JSON."""
    cache = index_dir() / f"{name}.json"
    if cache.exists():
        return read_json(cache)
    root = roots()[name]
    if any(root.rglob("*.tar")):
        recs, labels = _tar_records(root)
        kind = "tar"
    else:
        recs, labels = _folder_records(root)
        kind = "folder"
    if not recs:
        raise RuntimeError(f"no images found for {name} under {root}")
    subset = sorted(set(labels)) if name in ("imagenet_r", "imagenet_a") else None
    out = {"name": name, "kind": kind, "records": recs, "labels": labels, "class_subset": subset}
    atomic_write_json(cache, out)
    return out


def read_record(kind: str, rec: Any) -> Image.Image:
    if kind == "folder":
        return Image.open(rec)
    tp, off, size = rec
    with open(tp, "rb") as f:
        f.seek(int(off))
        data = f.read(int(size))
    return Image.open(io.BytesIO(data))
