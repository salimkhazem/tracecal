"""One GPU pass per (model, source, pool version): base + probe + extra views -> compact arrays on disk.

Outputs in ``$RESULTS_ROOT/features/<model_tag>/<source>/v<version>[_<bank>]/``:
  base_logits.npy (N,C) f32 | view_logits.npy (N,V,C) f16 | R.npy (N,K) f32 | KL.npy (N,K) f32 | flip.npy (N,K) bool
  feats.npy (N,D) f16 | labels.npy (N,) i64 | ids.npy (N,) i64 | meta.json | programs.json (pool versions > 0)
  manifest.json and an ``.ok`` marker written last (directory is renamed into place atomically).
R_k = log p_k(y0) - log p_0(y0) with y0 = argmax of the base view; KL_k = KL(p_0 || p_k).
"""

from __future__ import annotations

import fcntl
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import protocol_sha256
from .fsio import features_dir, is_done, source_dirname  # noqa: F401
from .data.sources import get_source
from .models.zoo import load
from .paths import get_paths
from .probes.bank import ProbeBank
from .shifts.programs import apply_programs
from .utils.io import atomic_write_json
from .utils.manifest import RunManifest


def _open(path: Path, shape, dtype):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(int(s) for s in shape))


@torch.inference_mode()
def run(model_tag: str, source: str, version: int = 0, bank: str = "default", batch_size: int = 256,
        num_workers: int = 12, limit: int | None = None, overwrite: bool = False, device: str = "cuda",
        model=None) -> Path:
    out = features_dir(model_tag, source, version, bank)
    if (out / ".ok").exists() and not overwrite:
        print(f"skip (done): {out}")
        return out
    # One writer per output directory: an exclusive non-blocking lock, and a private tmp directory per process (two
    # concurrent writers once clobbered each other's tmp directory and left a done marker without arrays).
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(out.with_name(out.name + ".lock"), "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"skip (another process is writing): {out}")
        lock_f.close()
        return out
    if (out / ".ok").exists() and not overwrite:
        lock_f.close()
        print(f"skip (done): {out}")
        return out
    tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    src = get_source(source)
    if version > 0 and not source.endswith(":pool"):
        raise ValueError("shift-program versions are only defined for calibration-pool sources")
    model = load(model_tag, device) if model is None else model
    pb = ProbeBank(src.family, bank)
    n = len(src.dataset) if limit is None else min(limit, len(src.dataset))
    c, d = model.n_classes, model.feat_dim
    k, v = len(pb.probe_names), len(pb.view_names)
    arr = {
        "base_logits": _open(tmp / "base_logits.npy", (n, c), np.float32),
        "view_logits": _open(tmp / "view_logits.npy", (n, v, c), np.float16),
        "R": _open(tmp / "R.npy", (n, k), np.float32),
        "KL": _open(tmp / "KL.npy", (n, k), np.float32),
        "flip": _open(tmp / "flip.npy", (n, k), np.bool_),
        "feats": _open(tmp / "feats.npy", (n, d), np.float16),
        "labels": _open(tmp / "labels.npy", (n,), np.int64),
        "ids": _open(tmp / "ids.npy", (n,), np.int64),
    }
    if src.in_memory:
        batches = src.dataset.batches(batch_size)  # type: ignore[attr-defined]
    else:
        ds = src.dataset if limit is None else torch.utils.data.Subset(src.dataset, range(n))
        batches = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                             persistent_workers=False, prefetch_factor=4 if num_workers > 0 else None)
    config = {"model": model_tag, "source": source, "version": version, "bank": bank, "batch_size": batch_size,
              "n": n, "n_classes": c, "feat_dim": d, "views": pb.view_names, "probes": pb.specs,
              "amp_dtype": str(model.amp_dtype), "protocol_sha256": protocol_sha256()}
    programs: list = []
    s = 0
    t0 = time.time()
    with RunManifest(tmp, config, {"program_base_seed": 0}, get_paths().project_root) as man:
        for xb, yb, ib in batches:
            if s >= n:
                break
            e = min(s + xb.shape[0], n)
            xb, yb, ib = xb[: e - s], yb[: e - s], ib[: e - s]
            x = xb.to(device, non_blocking=True).float().div_(255.0)
            if version > 0:
                x, progs = apply_programs(x, ib.tolist(), version, src.family)
                programs.extend(progs)
            logits0, feats = model(x)
            lp0 = torch.log_softmax(logits0, dim=1)
            p0 = lp0.exp()
            y0 = logits0.argmax(1, keepdim=True)
            lp0_y0 = lp0.gather(1, y0)
            arr["base_logits"][s:e] = logits0.cpu().numpy()
            arr["feats"][s:e] = feats.half().cpu().numpy()
            arr["labels"][s:e] = yb.numpy()
            arr["ids"][s:e] = ib.numpy()
            for j, (_, xv) in enumerate(pb.views(x)):
                lv, _ = model(xv)
                arr["view_logits"][s:e, j] = lv.half().cpu().numpy()
                if j < k:
                    lpv = torch.log_softmax(lv, dim=1)
                    arr["R"][s:e, j] = (lpv.gather(1, y0) - lp0_y0).squeeze(1).cpu().numpy()
                    arr["KL"][s:e, j] = (p0 * (lp0 - lpv)).sum(1).cpu().numpy()
                    arr["flip"][s:e, j] = (lv.argmax(1, keepdim=True) != y0).squeeze(1).cpu().numpy()
            s = e
            if (s // batch_size) % 20 == 0:
                print(f"[{model_tag} {source} v{version}] {s}/{n} {time.time() - t0:.0f}s", flush=True)
        if s != n:
            raise RuntimeError(f"wrote {s} rows, expected {n}")
        for a in arr.values():
            a.flush()
        man.data["metrics"] = {"seconds": time.time() - t0, "images_per_s": n / max(1e-9, time.time() - t0),
                               "forward_passes_per_image": 1 + v,
                               "top1_base": float((arr["base_logits"][:].argmax(1) == arr["labels"][:]).mean())}
        meta = dict(config, class_subset=src.class_subset, source_info=src.info)
        atomic_write_json(tmp / "meta.json", meta)
        if version > 0:
            atomic_write_json(tmp / "programs.json", programs)
    for name, a in arr.items():
        f = tmp / f"{name}.npy"
        if not f.exists() or np.load(f, mmap_mode="r").shape != a.shape:
            raise RuntimeError(f"{f} missing or wrong shape before publishing {out}")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    (out / ".ok").write_text("ok")
    lock_f.close()
    print(f"done {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


def required_sources(dataset: str) -> list[tuple[str, int]]:
    """Every (source, version) the analysis needs for one model of ``dataset``."""
    from .analysis.run import test_conditions
    from .shifts.programs import n_versions

    pool = [(f"{dataset}:pool", v) for v in range(n_versions())]
    tests = [(src, 0) for _, src in test_conditions(dataset)]
    nat = [(n, 0) for n in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")] if dataset == "imagenet" else []
    return pool + tests + nat


def run_many(model_tag: str, items: list[tuple[str, int]], bank: str = "default", batch_size: int = 256,
             num_workers: int = 12, device: str = "cuda") -> None:
    """Extract many (source, version) pairs with ONE model load; already-finished items are skipped."""
    todo = [(s, v) for s, v in items if not is_done(model_tag, s, v, bank)]
    print(f"[{model_tag}] {len(todo)}/{len(items)} extractions to do", flush=True)
    if not todo:
        return
    model = load(model_tag, device)
    for s, v in todo:
        run(model_tag, s, v, bank, batch_size, num_workers, device=device, model=model)
