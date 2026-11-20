"""Run manifests: every run records enough provenance to be re-executed and audited."""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .io import atomic_write_json, sha256_file

_PACKAGES = ("torch", "torchvision", "timm", "open_clip_torch", "numpy", "scipy", "scikit-learn", "pyarrow", "pillow", "pydantic", "huggingface-hub")


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_state(repo: Path) -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD", cwd=repo)
    status = _git("status", "--porcelain", cwd=repo)
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in _PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def cuda_state() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update(
                available=True,
                device_index=idx,
                gpu_name=props.name,
                vram_bytes=props.total_memory,
            )
    except Exception as e:  # pragma: no cover - diagnostic only
        info["error"] = repr(e)
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["driver"] = drv.stdout.strip().splitlines()[0] if drv.returncode == 0 else None
    except Exception:
        info["driver"] = None
    return info


class RunManifest:
    """Context manager writing ``manifest.json`` into a run directory, including failure status."""

    def __init__(self, run_dir: Path, config: dict[str, Any], seeds: dict[str, int], repo: Path):
        self.run_dir = Path(run_dir)
        self.data: dict[str, Any] = {
            "config": config,
            "seeds": seeds,
            "git": git_state(repo),
            "packages": package_versions(),
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "command": sys.argv,
            "datasets": {},
            "checkpoints": {},
            "artifacts": {},
            "metrics": {},
        }
        self._t0 = 0.0

    def add_dataset(self, name: str, **info: Any) -> None:
        self.data["datasets"][name] = info

    def add_checkpoint(self, name: str, **info: Any) -> None:
        self.data["checkpoints"][name] = info

    def add_artifact(self, path: Path, sha256: str | None = None) -> None:
        path = Path(path)
        self.data["artifacts"][str(path.relative_to(self.run_dir) if path.is_relative_to(self.run_dir) else path)] = (
            sha256 or sha256_file(path)
        )

    def __enter__(self) -> RunManifest:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self.data["start"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.data["cuda"] = cuda_state()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.data["status"] = "running"
        atomic_write_json(self.run_dir / "manifest.json", self.data)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.data["end"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.data["runtime_s"] = time.time() - self._t0
        try:
            import torch

            if torch.cuda.is_available():
                self.data["peak_gpu_mem_bytes"] = int(torch.cuda.max_memory_allocated())
                self.data["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        except Exception:
            pass
        try:
            import resource

            self.data["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            pass
        self.data["status"] = "failed" if exc_type else "ok"
        if exc_type:
            self.data["error"] = f"{exc_type.__name__}: {exc}"
        atomic_write_json(self.run_dir / "manifest.json", self.data)
        return False
