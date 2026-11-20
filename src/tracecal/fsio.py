"""Feature-directory layout (torch-free so the CPU analysis never imports torch)."""

from __future__ import annotations

from pathlib import Path

from .paths import get_paths


def source_dirname(source: str) -> str:
    return source.replace(":", "__")


def features_dir(model_tag: str, source: str, version: int = 0, bank: str = "default") -> Path:
    suffix = "" if bank == "default" else f"_{bank}"
    return get_paths().features_root / model_tag / source_dirname(source) / f"v{int(version)}{suffix}"


def is_done(model_tag: str, source: str, version: int = 0, bank: str = "default") -> bool:
    return (features_dir(model_tag, source, version, bank) / ".ok").exists()
