"""Validated path configuration.

Every machine-specific location is resolved from environment variables (see ``configs/ws/env.sh``);
source files never hard-code absolute paths.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    project_root: Path
    data_root: Path
    shared_datasets_root: Path | None
    checkpoint_root: Path
    results_root: Path
    server_root: str | None
    local_root: str | None

    @field_validator("project_root", "data_root", "checkpoint_root", "results_root")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        if not v.is_absolute():
            raise ValueError(f"path must be absolute: {v}")
        return v

    @property
    def features_root(self) -> Path:
        return self.results_root / "features"

    @property
    def raw_results_root(self) -> Path:
        return self.results_root / "raw"

    @property
    def logs_root(self) -> Path:
        return self.results_root / "logs"

    def ensure(self) -> Paths:
        for p in (self.results_root, self.features_root, self.raw_results_root, self.logs_root):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    project_root = _env_path("PROJECT_ROOT", _REPO_ROOT)
    assert project_root is not None
    return Paths(
        project_root=project_root,
        data_root=_env_path("DATA_ROOT", project_root / "data"),  # type: ignore[arg-type]
        shared_datasets_root=_env_path("SHARED_DATASETS_ROOT", None),
        checkpoint_root=_env_path("CHECKPOINT_ROOT", project_root / ".cache"),  # type: ignore[arg-type]
        results_root=_env_path("RESULTS_ROOT", project_root / "results"),  # type: ignore[arg-type]
        server_root=os.environ.get("SERVER_ROOT"),
        local_root=os.environ.get("LOCAL_ROOT"),
    )
