"""Frozen protocol access (configs/core/protocol.yaml). The protocol is read-only at run time."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = _REPO_ROOT / "configs" / "core" / "protocol.yaml"


@lru_cache(maxsize=1)
def protocol() -> dict[str, Any]:
    with open(PROTOCOL_PATH) as f:
        return yaml.safe_load(f)


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def family_of(dataset: str) -> str:
    """Resolution family ('cifar' or 'imagenet') of a dataset name."""
    d = dataset.lower()
    if d.startswith("cifar"):
        return "cifar"
    if d.startswith("imagenet"):
        return "imagenet"
    raise ValueError(f"unknown dataset family for {dataset!r}")


SECONDARY_BANKS_PATH = _REPO_ROOT / "configs" / "core" / "banks_secondary.yaml"
MODELS_V3_PATH = _REPO_ROOT / "configs" / "core" / "models_v3.yaml"
MODELS_V4_PATH = _REPO_ROOT / "configs" / "core" / "models_v4.yaml"
MODELS_V5_PATH = _REPO_ROOT / "configs" / "core" / "models_v5.yaml"


@lru_cache(maxsize=1)
def models_v3() -> dict[str, Any]:
    if not MODELS_V3_PATH.exists():
        return {"imagenet": {}}
    with open(MODELS_V3_PATH) as f:
        return yaml.safe_load(f)


def models_v4() -> dict[str, Any]:
    if not MODELS_V4_PATH.exists():
        return {"imagenet": {}}
    with open(MODELS_V4_PATH) as f:
        return yaml.safe_load(f)


def models_v5() -> dict[str, Any]:
    if not MODELS_V5_PATH.exists():
        return {"imagenet": {}}
    with open(MODELS_V5_PATH) as f:
        return yaml.safe_load(f)


def imagenet_model_spec(key: str) -> dict[str, Any]:
    """Spec of an ImageNet model: the frozen protocol's models first, then the fresh v3 and v4 confirmation models."""
    specs = protocol()["models"]["imagenet"]
    if key in specs:
        return specs[key]
    v3 = models_v3()["imagenet"]
    if key in v3:
        return v3[key]
    v4 = models_v4()["imagenet"]
    if key in v4:
        return v4[key]
    return models_v5()["imagenet"][key]


@lru_cache(maxsize=1)
def secondary_banks() -> dict[str, Any]:
    """Post-hoc restorer-strength banks (configs/core/banks_secondary.yaml); kept out of the frozen protocol."""
    with open(SECONDARY_BANKS_PATH) as f:
        return yaml.safe_load(f)


def probe_specs(family: str, bank: str = "default") -> list[dict[str, Any]]:
    p = protocol()["probes"]
    specs = list(p[family])
    if bank == "m2":
        specs = specs + list(p["m2_extra"][family])
    elif bank in ("weak", "strong", "ttacp", "camera"):
        specs = list(secondary_banks()[bank][family])
    elif bank != "default":
        raise ValueError(bank)
    return specs


def probe_names(family: str, bank: str = "default") -> list[str]:
    return [s["name"] for s in probe_specs(family, bank)]


def view_names(family: str, bank: str = "default") -> list[str]:
    """All extracted views after the base view: probes then the extra views (hflip)."""
    return probe_names(family, bank) + list(protocol()["probes"]["extra_views"])


def program_ranges(family: str) -> dict[str, Any]:
    return protocol()["programs"][family]


def families() -> list[str]:
    return list(protocol()["programs"]["families"])
