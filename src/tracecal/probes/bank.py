"""The fixed probe bank: K probes + extra views (hflip), applied in a fixed order."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..config import probe_specs, protocol
from .ops import apply_op, noise_field


class ProbeBank:
    def __init__(self, family: str, bank: str = "default"):
        self.family = family
        self.bank = bank
        self.specs = probe_specs(family, bank)
        self.extra = list(protocol()["probes"]["extra_views"])
        self.noise_seed = int(protocol()["probes"]["noise_seed"])
        self._fields: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def probe_names(self) -> list[str]:
        return [s["name"] for s in self.specs]

    @property
    def view_names(self) -> list[str]:
        return self.probe_names + self.extra

    def field(self, h: int, w: int, device) -> torch.Tensor:
        key = (h, w)
        if key not in self._fields:
            self._fields[key] = noise_field((3, h, w), self.noise_seed)
        return self._fields[key].to(device)

    def views(self, x: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        h, w = x.shape[-2:]
        for s in self.specs:
            noise = self.field(h, w, x.device) if s["op"] == "noise" else None
            yield s["name"], apply_op(x, s, noise=noise)
        for e in self.extra:
            yield e, apply_op(x, {"op": e})
