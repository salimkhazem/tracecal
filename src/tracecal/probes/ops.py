"""Image operations shared by the probe bank and the synthetic shift programs.

All ops take float tensors ``x`` of shape (B, 3, H, W) with values in [0, 1] (any device) and return a NEW tensor of
the same shape, clamped to [0, 1]. Scalar parameters apply to the whole batch; per-image parameters may be given as a
1-D tensor/sequence of length B for the photometric ops. Operations are deterministic: the noise op takes an explicit
noise field, and JPEG uses PIL (libjpeg) on CPU so results do not depend on the GPU model.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_GRAY_W = (0.299, 0.587, 0.114)
_POOL: ThreadPoolExecutor | None = None


def _per_image(v: float | Sequence[float] | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcastable (B,1,1,1) tensor (or scalar tensor) on x's device/dtype."""
    t = torch.as_tensor(v, dtype=x.dtype, device=x.device)
    if t.ndim == 0:
        return t
    if t.ndim != 1 or t.numel() != x.shape[0]:
        raise ValueError(f"per-image parameter must have length {x.shape[0]}, got shape {tuple(t.shape)}")
    return t.view(-1, 1, 1, 1)


def to_gray(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_GRAY_W, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def brightness(x: torch.Tensor, delta) -> torch.Tensor:
    return (x + _per_image(delta, x)).clamp(0.0, 1.0)


def contrast(x: torch.Tensor, factor) -> torch.Tensor:
    m = to_gray(x).mean(dim=(1, 2, 3), keepdim=True)
    return (m + _per_image(factor, x) * (x - m)).clamp(0.0, 1.0)


def saturation(x: torch.Tensor, factor) -> torch.Tensor:
    g = to_gray(x)
    return (g + _per_image(factor, x) * (x - g)).clamp(0.0, 1.0)


def gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    k = torch.exp(-0.5 * (t / float(sigma)) ** 2)
    return (k / k.sum()).to(dtype)


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding (radius ceil(3 sigma))."""
    if sigma <= 0:
        return x.clone()
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    h, w = x.shape[-2:]
    mode = "reflect" if r < min(h, w) else "replicate"
    y = F.pad(x, (r, r, 0, 0), mode=mode)
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous(), groups=c)
    y = F.pad(y, (0, 0, r, r), mode=mode)
    y = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous(), groups=c)
    return y.clamp(0.0, 1.0)


def sharpen(x: torch.Tensor, sigma: float, amount: float) -> torch.Tensor:
    """Unsharp masking: x + amount * (x - blur(x))."""
    return (x + float(amount) * (x - blur(x, sigma))).clamp(0.0, 1.0)


def add_noise(x: torch.Tensor, sigma, field: torch.Tensor) -> torch.Tensor:
    """x + sigma * field, where ``field`` is a standard-normal tensor of shape (3,H,W) or (B,3,H,W)."""
    f = field.to(device=x.device, dtype=x.dtype)
    if f.ndim == 3:
        f = f.unsqueeze(0)
    return (x + _per_image(sigma, x) * f).clamp(0.0, 1.0)


def noise_field(shape: tuple[int, int, int], seed: int) -> torch.Tensor:
    """Device-independent standard-normal field (numpy PCG64)."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))


def downup(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Resolution loss: antialiased bilinear downsampling by ``factor`` followed by bilinear upsampling."""
    h, w = x.shape[-2:]
    hh, ww = max(1, int(round(h * float(factor)))), max(1, int(round(w * float(factor))))
    y = F.interpolate(x, size=(hh, ww), mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def hflip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(-1)


def _jpeg_one(arr: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def jpeg(x: torch.Tensor, quality) -> torch.Tensor:
    """JPEG round trip through PIL/libjpeg at the given quality (scalar or per-image sequence)."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=16)
    b = x.shape[0]
    q = [int(quality)] * b if np.ndim(quality) == 0 else [int(v) for v in quality]
    if len(q) != b:
        raise ValueError("per-image JPEG quality must have length B")
    u8 = (x.detach() * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    outs = list(_POOL.map(_jpeg_one, list(u8), q))
    y = torch.from_numpy(np.stack(outs)).permute(0, 3, 1, 2).to(device=x.device, dtype=x.dtype) / 255.0
    return y


# ---- image-adaptive restorers (camera bank, banks_secondary.yaml: camera; added 2026-11-04) -----------------------
# Each estimates its correction from the image itself, so it is close to the identity on a well-exposed clean image.

def auto_gamma(x: torch.Tensor, target: float = 0.45, lo: float = 1.0 / 3.0, hi: float = 3.0) -> torch.Tensor:
    """Exposure correction: per-image gamma that maps the mean luminance to ``target`` (gamma clamped to [lo, hi])."""
    m = to_gray(x).mean(dim=(1, 2, 3), keepdim=True).clamp(1e-3, 0.999)
    g = (math.log(target) / torch.log(m)).clamp(lo, hi)
    return x.clamp_min(1e-6).pow(g).clamp(0.0, 1.0)


def stretch(x: torch.Tensor, q_lo: float = 0.01, q_hi: float = 0.99) -> torch.Tensor:
    """Contrast stretch: the luminance quantiles q_lo / q_hi of each image are mapped to 0 / 1 (all channels alike)."""
    g = to_gray(x).flatten(1)
    lo = torch.quantile(g, q_lo, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(g, q_hi, dim=1).view(-1, 1, 1, 1)
    span = (hi - lo).clamp_min(1e-3)
    return ((x - lo) / span).clamp(0.0, 1.0)


def equalize(x: torch.Tensor) -> torch.Tensor:
    """Histogram equalization of each channel (torchvision)."""
    import torchvision.transforms.v2.functional as TF

    u8 = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    return TF.equalize(u8).to(x.dtype) / 255.0


def grayworld(x: torch.Tensor, lo: float = 0.5, hi: float = 2.0) -> torch.Tensor:
    """White balance: scale each channel so that its mean equals the mean over channels (gains clamped)."""
    mu = x.mean(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    gain = (mu.mean(dim=1, keepdim=True) / mu).clamp(lo, hi)
    return (x * gain).clamp(0.0, 1.0)


def median3(x: torch.Tensor) -> torch.Tensor:
    """Edge-preserving denoising: 3x3 median filter (reflect padding)."""
    b, c, h, w = x.shape
    patches = F.unfold(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel_size=3)       # (B, C*9, H*W)
    return patches.view(b, c, 9, h * w).median(dim=2).values.view(b, c, h, w)


def apply_op(x: torch.Tensor, spec: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Apply one probe/program op described by ``spec`` (keys: op + parameters)."""
    op = spec["op"]
    if op == "brightness":
        return brightness(x, spec["delta"])
    if op == "contrast":
        return contrast(x, spec["factor"])
    if op == "saturation":
        return saturation(x, spec["factor"])
    if op == "blur":
        return blur(x, spec["sigma"])
    if op == "sharpen":
        return sharpen(x, spec["sigma"], spec.get("amount", 1.0))
    if op == "noise":
        if noise is None:
            raise ValueError("noise op needs a noise field")
        return add_noise(x, spec["sigma"], noise)
    if op == "jpeg":
        return jpeg(x, spec["quality"])
    if op == "downup":
        return downup(x, spec["factor"])
    if op == "hflip":
        return hflip(x)
    if op == "identity":
        return x.clone()
    if op == "auto_gamma":
        return auto_gamma(x, spec.get("target", 0.45))
    if op == "stretch":
        return stretch(x, spec.get("q_lo", 0.01), spec.get("q_hi", 0.99))
    if op == "equalize":
        return equalize(x)
    if op == "grayworld":
        return grayworld(x)
    if op == "median3":
        return median3(x)
    # TTA-CP policies (Shanmugam et al. 2025), banks_secondary.yaml: ttacp
    import torchvision.transforms.v2.functional as TF

    if op == "sharpness":
        return TF.adjust_sharpness(x, spec["factor"])
    if op == "autocontrast":
        return TF.autocontrast(x)
    if op == "invert":
        return TF.invert(x)
    if op == "posterize":
        u8 = (x.clamp(0, 1) * 255).round().to(torch.uint8)
        return TF.posterize(u8, spec["bits"]).to(x.dtype) / 255.0
    if op == "affine":
        return TF.affine(x, angle=spec["angle"], translate=list(spec["translate"]), scale=1.0,
                         shear=[spec["shear"], 0.0])
    if op == "color_jitter":
        y = TF.adjust_brightness(x, spec["brightness"])
        y = TF.adjust_contrast(y, spec["contrast"])
        y = TF.adjust_saturation(y, spec["saturation"])
        return TF.adjust_hue(y, spec["hue"]).clamp(0, 1)
    if op == "center_resized_crop":
        h, w = x.shape[-2:]
        c = int(spec["size"])
        return TF.resized_crop(x, (h - c) // 2, (w - c) // 2, c, c, [h, w], antialias=True)
    if op == "shift_crop":
        p = int(spec["pad"])
        oy, ox = spec["offset"]
        h, w = x.shape[-2:]
        return torch.nn.functional.pad(x, (p, p, p, p))[..., oy : oy + h, ox : ox + w]
    raise ValueError(f"unknown op {op!r}")
