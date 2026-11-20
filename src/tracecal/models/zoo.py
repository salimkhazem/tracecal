"""Frozen classifiers behind one interface: ``forward(x01) -> (logits fp32, features fp32)``.

``x01`` is a float batch in [0, 1] at the dataset's native resolution (CIFAR 32 px, ImageNet 224 px crops); the
wrapper resizes to the model's input size (bilinear) and normalises with the model's own statistics.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import imagenet_model_spec, protocol
from ..paths import get_paths

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------------------------------------------
# CIFAR ResNet-18 (3x3 stem, no max-pool), the standard CIFAR variant
# ----------------------------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)
        self.num_features = 512

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


# ----------------------------------------------------------------------------------------------------------------
# Wrapper
# ----------------------------------------------------------------------------------------------------------------
class Wrapped(nn.Module):
    def __init__(self, kind: str, net: nn.Module, input_size: int, mean, std, n_classes: int, feat_dim: int,
                 zs_weights: torch.Tensor | None = None, logit_scale: float | None = None, tag: str = ""):
        super().__init__()
        self.kind, self.net, self.input_size = kind, net, int(input_size)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.n_classes, self.feat_dim, self.tag = n_classes, feat_dim, tag
        if zs_weights is not None:
            self.register_buffer("zs_weights", zs_weights.float())
        self.logit_scale = logit_scale
        # inference precision: TRACECAL_AMP in {fp16, bf16, fp32}; fp32 disables autocast (TF32 off -> exact fp32)
        import os

        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
            os.environ.get("TRACECAL_AMP", "fp16")]
        if self.amp_dtype == torch.float32:  # exact fp32: TF32 off for both cuDNN convs and matmuls
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False

    def preprocess(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.shape[-1] != self.input_size or x01.shape[-2] != self.input_size:
            x01 = F.interpolate(x01, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return (x01 - self.mean) / self.std

    def forward(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.preprocess(x01)
        use_amp = x.is_cuda and self.amp_dtype != torch.float32
        with torch.autocast(device_type=x.device.type, dtype=self.amp_dtype, enabled=use_amp):
            if self.kind == "timm":
                feats = self.net.forward_head(self.net.forward_features(x), pre_logits=True)
            elif self.kind == "cifar_resnet":
                feats = self.net.features(x)
            elif self.kind == "clip":
                feats = self.net.encode_image(x)
            else:
                raise ValueError(self.kind)
        feats = feats.float()
        if self.kind == "timm":
            logits = self.net.get_classifier().float()(feats)
        elif self.kind == "cifar_resnet":
            logits = self.net.fc.float()(feats)
        else:  # clip zero-shot
            feats = F.normalize(feats, dim=-1)
            logits = float(self.logit_scale) * feats @ self.zs_weights
        return logits, feats


def checkpoint_path(tag: str) -> Path:
    return get_paths().results_root / "checkpoints" / f"{tag}.pt"


def parse_cifar_tag(tag: str) -> tuple[str, str, int]:
    """'c10_resnet18_s0' -> ('cifar10', 'resnet18', 0)."""
    ds, arch, seed = tag.split("_")
    return {"c10": "cifar10", "c100": "cifar100"}[ds], arch, int(seed.lstrip("s"))


def build_cifar_net(arch: str, n_classes: int, pretrained: bool) -> tuple[nn.Module, str, int, tuple, tuple]:
    if arch == "resnet18":
        return ResNet18Cifar(n_classes), "cifar_resnet", 32, CIFAR_MEAN, CIFAR_STD
    if arch == "vits16":
        import timm

        name = protocol()["models"]["cifar"]["vits16"]["arch"]
        net = timm.create_model(name, pretrained=pretrained, num_classes=n_classes)
        cfg = net.pretrained_cfg
        return net, "timm", int(protocol()["models"]["cifar"]["vits16"]["input"]), tuple(cfg["mean"]), tuple(cfg["std"])
    raise ValueError(arch)


def load(tag: str, device: str | torch.device = "cuda") -> Wrapped:
    """Model tags: in_rn50 | in_vitb16 | in_convnext_t | in_swin_t | in_clip_b16 | c10_resnet18_s0 | c100_vits16_s3 ..."""
    if tag.startswith("in_"):
        key = tag[3:]
        spec = imagenet_model_spec(key)
        if "robustbench" in spec:
            w = _load_robustbench(spec, tag)
        elif "timm" in spec:
            import timm

            net = timm.create_model(spec["timm"], pretrained=True)
            cfg = net.pretrained_cfg
            feat_dim = int(getattr(net, "head_hidden_size", None) or net.num_features)
            w = Wrapped("timm", net, 224, tuple(cfg["mean"]), tuple(cfg["std"]), net.num_classes, feat_dim, tag=tag)
        else:
            w = _load_clip(spec, device, tag)
    else:
        dataset, arch, _ = parse_cifar_tag(tag)
        nc = {"cifar10": 10, "cifar100": 100}[dataset]
        net, kind, size, mean, std = build_cifar_net(arch, nc, pretrained=False)
        state = torch.load(checkpoint_path(tag), map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"])
        feat_dim = int(getattr(net, "head_hidden_size", None) or getattr(net, "num_features", 512))
        w = Wrapped(kind, net, size, mean, std, nc, feat_dim, tag=tag)
    return w.eval().to(device)


def _load_robustbench(spec: dict, tag: str) -> Wrapped:
    """RobustBench ImageNet-corruption checkpoints (torchvision ResNet-50 state dicts, 'module.' prefix), loaded into
    timm's ResNet-50 (identical parameter names) so that the timm feature/classifier path is reused unchanged."""
    import hashlib

    import timm

    path = get_paths().checkpoint_root / "robust" / f"{spec['robustbench']}.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch for {path}: {digest}")
    net = timm.create_model(spec["arch"], pretrained=False, num_classes=1000)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return Wrapped("timm", net, 224, mean, std, 1000, int(net.num_features), tag=tag)


def _load_clip(spec: dict, device, tag: str) -> Wrapped:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(spec["open_clip"], pretrained=spec["pretrained"],
                                                        cache_dir=str(get_paths().checkpoint_root / "open_clip"))
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(spec["open_clip"])
    from open_clip.zero_shot_classifier import build_zero_shot_classifier
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

    with torch.no_grad():
        weights = build_zero_shot_classifier(model, tokenizer, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES,
                                             num_classes_per_batch=50, device=device, use_tqdm=False)
    mean = tuple(getattr(model.visual, "image_mean", None) or (0.48145466, 0.4578275, 0.40821073))
    std = tuple(getattr(model.visual, "image_std", None) or (0.26862954, 0.26130258, 0.27577711))
    scale = float(model.logit_scale.exp().item())
    return Wrapped("clip", model, 224, mean, std, 1000, int(weights.shape[0]), zs_weights=weights,
                   logit_scale=scale, tag=tag)


def default_eval_transform(tag: str):
    """The model's own published eval transform (used ONLY for the G0 accuracy reproduction)."""
    key = tag[3:]
    spec = imagenet_model_spec(key)
    if "robustbench" in spec:  # torchvision-style evaluation used by RobustBench (resize 256, center crop 224)
        from torchvision import transforms as T

        return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if "timm" in spec:
        import timm
        from timm.data import create_transform, resolve_data_config

        net = timm.create_model(spec["timm"], pretrained=False)
        # use_test_size: timm reports results at the TEST resolution (e.g. resnet50.tv2_in1k trains at 176, tests at 224)
        return create_transform(**resolve_data_config({}, model=net, use_test_size=True))
    import open_clip

    _, _, preprocess = open_clip.create_model_and_transforms(spec["open_clip"], pretrained=spec["pretrained"],
                                                             cache_dir=str(get_paths().checkpoint_root / "open_clip"))
    return preprocess
