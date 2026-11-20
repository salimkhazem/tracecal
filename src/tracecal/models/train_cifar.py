"""Train the CIFAR classifiers on the 40k non-pool training images (GPU-resident data, GPU augmentation)."""

from __future__ import annotations

import math
import time


import numpy as np
import torch
import torch.nn.functional as F

from ..config import protocol
from ..data.cifar import N_CLASSES, load_cifar
from ..data.splits import cifar_holdout
from ..paths import get_paths
from ..utils.io import atomic_write_json
from ..utils.manifest import RunManifest
from ..utils.seed import seed_everything
from .zoo import Wrapped, build_cifar_net, checkpoint_path


def _augment(xb: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop (zero padding 4) + horizontal flip, per image, on GPU. xb: float (B,3,32,32)."""
    b = xb.shape[0]
    dev = xb.device
    padded = F.pad(xb, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b,), generator=gen, device=dev)
    ox = torch.randint(0, 9, (b,), generator=gen, device=dev)
    ar = torch.arange(32, device=dev)
    rows = (oy[:, None] + ar[None, :])[:, None, :, None]
    cols = (ox[:, None] + ar[None, :])[:, None, None, :]
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(3, device=dev)[None, :, None, None]
    out = padded[bi, ci, rows, cols]
    flip = torch.rand(b, generator=gen, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)

@torch.no_grad()
def evaluate(model: Wrapped, x: np.ndarray, y: np.ndarray, bs: int = 500) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(y), bs):
        xb = torch.from_numpy(x[s : s + bs]).permute(0, 3, 1, 2).cuda().float().div_(255)
        logits, _ = model(xb)
        correct += int((logits.argmax(1).cpu().numpy() == y[s : s + bs]).sum())
    return correct / len(y)


def train(dataset: str, arch: str, seed: int) -> dict:
    cfg = protocol()["models"]["cifar"][arch]
    tag = f"{'c10' if dataset == 'cifar10' else 'c100'}_{arch}_s{seed}"
    out = checkpoint_path(tag)
    run_dir = get_paths().results_root / "runs" / "train" / tag
    seed_everything(seed, deterministic=False)
    torch.backends.cudnn.benchmark = True
    nc = N_CLASSES[dataset]
    xtr, ytr = load_cifar(dataset, "train")
    tr_idx, pool_idx = cifar_holdout(ytr)
    assert len(np.intersect1d(tr_idx, pool_idx)) == 0
    X = torch.from_numpy(xtr[tr_idx]).permute(0, 3, 1, 2).contiguous().cuda()
    Y = torch.from_numpy(ytr[tr_idx]).cuda()
    net, kind, size, mean, std = build_cifar_net(arch, nc, pretrained=(arch == "vits16"))
    model = Wrapped(kind, net, size, mean, std, nc, int(getattr(net, "num_features", 512)), tag=tag).cuda()
    model.amp_dtype = torch.bfloat16
    if kind == "cifar_resnet":
        model = model.to(memory_format=torch.channels_last)
    epochs, bs = int(cfg["epochs"]), int(cfg["batch"])
    steps_per_epoch = math.ceil(len(Y) / bs)
    total = epochs * steps_per_epoch
    if arch == "resnet18":
        opt = torch.optim.SGD(model.parameters(), lr=float(cfg["lr"]), momentum=float(cfg["momentum"]),
                              weight_decay=float(cfg["wd"]), nesterov=True)
        warm = 0
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
        warm = int(cfg["warmup_epochs"]) * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    config = {"dataset": dataset, "arch": arch, "seed": seed, **cfg, "n_train": int(len(Y))}
    with RunManifest(run_dir, config, {"seed": seed}, get_paths().project_root) as man:
        t0 = time.time()
        step = 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Y), generator=gen, device="cuda")
            running = 0.0
            for i in range(steps_per_epoch):
                idx = perm[i * bs : (i + 1) * bs]
                xb = _augment(X[idx].float().div_(255), gen)
                logits, _ = model(xb)
                loss = F.cross_entropy(logits, Y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                running += float(loss.detach())
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"[{tag}] epoch {ep + 1}/{epochs} loss {running / steps_per_epoch:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        model.eval()
        xte, yte = load_cifar(dataset, "test")
        acc = evaluate(model, xte, yte)
        pool_acc = evaluate(model, xtr[pool_idx], ytr[pool_idx])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.net.state_dict(), "config": config}, out)
        metrics = {"test_top1": acc, "pool_top1": pool_acc, "train_seconds": time.time() - t0}
        man.data["metrics"] = metrics
        man.add_artifact(out)
        atomic_write_json(run_dir / "metrics.json", metrics)
    print(f"[{tag}] test top-1 {acc:.4f}  pool top-1 {pool_acc:.4f}", flush=True)
    return metrics
