"""Command-line entry point: ``uv run tracecal --help``."""

from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _locked(kind: str, tag: str, resplit: int, fn, sub: str = "") -> None:
    """Run fn() holding an exclusive lock on its output (a duplicate invocation skips instead of racing)."""
    from .paths import get_paths
    from .utils.io import RunLock

    out = get_paths().raw_results_root / kind / tag / f"r{resplit}"
    with RunLock(out / sub if sub else out) as got:
        if not got:
            print(f"skip (another process holds the lock): {kind} {tag} r{resplit} {sub}")
            return
        fn()
data_app = typer.Typer(no_args_is_help=True)
gates_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="dataset registry and integrity checks")
app.add_typer(gates_app, name="gates", help="evaluate pre-registered gates")


@data_app.command("verify")
def data_verify() -> None:
    """Existence / shape checks of every dataset in the registry."""
    from .data.registry import verify

    typer.echo(json.dumps(verify(), indent=2))


@data_app.command("cifar-align")
def data_cifar_align() -> None:
    """CIFAR-C rows must align with the clean test set (labels equal for every severity block)."""
    from .data.cifar import check_c_alignment

    out = {n: check_c_alignment(n) for n in ("cifar10", "cifar100")}
    typer.echo(json.dumps(out, indent=2))
    if not all(v["all_ok"] for v in out.values()):
        raise typer.Exit(1)


@data_app.command("idmap")
def data_idmap(n: int = 500) -> None:
    """ImageNet val <-> ImageNet-C alignment: label agreement on all 50k IDs + brightness/1 pixel correlation."""
    from .analysis.idmap import check

    out = check(n)
    typer.echo(json.dumps(out, indent=2))
    if not out["pass"]:
        raise typer.Exit(1)


@data_app.command("index-naturals")
def data_index_naturals() -> None:
    from .data.imagenet import natural_index

    for name in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a"):
        idx = natural_index(name)
        typer.echo(f"{name}: kind={idx['kind']} n={len(idx['records'])} "
                   f"classes={len(set(idx['labels']))} subset={None if idx['class_subset'] is None else len(idx['class_subset'])}")


@app.command("train-cifar")
def train_cifar(dataset: str, arch: str, seed: int) -> None:
    from .models.train_cifar import train

    typer.echo(json.dumps(train(dataset, arch, seed), indent=2))


@app.command("extract")
def extract(model: str, source: str, version: int = 0, bank: str = "default", batch_size: int = 256,
            workers: int = 12, limit: int | None = None, overwrite: bool = False) -> None:
    from .extract import run

    run(model, source, version, bank, batch_size, workers, limit, overwrite)


@app.command("extract-all")
def extract_all(model: str, dataset: str, bank: str = "default", batch_size: int = 256, workers: int = 12,
                only: str = "") -> None:
    """All sources one model needs (pool versions, test, corruptions, naturals) with one model load.
    ``--only pool|test|naturals`` restricts the set."""
    from .extract import required_sources, run_many

    items = required_sources(dataset)
    if only == "pool":
        items = [i for i in items if i[0].endswith(":pool")]
    elif only == "test":
        items = [i for i in items if not i[0].endswith(":pool") and not i[0].startswith("imagenet_")]
    elif only == "naturals":
        items = [i for i in items if i[0].startswith("imagenet_")]
    run_many(model, items, bank, batch_size, workers)


@app.command("analyze")
def analyze(model: str, dataset: str, resplit: int, proto: str = "A", alpha: float = 0.1, score: str = "lac",
            no_tta: bool = False, naturals: bool = False) -> None:
    from .analysis.run import run

    run(model, dataset, resplit, proto, alpha, score, with_tta=not no_tta, naturals=naturals)


@app.command("frontier")
def frontier(model: str, dataset: str, resplit: int, naturals: bool = False) -> None:
    """Pre-registration v2 confirmatory evaluation (coverage/size frontiers for RU-CP and baselines)."""
    from .analysis.frontier import run_frontier

    _locked("frontier", model, resplit, lambda: run_frontier(model, dataset, resplit, naturals=naturals))


@app.command("frontier-multi")
def frontier_multi(model: str, dataset: str, resplits: str = "0,1,2,3,4", out_root: str = "") -> None:
    """Same as `frontier` for several re-splits of one model, reading every test condition once (I/O bound runs)."""
    from pathlib import Path

    from .analysis.frontier import run_frontier_multi
    from .paths import get_paths
    from .utils.io import RunLock

    rs = [int(x) for x in resplits.split(",") if x != ""]
    root = Path(out_root) if out_root else None
    # hold the per-re-split locks used by `frontier` so the two entry points never write the same run
    base = (root or get_paths().raw_results_root / "frontier") / model
    locks = [RunLock(base / f"r{r}") for r in rs]
    got = [lk.__enter__() for lk in locks]
    try:
        mine = [r for r, g in zip(rs, got) if g]
        if len(mine) < len(rs):
            print(f"skipping re-splits held by another process: {[r for r, g in zip(rs, got) if not g]}")
        if mine:
            run_frontier_multi(model, dataset, mine, out_root=root)
    finally:
        for lk in locks:
            lk.__exit__(None, None, None)


def _multi_locked(kind: str, model: str, resplits: str, fn) -> None:
    """Hold the per-re-split locks of `kind` (same lock files as the single-re-split commands), run fn(mine)."""
    from .paths import get_paths
    from .utils.io import RunLock

    rs = [int(x) for x in resplits.split(",") if x != ""]
    base = get_paths().raw_results_root / kind / model
    locks = [RunLock(base / f"r{r}") for r in rs]
    got = [lk.__enter__() for lk in locks]
    try:
        mine = [r for r, g in zip(rs, got) if g]
        if len(mine) < len(rs):
            print(f"skipping re-splits held by another process: {[r for r, g in zip(rs, got) if not g]}")
        if mine:
            fn(mine)
    finally:
        for lk in locks:
            lk.__exit__(None, None, None)


@app.command("dense-check-multi")
def dense_check_multi(model: str, dataset: str, resplits: str = "0,1,2,3,4") -> None:
    """dense-check for several re-splits of one model, reading every test condition once."""
    from .analysis.dense_check import run_multi

    _multi_locked("dense_check", model, resplits, lambda rs: run_multi(model, dataset, rs))


@app.command("nonempty-multi")
def nonempty_multi(model: str, dataset: str, resplits: str = "0,1,2,3,4") -> None:
    """nonempty for several re-splits of one model, reading every test condition once."""
    from .analysis.nonempty import run_multi

    _multi_locked("nonempty", model, resplits, lambda rs: run_multi(model, dataset, rs))


@app.command("frontier-v3")
def frontier_v3(model: str, dataset: str, resplits: str = "0,1,2,3,4", out_root: str = "") -> None:
    """Pre-registration v3: RBE-CP (lam selected on synthetic corruptions) and baselines, all re-splits of one model."""
    from pathlib import Path

    from .analysis.frontier_v3 import run_multi

    root = Path(out_root) if out_root else None
    if root is None:
        _multi_locked("frontier_v3", model, resplits, lambda rs: run_multi(model, dataset, rs))
    else:
        run_multi(model, dataset, [int(x) for x in resplits.split(",") if x != ""], out_root=root)


@app.command("frontier-v3-naturals")
def frontier_v3_naturals(model: str, resplits: str = "0,1,2,3,4") -> None:
    """v3 K5 (report only): natural-shift curves on a fresh ImageNet model."""
    from .analysis.frontier_v3 import run_naturals_v3

    run_naturals_v3(model, [int(x) for x in resplits.split(",") if x != ""])


@app.command("frontier-v4")
def frontier_v4(model: str, dataset: str, resplits: str = "0,1,2,3,4") -> None:
    """Pre-registration v4: RBE-CP with the symmetric (identity + flip) anchor, v3 RBE-CP and baselines."""
    from .analysis.frontier_v4 import run_multi

    _multi_locked("frontier_v4", model, resplits, lambda rs: run_multi(model, dataset, rs))


CBAR = ["blue_noise_sample", "brownish_noise", "caustic_refraction", "checkerboard_cutout", "cocentric_sine_waves",
        "inverse_sparkles", "perlin_noise", "plasma_noise", "single_frequency_greyscale", "sparkles"]


def cbar_conditions() -> list[tuple[str, str]]:
    return [("clean", "imagenet:test10k")] + [(f"{c}:{s}", f"imagenetcbar:{c}:{s}") for c in CBAR for s in range(1, 6)]


@app.command("extract-cbar")
def extract_cbar(model: str, batch_size: int = 256, workers: int = 10) -> None:
    """Pre-registration v5b: the 50 ImageNet-C-bar test conditions (10k subset) with one model load."""
    from .extract import run_many

    run_many(model, [(src, 0) for _, src in cbar_conditions()], "default", batch_size, workers)


@app.command("extract-ttacp")
def extract_ttacp(model: str, batch_size: int = 256, workers: int = 10) -> None:
    """Baseline check (audits/baseline_ttacp_policies.yaml): TTA-CP's own augmentation policies (bank ttacp) on the
    clean pool, the clean test subset and the 95 ImageNet-C conditions."""
    from .analysis.run import test_conditions
    from .extract import run_many

    items = [("imagenet:pool", 0)] + [(src, 0) for _, src in test_conditions("imagenet")]
    run_many(model, items, "ttacp", batch_size, workers)


@app.command("extract-e1")
def extract_e1(model: str, batch_size: int = 512, workers: int = 4) -> None:
    """E1 (internal review): CIFAR test images under three exactly-invertible shifts (default 13-view bank)."""
    from .extract import run_many

    ds = "cifar10" if model.startswith("c10_") else "cifar100" if model.startswith("c100_") else "imagenet"
    run_many(model, [(f"{ds}rs:{s}", 0) for s in ("dark", "lowcontrast", "desat")], "default", batch_size, workers)


@app.command("extract-magnitude")
def extract_magnitude(model: str, batch_size: int = 512, workers: int = 4) -> None:
    """E2 (internal review): the 'weak' and 'strong' restorer banks on the pool versions and test conditions (CIFAR)."""
    from .extract import required_sources, run_many

    ds = "cifar10" if model.startswith("c10_") else "cifar100"
    for bank in ("weak", "strong"):
        run_many(model, required_sources(ds), bank, batch_size, workers)


@app.command("frontier-v4-cbar")
def frontier_v4_cbar(model: str, resplits: str = "0,1,2,3,4") -> None:
    """Pre-registration v5b: the frozen v3 selection (rbe3_sel) and baselines on ImageNet-C-bar."""
    from .analysis.frontier_v4 import run_multi
    from .paths import get_paths

    root = get_paths().raw_results_root / "frontier_v4_cbar"
    _multi_locked("frontier_v4_cbar", model, resplits,
                  lambda rs: run_multi(model, "imagenet", rs, out_root=root, conds=cbar_conditions()))


def es_conditions() -> list[tuple[str, str]]:
    """Pre-registration v5d: reference + 27 manual settings x 2 lights + 5 auto-exposure shots x 2 lights = 65."""
    return [("reference", "imagenetes:ref")] + \
        [(f"pc:{l}:{k}", f"imagenetes:pc:{l}:{k}") for l in ("l1", "l5") for k in range(1, 28)] + \
        [(f"ae:{l}:{k}", f"imagenetes:ae:{l}:{k}") for l in ("l1", "l5") for k in range(1, 6)]


@app.command("extract-es")
def extract_es(model: str, batch_size: int = 256, workers: int = 10) -> None:
    """Pre-registration v5d: the 65 ImageNet-ES es-test conditions (default 13-view bank) with one model load."""
    from .extract import run_many

    run_many(model, [(src, 0) for _, src in es_conditions()], "default", batch_size, workers)


@app.command("frontier-v4-es")
def frontier_v4_es(model: str, resplits: str = "0,1,2,3,4") -> None:
    """Pre-registration v5d: the frozen RBE-CP (lam* and TTA-CP weights of the main v4 run) on ImageNet-ES."""
    from .analysis.frontier_v4 import run_es_v4
    from .data.sources import imagenet_es_index

    idx = imagenet_es_index()
    run_es_v4(model, [int(x) for x in resplits.split(",") if x != ""], es_conditions(), idx["class_subset"],
              idx["source_ids"])


def esval_conditions() -> list[tuple[str, str]]:
    """ImageNet-ES es-val (development split for v6): reference + 64 manual settings x 2 lights + 5 auto x 2 = 139."""
    return [("reference", "imagenetesval:ref")] + \
        [(f"pc:{l}:{k}", f"imagenetesval:pc:{l}:{k}") for l in ("l1", "l5") for k in range(1, 65)] + \
        [(f"ae:{l}:{k}", f"imagenetesval:ae:{l}:{k}") for l in ("l1", "l5") for k in range(1, 6)]


def inc_dev_conditions() -> list[tuple[str, str]]:
    from .analysis.run import test_conditions

    return [(c, s) for c, s in test_conditions("imagenet") if c == "clean" or c.endswith(":3") or c.endswith(":5")]


@app.command("extract-camera-dev")
def extract_camera_dev(model: str, batch_size: int = 256, workers: int = 10) -> None:
    """v6 development (v2 development models only): es-val with the default and camera banks, the camera bank on the
    calibration pool (lambda selection) and on an ImageNet-C subset (clean, severities 3 and 5)."""
    from .extract import run_many
    from .shifts.programs import n_versions

    es = [(src, 0) for _, src in esval_conditions()]
    run_many(model, es, "default", batch_size, workers)
    run_many(model, es + [("imagenet:pool", v) for v in range(n_versions())]
             + [(src, 0) for _, src in inc_dev_conditions()], "camera", batch_size, workers)


@app.command("extract-camera")
def extract_camera(model: str, batch_size: int = 256, workers: int = 10) -> None:
    """Pre-registration v6 (fresh models): the camera bank on the calibration pool, every ImageNet-C test condition and
    ImageNet-ES es-test (the default bank exists already)."""
    from .analysis.run import test_conditions
    from .extract import run_many
    from .shifts.programs import n_versions

    items = [("imagenet:pool", v) for v in range(n_versions())] + [(s, 0) for _, s in test_conditions("imagenet")] + \
        [(s, 0) for _, s in es_conditions()]
    run_many(model, items, "camera", batch_size, workers)


@app.command("extract-camera-naturals")
def extract_camera_naturals(model: str, batch_size: int = 256, workers: int = 10) -> None:
    """v6 report-only (not pre-registered): the camera bank on ImageNet-V2, -Sketch, -R and -A."""
    from .extract import run_many

    run_many(model, [(n, 0) for n in ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")], "camera",
             batch_size, workers)


@app.command("frontier-v6")
def frontier_v6(model: str, resplits: str = "0,1,2,3,4") -> None:
    """Pre-registration v6: RBE-CP with the camera-aware bank on ImageNet-C and ImageNet-ES es-test."""
    from .analysis.frontier_v6 import run_v6
    from .data.sources import imagenet_es_index

    idx = imagenet_es_index("es-test")
    run_v6(model, [int(x) for x in resplits.split(",") if x != ""], es_conditions(), idx["class_subset"],
           idx["source_ids"])


@app.command("frontier-v6-naturals")
def frontier_v6_naturals(model: str, resplits: str = "0,1,2,3,4") -> None:
    """v6 report only: the camera-aware RBE-CP on ImageNet-V2, -Sketch, -R and -A."""
    from .analysis.frontier_v6 import run_naturals_v6

    run_naturals_v6(model, [int(x) for x in resplits.split(",") if x != ""])


@app.command("frontier-v4-naturals")
def frontier_v4_naturals(model: str, resplits: str = "0,1,2,3,4") -> None:
    """v4: natural-shift curves (ImageNet models) with the selected lam of the main v4 run."""
    from .analysis.frontier_v4 import run_naturals_v4

    run_naturals_v4(model, [int(x) for x in resplits.split(",") if x != ""])


@app.command("frontier-naturals")
def frontier_naturals(model: str, resplit: int) -> None:
    """H5 (report only): natural-shift frontiers for an existing ImageNet frontier run."""
    from .analysis.frontier import run_naturals

    _locked("frontier", model, resplit, lambda: run_naturals(model, resplit), sub="naturals")


@app.command("ablation-bank")
def ablation_bank(model: str, dataset: str, resplit: int) -> None:
    """SECONDARY bank ablations for RU-CP (leave-one-restorer-out, +degraders, +flip, bound tightness)."""
    from .analysis.ablation_bank import run

    _locked("ablation_bank", model, resplit, lambda: run(model, dataset, resplit))


@app.command("complementary")
def complementary(model: str, dataset: str, resplit: int) -> None:
    """SECONDARY: QTC recalibration applied to base sets vs restorative-union sets."""
    from .analysis.complementary import run

    _locked("complementary", model, resplit, lambda: run(model, dataset, resplit))


@app.command("mixed-stream")
def mixed_stream(model: str, dataset: str, resplit: int) -> None:
    """SECONDARY: mixed acquisition stream (one random condition per test image)."""
    from .analysis.mixed_stream import run

    _locked("mixed_stream", model, resplit, lambda: run(model, dataset, resplit))


@app.command("dense-check")
def dense_check(model: str, dataset: str, resplit: int) -> None:
    """ROBUSTNESS: efficiency-matched statistic on a dense alpha' grid."""
    from .analysis.dense_check import run

    _locked("dense_check", model, resplit, lambda: run(model, dataset, resplit))


@app.command("nonempty")
def nonempty(model: str, dataset: str, resplit: int) -> None:
    """SECONDARY: RU-CP vs non-empty LAC (always include the top-1 label)."""
    from .analysis.nonempty import run

    _locked("nonempty", model, resplit, lambda: run(model, dataset, resplit))


@app.command("strength")
def strength(model: str, dataset: str, resplit: int) -> None:
    """SECONDARY: restorer-magnitude sensitivity (weak / strong banks, configs/core/banks_secondary.yaml)."""
    from .analysis.strength import run

    _locked("strength", model, resplit, lambda: run(model, dataset, resplit))


@app.command("g0-accuracy")
def g0_accuracy(model: str, batch_size: int = 128, workers: int = 16) -> None:
    """Full ImageNet-val top-1 under the model's own default eval transform (G0 reproduction)."""
    from .analysis.g0 import imagenet_default_accuracy

    typer.echo(json.dumps(imagenet_default_accuracy(model, batch_size, workers), indent=2))


@app.command("numerics")
def numerics(model: str, source: str, n: int = 512) -> None:
    """fp16 vs fp32 response-profile discrepancy (G0 numerics; decides the extraction precision)."""
    from .analysis.numerics import compare

    typer.echo(json.dumps(compare(model, source, n), indent=2))


@gates_app.command("g1")
def gate_g1() -> None:
    from .analysis.g1 import g1
    from .analysis.gates_eval import write

    res = g1()
    write({"G1_signal": res})
    typer.echo(json.dumps(res, indent=2))


@gates_app.command("g2")
def gate_g2() -> None:
    from .analysis.gates_eval import g2, write

    res = g2()
    write({"G2_decisive": res})
    typer.echo(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=2))


@gates_app.command("g3")
def gate_g3() -> None:
    from .analysis.gates_eval import g3, write

    res = g3()
    write({"G3_imagenet": res})
    typer.echo(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=2))


@gates_app.command("v2")
def gates_v2(which: str = "all") -> None:
    """Pre-registration v2 gates: h1 (CIFAR seeds 1-4), h2 (ImageNet), h3 (guarantee sanity), h4 (mechanism)."""
    from .analysis import gates_v2_eval as g

    cifar_conf = [f"{d}_{a}_s{s}" for d in ("c10", "c100") for a in ("resnet18", "vits16") for s in (1, 2, 3, 4)]
    imagenet = ["in_rn50", "in_vitb16", "in_convnext_t", "in_swin_t", "in_clip_b16"]
    res = {}
    if which in ("all", "h1"):
        res["H1_cifar_confirm"] = g.h1()
    if which in ("all", "h2"):
        res["H2_imagenet_confirm"] = g.h2()
    if which in ("all", "h3"):
        res["H3_guarantee_sanity"] = g.h3(cifar_conf + imagenet)
    if which in ("all", "h4"):
        res["H4_mechanism"] = g.h4(cifar_conf + imagenet)
    g.write(res)
    typer.echo(json.dumps(res, indent=1, default=float)[:6000])


@app.command("smoke")
def smoke(dataset: str = "cifar10", n: int = 500) -> None:
    """End-to-end smoke test on a tiny subset (requires a trained c10_resnet18_s0 or any model tag)."""
    from .analysis.smoke import run_smoke

    run_smoke(dataset, n)


if __name__ == "__main__":
    app()
