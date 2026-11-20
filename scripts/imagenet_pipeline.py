#!/usr/bin/env python
"""On ws: run every ImageNet analysis as soon as its inputs exist (instead of waiting for the whole extraction queue).

Per model m and re-split r:
  frontier m r            needs pool + test features          -> H2, H3, H4 inputs
  frontier-naturals m r   needs frontier m r + natural feats  -> H5 (report only)
  ablation / complementary / mixed-stream m r  need pool + test (secondary)
  dense-check m r         needs frontier m r (reuses its TTA weights)
When every frontier run exists: report_gates_v2 all (writes "gates done" to results/logs/after_imagenet.log).
When everything exists: secondary reporters, then "all done".
Jobs run at low CPU priority (nice) so the GPU extraction's data loaders keep priority.

  nohup nice -n 10 uv run python scripts/imagenet_pipeline.py --workers 3 > results/logs/imagenet_pipeline.out 2>&1 &
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecal.analysis.run import test_conditions  # noqa: E402
from tracecal.fsio import is_done  # noqa: E402
from tracecal.paths import get_paths  # noqa: E402
from tracecal.shifts.programs import n_versions  # noqa: E402

MODELS = ["in_rn50", "in_convnext_t", "in_swin_t", "in_vitb16", "in_clip_b16"]
RESPLITS = range(5)
NATURALS = ("imagenet_v2", "imagenet_sketch", "imagenet_r", "imagenet_a")
RAW = get_paths().raw_results_root
LOG = ROOT / "results" / "logs" / "after_imagenet.log"


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{msg} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(msg, flush=True)


def ready_test(m: str) -> bool:
    return all(is_done(m, "imagenet:pool", v) for v in range(n_versions())) and \
        all(is_done(m, src, 0) for _, src in test_conditions("imagenet"))


def ready_nat(m: str) -> bool:
    return all(is_done(m, n, 0) for n in NATURALS)


class AllDone:
    """Marker standing for 'every re-split of this model has a run of `kind`' (exists() like a Path)."""

    def __init__(self, m: str, kind: str = "frontier"):
        self.m, self.kind = m, kind

    def exists(self) -> bool:
        return all((RAW / self.kind / self.m / f"r{r}" / ".ok").exists() for r in RESPLITS)


def active_single(m: str, r: int) -> bool:
    """A single-re-split frontier process (started before the multi runner existed) is still running."""
    return subprocess.run(["pgrep", "-f", f"[t]racecal frontier {m} imagenet {r}$"], capture_output=True).returncode == 0


MULTI_KIND = {"frontier-multi": "frontier", "dense-check-multi": "dense_check", "nonempty-multi": "nonempty"}


def is_locked(run_dir: Path) -> bool:
    """True if another process holds the RunLock of this output (tracecal.utils.io.RunLock)."""
    import fcntl

    lp = run_dir.with_name(run_dir.name + ".lock")
    if not lp.exists():
        return False
    with open(lp, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def multi_cmd(cmd: str, m: str) -> str:
    kind = MULTI_KIND[cmd]
    rs = [r for r in RESPLITS if not (RAW / kind / m / f"r{r}" / ".ok").exists()
          and not is_locked(RAW / kind / m / f"r{r}") and not (kind == "frontier" and active_single(m, r))]
    return f"uv run tracecal {cmd} {m} imagenet --resplits {','.join(map(str, rs))}" if rs else ""


def tasks():
    out = []
    for m in MODELS:
        # H2: one process per model reads each test condition once for all pending re-splits
        out.append((0, f"frontier-multi {m}", AllDone(m), lambda m=m: ready_test(m)))
        # secondary (after H2): one process per model and analysis; mixed streams read few rows, so per re-split
        out.append((3, f"dense-check-multi {m}", AllDone(m, "dense_check"), lambda m=m: AllDone(m).exists()))
        out.append((3, f"nonempty-multi {m}", AllDone(m, "nonempty"), lambda m=m: ready_test(m)))
        for r in RESPLITS:
            fr = RAW / "frontier" / m / f"r{r}"
            out += [
                (1, f"uv run tracecal frontier-naturals {m} {r}", fr / ".ok_naturals",
                 lambda m=m, fr=fr: (fr / ".ok").exists() and ready_nat(m)),
                (2, f"uv run tracecal mixed-stream {m} imagenet {r}", RAW / "mixed_stream" / m / f"r{r}" / ".ok",
                 lambda m=m: ready_test(m)),
            ]
    return out


def _run(cmd: str) -> tuple[str, int]:
    name = "_".join(cmd.split()[2:]).replace(":", "-").replace(",", "")
    lp = ROOT / "results" / "logs" / "analysis" / f"{name}_{time.strftime('%m%d-%H%M%S')}.log"
    lp.parent.mkdir(parents=True, exist_ok=True)
    full = f"source configs/ws/env.sh && export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 " \
           f"MKL_NUM_THREADS=4 && {cmd}"
    with open(lp, "w") as f:
        rc = subprocess.run(["bash", "-o", "pipefail", "-c", full], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return cmd, rc


def report(cmd: str, out: str) -> None:
    with open(ROOT / "results" / "logs" / out, "w") as f:
        subprocess.run(["bash", "-c", f"source configs/ws/env.sh && {cmd}"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    T = tasks()
    running: dict = {}
    failed: set[str] = set()
    gates_done = False
    log(f"imagenet pipeline start ({len(T)} tasks)")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        while True:
            for fut in [f for f in running if f.done()]:
                real, rc = fut.result()
                cmd = running.pop(fut)
                if rc != 0:
                    failed.add(cmd)
                log(f"rc={rc} {real}")
            todo = [t for t in T if not t[2].exists() and t[1] not in failed]
            busy = set(running.values())
            # H2 first: secondary analyses (priority >= 2) wait until every frontier run exists or has failed
            h2_pending = any(t[0] == 0 for t in todo)
            for pr, cmd, marker, ready in sorted(todo, key=lambda t: t[0]):
                if len(running) >= a.workers:
                    break
                if pr >= 2 and h2_pending:
                    continue
                if cmd in busy or not ready():
                    continue
                real = multi_cmd(cmd.split()[0], cmd.split()[1]) if cmd.split()[0] in MULTI_KIND else cmd
                if not real:
                    continue   # every pending re-split is being computed by an older single-re-split process
                fut = ex.submit(_run, real)
                running[fut] = cmd
                busy.add(cmd)
            if not gates_done and all((RAW / "frontier" / m / f"r{r}" / ".ok").exists() for m in MODELS for r in RESPLITS):
                report("uv run python scripts/report_gates_v2.py all", "gates_v2_all.log")
                log("gates done")
                gates_done = True
            if not todo and not running:
                break
            if not running and all(t[1] in failed for t in todo):
                log(f"stopping: {len(failed)} failed tasks remain")
                break
            time.sleep(60)
    if not gates_done:
        report("uv run python scripts/report_gates_v2.py all", "gates_v2_all.log")
        log("gates done (with missing runs)")
    for cmd, out in (("uv run python scripts/report_gates_v2.py h5", "gates_v2_h5.log"),
                     ("uv run python scripts/report_secondary.py imagenet", "secondary_imagenet.log"),
                     ("uv run python scripts/report_mixed_stream.py imagenet", "mixed_imagenet.log"),
                     ("uv run python scripts/report_dense_check.py imagenet", "dense_imagenet.log"),
                     ("uv run python scripts/report_nonempty.py imagenet", "nonempty_imagenet.log")):
        report(cmd, out)
    log(f"all done ({len(failed)} failed)")


if __name__ == "__main__":
    main()
