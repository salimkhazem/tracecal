#!/usr/bin/env python
"""Minimal multi-GPU job queue: one job per GPU at a time, phases run sequentially.

Usage: uv run python scripts/gpu_queue.py JOBS.json --gpus 0,1,2
JOBS.json: {"phases": [[{"name": ..., "cmd": ...}, ...], ...]}
Each job's stdout/stderr goes to results/logs/<name>.log; status lines are appended to results/logs/queue.jsonl.
A job whose log ends with a success marker file (<name>.ok) is skipped on re-runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run_phase(jobs: list[dict], gpus: list[int], log_dir: Path) -> list[dict]:
    pending = [j for j in jobs if not (log_dir / f"{j['name']}.ok").exists()]
    running: dict[int, tuple] = {}
    done: list[dict] = []
    while pending or running:
        for g in gpus:
            if g not in running and pending:
                job = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                log = open(log_dir / f"{job['name']}.log", "w")
                p = subprocess.Popen(["bash", "-o", "pipefail", "-c", job["cmd"]], env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
                running[g] = (p, job, log, time.time())
                _status(log_dir, {"event": "start", "name": job["name"], "gpu": g})
        for g, (p, job, log, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                log.close()
                rec = {"event": "end", "name": job["name"], "gpu": g, "rc": rc, "seconds": round(time.time() - t0, 1)}
                if rc == 0:
                    (log_dir / f"{job['name']}.ok").write_text("ok")
                _status(log_dir, rec)
                done.append(rec)
                del running[g]
        time.sleep(3)
    return done


def _status(log_dir: Path, rec: dict) -> None:
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_dir / "queue.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="0,1,2")
    args = ap.parse_args()
    spec = json.loads(Path(args.jobs).read_text())
    log_dir = Path(os.environ.get("RESULTS_ROOT", "results")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    for i, phase in enumerate(spec["phases"]):
        res = run_phase(phase, gpus, log_dir)
        failed = [r["name"] for r in res if r["rc"] != 0]
        _status(log_dir, {"event": "phase_end", "phase": i, "failed": failed})
        if failed and spec.get("stop_on_failure", True):
            raise SystemExit(f"phase {i} failed: {failed}")


if __name__ == "__main__":
    main()
