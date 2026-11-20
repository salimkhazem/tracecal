#!/usr/bin/env bash
# Pre-registration v6 (audits/gates_v6.yaml): camera-bank extraction for the 10 fresh models on GPUs 0,1,2; the CPU
# analysis (frontier-v6) of each model starts as soon as its extraction has finished (at most 3 at a time); then the
# report. Never kills anything.
set -uo pipefail
cd "$(dirname "$0")/.."
source configs/ws/env.sh
ulimit -n "$(ulimit -Hn)"
L=results/logs
uv run python scripts/gpu_queue.py configs/core/jobs_v6.json --gpus 0,1,2 &
QPID=$!
MODELS="in_deit_s in_effnet_b0 in_regnety16 in_convnext_s in_vit_s16 in_densenet121 in_mnv3_l in_deit3_s in_maxvit_t in_mixer_b16"
for m in $MODELS; do
  until [ -f $L/v6x_$m.ok ]; do sleep 120; done
  while [ "$(pgrep -fc "[t]racecal frontier-v6")" -ge 3 ]; do sleep 60; done
  (CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 nice -n 5 uv run tracecal frontier-v6 $m > $L/v6a_$m.log 2>&1 \
     && echo ok > $L/v6a_$m.ok) &
  sleep 5
done
wait
CUDA_VISIBLE_DEVICES= uv run python scripts/report_gates_v6.py > $L/report_v6.log 2>&1
echo "v6 done $(date)"
