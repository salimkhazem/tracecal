#!/usr/bin/env bash
# E2: weak/strong restorer banks for ONE fresh seed per CIFAR setting (seed 5; scope reduced from 16 to 4 models on
# 2026-10-23 for compute, before any E2 number existed), then the analysis.
set -uo pipefail
cd "$(dirname "$0")/.."
source configs/ws/env.sh
MODELS="c10_resnet18_s5 c10_vits16_s5 c100_resnet18_s5 c100_vits16_s5"
for m in $MODELS; do
  CUDA_VISIBLE_DEVICES=0 nice -n 5 uv run tracecal extract-magnitude $m --workers 4 || echo "FAIL $m"
done
printf "%s\n" $MODELS | CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 xargs -P 2 -I{} nice -n 12 uv run python scripts/run_magnitude_ablation.py {}
echo "e2 done $(date)"
