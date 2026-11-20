#!/usr/bin/env bash
# Ablations v5c for every model in audits/ablations_v5c.yaml. Skips models whose result exists. Never uses a GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
source configs/ws/env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
L=results/logs; T=results/tables
run() { [ -f "$T/ablate_$1.json" ] || nice -n 12 uv run python scripts/run_ablations.py "$1" > "$L/ablate_$1.log" 2>&1; }
export -f run; export L T
# 1) CIFAR fresh seeds 5-8
for d in c10 c100; do for a in resnet18 vits16; do for s in 5 6 7 8; do echo "${d}_${a}_s${s}"; done; done; done \
  | xargs -P 4 -I{} bash -c 'run {}'
echo "cifar done $(date)"
# 2) the 10 fresh ImageNet models, once the severity/tail batch has finished
until [ "$(ls $T/sevtail_*.json 2>/dev/null | wc -l)" -ge 10 ]; do sleep 300; done
printf "%s\n" in_deit_s in_effnet_b0 in_regnety16 in_convnext_s in_vit_s16 in_densenet121 in_mnv3_l in_deit3_s \
  in_maxvit_t in_mixer_b16 | xargs -P 3 -I{} bash -c 'run {}'
echo "imagenet-10 done $(date)"
# 3) the 7 v5 models, each after its extraction+evaluation job ended
for m in in_vit_l16 in_convnext_l in_swin_b in_rn50_tv1 in_augmix_rn50 in_deepaug_rn50 in_clip_l14; do
  until grep "\"end\"" $L/queue.jsonl | grep -q "\"v5_$m\""; do sleep 600; done
  run "$m"
done
echo "all ablations done $(date)"
