#!/usr/bin/env bash
# G_majority_vote for every model, each as soon as its ablation curves exist (CPU only, low priority).
set -uo pipefail
cd "$(dirname "$0")/.."
source configs/ws/env.sh
T=results/tables
CIFAR=$(for d in c10 c100; do for a in resnet18 vits16; do for s in 5 6 7 8; do echo ${d}_${a}_s$s; done; done; done)
printf "%s\n" $CIFAR | CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 xargs -P 2 -I{} nice -n 15 uv run python scripts/run_majority_vote.py {}
echo "cifar done $(date)"
for m in in_deit_s in_effnet_b0 in_regnety16 in_convnext_s in_vit_s16 in_densenet121 in_mnv3_l in_deit3_s in_maxvit_t \
         in_mixer_b16 in_vit_l16 in_convnext_l in_swin_b in_clip_l14 in_rn50_tv1 in_augmix_rn50 in_deepaug_rn50; do
  until [ -f $T/ablate_${m}_curves.npz ]; do sleep 600; done
  [ -f $T/majority_$m.json ] || CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 15 uv run python scripts/run_majority_vote.py $m
done
echo "majority done $(date)"
