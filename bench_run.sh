#!/bin/bash
# Sequential driver for the v2 deep grid. One docker invocation per run so a
# crash in one model cannot take the queue down; each run self-skips when its
# summary_test.csv already exists (addendum D1).
#   ./bench_run.sh A1        run one group
#   ./bench_run.sh A1 A2 A3  run several
set -uo pipefail
cd "$(dirname "$0")"

declare -A GROUP=(
  [A1]="efficientnet_b0 resnet50 densenet201"
  [A2]="efficientnet_v2_s convnext_tiny mobilenet_v3_large shufflenet_v2_x1_0"
  [A3]="vit_b_16 swin_t swin_s maxvit_t"
  [A4]="alexnet vgg16 googlenet resnet152 densenet121 efficientnet_b7"
)
declare -A TIER=(
  [A1]="tier2-classic-cnn" [A2]="tier3-efficient-cnn"
  [A3]="tier4-transformer" [A4]="tier2-classic-cnn"
)
# A2 mixes an efficient-CNN group; convnext_tiny is filed as efficient too.
LOGDIR=log; mkdir -p "$LOGDIR"

for g in "$@"; do
  models="${GROUP[$g]:-}"
  [ -z "$models" ] && { echo "unknown group $g"; exit 1; }
  for task in 5class binary; do
    for m in $models; do
      echo "### [$g/$task] $m  $(date +%H:%M:%S)"
      ./bench.sh python3 bench/train_dl.py --model "$m" --task "$task" \
          --epochs 100 --batch_size 16 --lr 1e-4 --image_size 224 \
          --tier "${TIER[$g]}" 2>&1 \
        | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license" \
        | grep -vE "^  ep +[0-9]+/100" | tail -12
    done
  done
done
echo "GROUPS DONE: $*  $(date +%H:%M:%S)"
