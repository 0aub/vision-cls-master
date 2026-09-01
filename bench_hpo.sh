#!/bin/bash
# Per-tier hyper-parameter selection sweep.
#
# One representative model per tier, six recipes each, 50 epochs, selected on
# VALIDATION macro F1. The uniform protocol (Adam 1e-4, wd 0, no warmup, flips
# only, plain CE) drives ViT-B/16 to its best epoch at 6 of 100 and then
# degrades it for 94 more, so the grid is measuring the protocol as much as the
# architecture. This sweep fixes a recipe per tier; bench_tuned.sh then re-runs
# the whole grid with it.
set -uo pipefail
cd "$(dirname "$0")"
R() { # R <model> <tier> <name-suffix> <extra args...>
  local m="$1" tier="$2" tag="$3"; shift 3
  ./bench.sh python3 bench/train_dl.py --model "$m" --task 5class --epochs 50 \
      --tier hpo-sweep --protocol sweep --name "hpo_${m}__${tag}" "$@" 2>&1 \
    | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache" \
    | grep -vE "^  ep +[0-9]+/" | tail -9
}
sweep() { # sweep <model> <tier> [extra source args...]
  local m="$1" tier="$2"; shift 2
  echo "===== sweeping $m ($tier)  $(date +%H:%M:%S)"
  R "$m" "$tier" base            "$@" --optimizer adam  --lr 1e-4
  R "$m" "$tier" aug             "$@" --optimizer adam  --lr 1e-4 --aug strong
  R "$m" "$tier" awd3e5_light    "$@" --optimizer adamw --lr 3e-5 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1
  R "$m" "$tier" awd3e5_strong   "$@" --optimizer adamw --lr 3e-5 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1 --aug strong
  R "$m" "$tier" awd1e5_strong   "$@" --optimizer adamw --lr 1e-5 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1 --aug strong
  R "$m" "$tier" awd1e4_strong   "$@" --optimizer adamw --lr 1e-4 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1 --aug strong
}

sweep vit_b_16      tier4-transformer
sweep convnext_tiny tier3-efficient-cnn
sweep resnet50      tier2-classic-cnn
# tier 5: LoRA adapters are randomly initialised and tiny, so they tolerate -
# and usually need - a much larger learning rate than a full fine-tune.
echo "===== sweeping dinov2_vitb14 LoRA (tier5-foundation)  $(date +%H:%M:%S)"
for lr in 1e-4 3e-4 1e-3; do
  for a in light strong; do
    ./bench.sh python3 bench/train_dl.py --model dinov2_vitb14 --source hub-dinov2 \
        --train_mode lora --task 5class --epochs 50 --tier hpo-sweep --protocol sweep \
        --name "hpo_dinov2_vitb14_lora__${lr}_${a}" --optimizer adamw --lr "$lr" \
        --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1 --aug "$a" 2>&1 \
      | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache" \
      | grep -vE "^  ep +[0-9]+/" | tail -9
  done
done
echo "HPO SWEEP DONE $(date +%H:%M:%S)"
