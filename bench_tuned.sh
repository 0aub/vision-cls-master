#!/bin/bash
# Protocol B: re-run the whole supervised grid with the per-tier recipe that
# bench/hpo_select.py chose on validation macro F1.
#   ./bench_tuned.sh            both tasks, every architecture
set -uo pipefail
cd "$(dirname "$0")"
RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
[ -z "$RECIPES" ] && { echo "no recipes: run ./bench_hpo.sh first"; exit 1; }
recipe_for() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }

T2="tier2-classic-cnn"; T3="tier3-efficient-cnn"; T4="tier4-transformer"
declare -A TIER=(
  [efficientnet_b0]=$T2 [resnet50]=$T2 [densenet201]=$T2 [alexnet]=$T2 [vgg16]=$T2
  [googlenet]=$T2 [resnet152]=$T2 [densenet121]=$T2 [efficientnet_b7]=$T2
  [efficientnet_v2_s]=$T3 [convnext_tiny]=$T3 [mobilenet_v3_large]=$T3
  [shufflenet_v2_x1_0]=$T3
  [vit_b_16]=$T4 [swin_t]=$T4 [swin_s]=$T4 [maxvit_t]=$T4
)
ORDER="efficientnet_b0 resnet50 densenet201 efficientnet_v2_s convnext_tiny \
mobilenet_v3_large shufflenet_v2_x1_0 vit_b_16 swin_t swin_s maxvit_t \
alexnet vgg16 googlenet resnet152 densenet121 efficientnet_b7"

echo "recipes in force:"; echo "$RECIPES"
for task in 5class binary; do
  for m in $ORDER; do
    t="${TIER[$m]}"; r=$(recipe_for "$t")
    echo "### [tuned/$task] $m  [$t] $r  $(date +%H:%M:%S)"
    # --epochs comes from the recipe: the sweep validated a 50-epoch cosine
    # anneal, and re-annealing the same recipe over 100 epochs is a different
    # trajectory, so it would not reproduce what validation chose.
    ./bench.sh python3 bench/train_dl.py --model "$m" --task "$task" \
        --batch_size 16 --image_size 224 --tier "$t" \
        --protocol tuned --name "${m}__tuned" $r 2>&1 \
      | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license" \
      | grep -vE "^  ep +[0-9]+/" | tail -10
  done
done
# tier 5: the LoRA runs get their own recipe
r5=$(recipe_for tier5-foundation)
for task in 5class binary; do
  for bb in dinov2_vits14 dinov2_vitb14; do
    echo "### [tuned/$task] ${bb} LoRA  $r5  $(date +%H:%M:%S)"
    ./bench.sh python3 bench/train_dl.py --model "$bb" --source hub-dinov2 \
        --train_mode lora --task "$task" --tier tier5-foundation \
        --protocol tuned --name "${bb}_lora__tuned" $r5 2>&1 \
      | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache" \
      | grep -vE "^  ep +[0-9]+/" | tail -10
  done
done
echo "TUNED GRID DONE $(date +%H:%M:%S)"
