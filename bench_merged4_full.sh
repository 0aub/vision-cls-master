#!/bin/bash
# Complete merged4 as a first-class third task: the full deep grid under the
# per-tier tuned recipes, the tier-5 frozen/zero-shot regimes, the classical
# tier on cached embeddings, and patient-grouped cross-validation.
set -uo pipefail
cd "$(dirname "$0")"
GPU() { ./bench.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache|unauthenticated" \
  | grep -vE "^  ep +[0-9]+/" | tail -10; }
CPU() { ./bench-cpu.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com" | tail -10; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }

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

stamp "MERGED4 grid start"
for m in $ORDER; do
  t="${TIER[$m]}"; r=$(rec "$t")
  echo "### [merged4] $m [$t]  $(date +%H:%M:%S)"
  GPU python3 bench/train_dl.py --model "$m" --task merged4 --batch_size 16 \
      --image_size 224 --tier "$t" --protocol tuned --name "${m}__merged4" $r
done
R5=$(rec tier5-foundation)
for bb in dinov2_vits14 dinov2_vitb14; do
  echo "### [merged4] ${bb} LoRA  $(date +%H:%M:%S)"
  GPU python3 bench/train_dl.py --model "$bb" --source hub-dinov2 --train_mode lora \
      --task merged4 --tier tier5-foundation --protocol tuned \
      --name "${bb}_lora__merged4" $R5
done
# tier 5 without gradient descent
for bb in dinov2_vits14 dinov2_vitb14; do
  GPU python3 bench/foundation.py --mode dinov2-probe --backbone $bb --task merged4
  GPU python3 bench/foundation.py --mode dinov2-knn   --backbone $bb --task merged4
done
GPU python3 bench/foundation.py --mode biomedclip-zeroshot --task merged4
GPU python3 bench/foundation.py --mode biomedclip-probe    --task merged4
stamp "MERGED4 grid end"

# tier 1 on the embeddings that won the 5-class task
BEST=$(./bench-cpu.sh python3 bench/pick.py --task 5class --deep --source torchvision \
        --field arch --top 1 2>/dev/null | sed -n 's/^PICK\t//p')
for feat in "embed:dinov2_vitb14" "embed:${BEST}_ft_5class"; do
  for m in logistic_regression decision_tree random_forest knn naive_bayes adaboost lda qda mlp svm; do
    echo "### [ML/merged4] $m on $feat"
    CPU python3 bench/train_ml.py --model $m --task merged4 --features "$feat" \
        --select --tier tier1-classical
  done
done
stamp "MERGED4 classical end"

# patient-grouped CV on the merged task, same recipes
for m in efficientnet_v2_s swin_s efficientnet_b0; do
  T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
  GPU python3 bench/cv.py --model "$m" --task merged4 $(rec "${T:-tier2-classic-cnn}")
done
GPU python3 bench/cv.py --model dinov2_vitb14 --source hub-dinov2 --train_mode lora \
    --task merged4 --name dinov2_vitb14_lora $R5
stamp "MERGED4 cv end"

CPU python3 bench/leak_audit.py | tail -20
CPU python3 bench/trust.py
CPU python3 bench/stats.py --bootstrap --mcnemar
CPU python3 bench/figures.py
CPU python3 bench/report.py --phase FINAL
CPU python3 bench/package.py --out bench-results.zip
CPU python3 bench/package.py --out bench-results-merged4.zip --phase merged4
echo "MERGED4 COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
echo "ALL DONE $(date +%H:%M:%S)"
