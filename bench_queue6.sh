#!/bin/bash
# Queue 6, driven by queue 5's finding: resolution helps CNNs and hurts
# transformers, and timm's checkpoints are much stronger than torchvision's.
# So: push CNN resolution further, re-run the grid on timm weights, and
# cross-validate the new best rather than trusting one 226-frame split.
set -uo pipefail
cd "$(dirname "$0")"
export BENCH_MASKDIR=log/bench-lesion-masks-v21
G() { ./bench.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 YOLO_CONFIG_DIR=/app/.cache/ultra "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache|unauthenticated" \
  | grep -vE "^  ep +[0-9]+/" | tail -8; }
P() { ./bench-cpu.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not|Container Toolkit|docs.nvidia.com" | tail -10; }
say() { echo; echo "########## $* :: $(date '+%m-%d %H:%M:%S')"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
Rr=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$Rr" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
R3=$(rec tier3-efficient-cnn); R2=$(rec tier2-classic-cnn); R4=$(rec tier4-transformer)

# ---- 1. how far does CNN resolution go? -------------------------------------
stamp "res extend start"
for sz in 640 768; do
  for m in efficientnet_v2_s convnext_tiny; do
    say "RES $m @${sz} / merged4"
    G python3 bench/train_dl.py --model "$m" --task merged4 --image_size $sz \
        --batch_size 2 --tier "res-${sz}" --protocol tuned --name "${m}__res${sz}" $R3
  done
done
# more CNNs at the resolution that won
for m in efficientnet_b0 densenet201 resnet50 mobilenet_v3_large densenet121; do
  T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
  for task in merged4 5class; do
    say "RES $m @512 / $task"
    G python3 bench/train_dl.py --model "$m" --task $task --image_size 512 \
        --batch_size 4 --tier "res-512" --protocol tuned --name "${m}__res512" \
        $(rec "${T:-tier2-classic-cnn}")
  done
done
for m in efficientnet_v2_s convnext_tiny; do
  say "RES $m @512 / 5class"
  G python3 bench/train_dl.py --model "$m" --task 5class --image_size 512 \
      --batch_size 4 --tier "res-512" --protocol tuned --name "${m}__res512" $R3
done
stamp "res extend end"

# ---- 2. timm weights are stronger; re-run the tiers on them -----------------
stamp "timm weights start"
for m in convnext_tiny vit_s_16 deit3_s swin_s; do
  for task in merged4 5class; do
    say "TIMM-W $m @224 / $task"
    G python3 bench/train_dl.py --model "$m" --source timm --task $task \
        --image_size 224 --batch_size 8 --tier tier4-transformer --protocol tuned \
        --name "${m}__timmw224" $R4
  done
done
stamp "timm weights end"

# ---- 3. cross-validate the new best under the pre-registered rule -----------
stamp "best CV start"
for m in efficientnet_v2_s convnext_tiny; do
  say "CV merged4 $m @512"
  G python3 bench/cv.py --model "$m" --task merged4 --image_size 512 --batch_size 4 \
      --name "${m}_res512" $R3
done
say "CV merged4 vit_b_16 (timm) @224"
G python3 bench/cv.py --model vit_b_16 --source timm --task merged4 --image_size 224 \
    --batch_size 8 --name "vit_b_16_timm224" $R4
stamp "best CV end"
P python3 bench/cv_recompute.py

# ---- 4. refresh -------------------------------------------------------------
P python3 bench/study_level.py
P python3 bench/ensemble.py
P python3 bench/trust.py
P python3 bench/stats.py --bootstrap --mcnemar
P python3 bench/figures.py
P python3 bench/report.py --phase FINAL
P python3 bench/package.py --out bench-results.zip
echo "QUEUE6 COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
say "QUEUE6 COMPLETE"
