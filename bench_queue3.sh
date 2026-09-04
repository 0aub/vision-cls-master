#!/bin/bash
# Queue 3: the two things the benchmark had not tested - SOTA detectors, and
# input representation for classification. Every step self-skips when done.
set -uo pipefail
cd "$(dirname "$0")"
export BENCH_MASKDIR=log/bench-lesion-masks-v21
G() { ./bench.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 YOLO_CONFIG_DIR=/app/.cache/ultra "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache|unauthenticated" \
  | grep -vE "^  ep +[0-9]+/" | tail -10; }
P() { ./bench-cpu.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not|Container Toolkit|docs.nvidia.com" | tail -10; }
say() { echo; echo "########## $* :: $(date '+%m-%d %H:%M:%S')"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
Rr=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$Rr" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
R2=$(rec tier2-classic-cnn); R3=$(rec tier3-efficient-cnn); R5=$(rec tier5-foundation)

# ---- 1. SOTA detectors ------------------------------------------------------
stamp "SOTA detect start"
P python3 bench/yolo_export.py --task merged4
P python3 bench/yolo_export.py --task 5class
for m in yolo11s yolov8s yolo12s yolo11m rtdetr-l yolov8n yolo11n; do
  say "DET $m / merged4 @640"
  G python3 bench/yolo_train.py --model "$m" --task merged4 --epochs 150 --imgsz 640
done
for m in yolo11s rtdetr-l; do
  say "DET $m / 5class @640"
  G python3 bench/yolo_train.py --model "$m" --task 5class --epochs 150 --imgsz 640
done
say "DET yolo11s / merged4 @1024"
G python3 bench/yolo_train.py --model yolo11s --task merged4 --epochs 150 --imgsz 1024 --batch 8
stamp "SOTA detect end"
P python3 bench/detect_report.py

# ---- 2. resolution: the untested axis for classification --------------------
stamp "resolution sweep start"
for sz in 320 384 448; do
  for m in efficientnet_v2_s convnext_tiny densenet121; do
    T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
    say "RES $m @${sz} / merged4"
    G python3 bench/train_dl.py --model "$m" --task merged4 --image_size $sz \
        --batch_size 8 --tier "res-${sz}" --protocol tuned \
        --name "${m}__res${sz}" $(rec "${T:-tier3-efficient-cnn}")
  done
done
for sz in 384 448; do
  say "RES efficientnet_v2_s @${sz} / 5class"
  G python3 bench/train_dl.py --model efficientnet_v2_s --task 5class --image_size $sz \
      --batch_size 8 --tier "res-${sz}" --protocol tuned \
      --name "efficientnet_v2_s__res${sz}" $R3
done
stamp "resolution sweep end"

# ---- 3. lesion-centred crops as extra training data ------------------------
stamp "crops start"
P python3 bench/lesion_crops.py --per_frame 2
for m in efficientnet_v2_s convnext_tiny; do
  for task in merged4 5class; do
    say "CROPS $m / $task"
    G python3 bench/train_dl.py --model "$m" --task $task \
        --extra_train_dir data/synthetic/V8-crops --tier crops --protocol tuned \
        --name "${m}__crops" $R3
  done
done
say "CROPS + res384 efficientnet_v2_s / merged4"
G python3 bench/train_dl.py --model efficientnet_v2_s --task merged4 --image_size 384 \
    --batch_size 8 --extra_train_dir data/synthetic/V8-crops --tier crops \
    --protocol tuned --name "efficientnet_v2_s__crops_res384" $R3
stamp "crops end"

# ---- 4. bigger foundation backbone -----------------------------------------
say "DINOv2 ViT-L/14 LoRA"
G python3 bench/embed.py --name dinov2_vitl14 --source hub-dinov2 --feature_mode cls
for task in merged4 5class; do
  G python3 bench/train_dl.py --model dinov2_vitl14 --source hub-dinov2 --train_mode lora \
      --task $task --tier tier5-foundation --protocol tuned \
      --name "dinov2_vitl14_lora__tuned" $R5
  G python3 bench/foundation.py --mode dinov2-probe --backbone dinov2_vitl14 --task $task
done

# ---- 5. refresh -------------------------------------------------------------
P python3 bench/cv_recompute.py
P python3 bench/study_level.py
P python3 bench/ensemble.py
P python3 bench/trust.py
P python3 bench/stats.py --bootstrap --mcnemar
P python3 bench/figures.py
P python3 bench/report.py --phase FINAL
P python3 bench/package.py --out bench-results.zip
echo "QUEUE3 COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
say "QUEUE3 COMPLETE"
