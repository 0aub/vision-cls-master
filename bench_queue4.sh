#!/bin/bash
# Queue 4, driven by what queue 3 found: resolution is the strongest lever, and
# crops fail when mixed into whole-frame training.
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
R3=$(rec tier3-efficient-cnn); R5=$(rec tier5-foundation); R2=$(rec tier2-classic-cnn)

# ---- 1. push resolution further; it was the biggest lever found -------------
stamp "res push start"
for sz in 512 576; do
  for m in convnext_tiny efficientnet_v2_s; do
    say "RES $m @${sz} / merged4"
    G python3 bench/train_dl.py --model "$m" --task merged4 --image_size $sz \
        --batch_size 4 --tier "res-${sz}" --protocol tuned --name "${m}__res${sz}" $R3
  done
done
# DINOv2 needs multiples of 14: 448 = 32x14, 518 = 37x14
for sz in 448 518; do
  for task in merged4 5class; do
    say "RES dinov2_vitb14 LoRA @${sz} / $task"
    G python3 bench/train_dl.py --model dinov2_vitb14 --source hub-dinov2 \
        --train_mode lora --task $task --image_size $sz --batch_size 4 \
        --tier tier5-foundation --protocol tuned \
        --name "dinov2_vitb14_lora__res${sz}" $R5
  done
done
for sz in 448; do
  for m in convnext_tiny efficientnet_v2_s densenet121 resnet50; do
    say "RES $m @${sz} / 5class"
    G python3 bench/train_dl.py --model "$m" --task 5class --image_size $sz \
        --batch_size 8 --tier "res-${sz}" --protocol tuned --name "${m}__res${sz}" $R3
  done
done
stamp "res push end"

# ---- 2. cross-validate the resolution winners under the pre-registered rule -
stamp "res CV start"
for m in convnext_tiny efficientnet_v2_s; do
  say "CV merged4 $m @448"
  G python3 bench/cv.py --model "$m" --task merged4 --image_size 448 --batch_size 8 \
      --name "${m}_res448" $R3
done
stamp "res CV end"
P python3 bench/cv_recompute.py

# ---- 3. bigger / longer detectors ------------------------------------------
stamp "detect big start"
for m in yolo11m yolo11n; do
  say "DET $m / merged4 @1024"
  G python3 bench/yolo_train.py --model "$m" --task merged4 --epochs 300 --imgsz 1024 --batch 8
done
say "DET rtdetr-l / merged4 @640 e300"
G python3 bench/yolo_train.py --model rtdetr-l --task merged4 --epochs 300 --imgsz 640 --name rtdetr-l_640_e300
say "DET yolo11m / 5class @1024"
G python3 bench/yolo_train.py --model yolo11m --task 5class --epochs 300 --imgsz 1024 --batch 8
stamp "detect big end"
P python3 bench/detect_report.py

# ---- 4. refresh -------------------------------------------------------------
P python3 bench/study_level.py
P python3 bench/ensemble.py
P python3 bench/trust.py
P python3 bench/stats.py --bootstrap --mcnemar
P python3 bench/figures.py
P python3 bench/report.py --phase FINAL
P python3 bench/package.py --out bench-results.zip
echo "QUEUE4 COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
say "QUEUE4 COMPLETE"
