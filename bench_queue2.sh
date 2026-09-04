#!/bin/bash
# Queue 2. Deep, and every step self-skips when its output exists.
set -uo pipefail
cd "$(dirname "$0")"
export BENCH_MASKDIR=log/bench-lesion-masks-v21
G() { ./bench.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache|unauthenticated" \
  | grep -vE "^  ep +[0-9]+/" | tail -8; }
P() { ./bench-cpu.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not|Container Toolkit|docs.nvidia.com" | tail -10; }
say() { echo; echo "########## $* :: $(date '+%m-%d %H:%M:%S')"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
R=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$R" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
R2=$(rec tier2-classic-cnn); R3=$(rec tier3-efficient-cnn)
R4=$(rec tier4-transformer); R5=$(rec tier5-foundation)

# ---- 1. detection: longer schedule, and the patient-grouped CV that is the
#         pre-registered Track B trigger --------------------------------------
stamp "detect long start"
for task in merged4 5class; do
  for m in retinanet fcos fasterrcnn; do
    say "DETECT-100 $m / $task"
    G python3 bench/detect.py --model "$m" --task "$task" --epochs 100 --name "${m}_e100"
  done
done
stamp "detect long end"
P python3 bench/detect_report.py

# ---- 2. merged4 CV across the whole grid, not just the winners --------------
stamp "cv sweep start"
for m in resnet50 resnet152 densenet201 efficientnet_b0 efficientnet_b7 alexnet googlenet \
         mobilenet_v3_large shufflenet_v2_x1_0 swin_t maxvit_t; do
  T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
  say "CV merged4 $m"
  G python3 bench/cv.py --model "$m" --task merged4 $(rec "${T:-tier2-classic-cnn}")
done
P python3 bench/cv_recompute.py

# ---- 3. same sweep on 5class, so both tasks have full CV coverage -----------
for m in convnext_tiny densenet121 resnet50 swin_t maxvit_t vit_b_16 mobilenet_v3_large; do
  T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
  say "CV 5class $m"
  G python3 bench/cv.py --model "$m" --task 5class $(rec "${T:-tier2-classic-cnn}")
done
stamp "cv sweep end"
P python3 bench/cv_recompute.py

# ---- 4. binary CV coverage --------------------------------------------------
for m in densenet201 vgg16 swin_s efficientnet_v2_s; do
  T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
  say "CV binary $m"
  G python3 bench/cv.py --model "$m" --task binary $(rec "${T:-tier2-classic-cnn}")
done
P python3 bench/cv_recompute.py

# ---- 5. refresh everything --------------------------------------------------
P python3 bench/study_level.py
P python3 bench/ensemble.py
P python3 bench/trust.py
P python3 bench/stats.py --bootstrap --mcnemar
P python3 bench/figures.py
P python3 bench/report.py --phase FINAL
P python3 bench/package.py --out bench-results.zip
echo "QUEUE2 COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
say "QUEUE2 COMPLETE"
