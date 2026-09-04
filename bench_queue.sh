#!/bin/bash
# Long-running work queue. Every step self-skips if its output exists, so the
# queue is restartable and never repeats finished work. Deliberately long: the
# machine should not run dry.
set -uo pipefail
cd "$(dirname "$0")"
export BENCH_MASKDIR=log/bench-lesion-masks-v21
G() { ./bench.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache|unauthenticated" \
  | grep -vE "^  ep +[0-9]+/" | tail -8; }
P() { ./bench-cpu.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not|Container Toolkit|docs.nvidia.com" | tail -8; }
say() { echo; echo "########## $* :: $(date '+%m-%d %H:%M:%S')"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }

# =============== A2: detection ===============================================
stamp "A2 detection start"
for task in merged4 5class; do
  for m in retinanet fcos fasterrcnn; do
    say "DETECT $m / $task"
    G python3 bench/detect.py --model "$m" --task "$task" --epochs 30
  done
done
stamp "A2 detection end"
P python3 bench/detect_report.py || true
P python3 bench/report.py --phase A2
P python3 bench/package.py --out bench-results-detect.zip --phase A2
echo "A2 DETECTION DONE $(date -Iseconds)" >> log/bench-progress.txt

# =============== merged4 CV for the remaining tier winners ===================
stamp "merged4 CV extra start"
R2=$(rec tier2-classic-cnn); R3=$(rec tier3-efficient-cnn); R4=$(rec tier4-transformer)
for m in convnext_tiny densenet121 vgg16; do
  T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
  say "CV merged4 $m"
  G python3 bench/cv.py --model "$m" --task merged4 $(rec "${T:-tier2-classic-cnn}")
done
say "CV merged4 vit_b_16"
G python3 bench/cv.py --model vit_b_16 --task merged4 $R4
say "CV merged4 dinov2_vits14 LoRA"
G python3 bench/cv.py --model dinov2_vits14 --source hub-dinov2 --train_mode lora \
    --task merged4 --name dinov2_vits14_lora $(rec tier5-foundation)
stamp "merged4 CV extra end"
P python3 bench/cv_recompute.py

# =============== attention: cross-validate the two that looked best ==========
stamp "Track C CV start"
for att in simam srm; do
  say "CV merged4 efficientnet_v2_s + $att"
  G python3 bench/cv.py --model efficientnet_v2_s --task merged4 \
      --attention "$att" --name "efficientnet_v2_s_att_${att}" $R3
done
stamp "Track C CV end"
P python3 bench/cv_recompute.py

# =============== refresh every downstream artefact ===========================
P python3 bench/study_level.py
P python3 bench/ensemble.py
P python3 bench/trust.py
P python3 bench/stats.py --bootstrap --mcnemar
P python3 bench/leak_audit.py
P python3 bench/figures.py
P python3 bench/report.py --phase FINAL
P python3 bench/package.py --out bench-results.zip
echo "QUEUE COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
say "QUEUE COMPLETE"
