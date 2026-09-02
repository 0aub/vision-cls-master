#!/bin/bash
# Re-run the merged4 cross-validation with the corrected task mapping, once the
# merged4 grid driver has finished, then recompute and repackage.
set -uo pipefail
cd "$(dirname "$0")"
while ps -eo cmd | grep -q "[b]ench_merged4_full"; do sleep 60; done
echo "[cv] grid driver finished $(date +%H:%M:%S)"
CPU() { ./bench-cpu.sh "$@" 2>&1 | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com"; }
RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
for m in efficientnet_v2_s swin_s efficientnet_b0; do
  T=$(./bench-cpu.sh python3 bench/tier_of.py "$m" 2>/dev/null | sed -n 's/^TIER\t//p')
  echo "### CV merged4 $m  $(date +%H:%M:%S)"
  ./bench.sh python3 bench/cv.py --model "$m" --task merged4 $(rec "${T:-tier2-classic-cnn}") 2>&1 \
    | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license" | tail -8
done
echo "### CV merged4 dinov2_vitb14 LoRA  $(date +%H:%M:%S)"
./bench.sh python3 bench/cv.py --model dinov2_vitb14 --source hub-dinov2 --train_mode lora \
    --task merged4 --name dinov2_vitb14_lora $(rec tier5-foundation) 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache" | tail -8
CPU python3 bench/cv_recompute.py | tail -20
CPU python3 bench/report.py --phase FINAL
CPU python3 bench/package.py --out bench-results.zip
echo "CV MERGED4 CORRECTED $(date -Iseconds)" >> log/bench-progress.txt
echo "DONE $(date +%H:%M:%S)"
