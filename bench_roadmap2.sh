#!/bin/bash
# C2 at full scope (100% mask coverage) -> A4 Ulcer-only copy-paste, CV-selected.
set -uo pipefail
cd "$(dirname "$0")"
export BENCH_MASKDIR=log/bench-lesion-masks-v21
GPUE() { ./bench.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache" \
  | grep -vE "^  ep +[0-9]+/" | tail -12; }
CPUE() { ./bench-cpu.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
R3=$(rec tier3-efficient-cnn)

# ---- C2 at full scope --------------------------------------------------------
stamp "C2 full-scope start"
M1=$(./bench-cpu.sh python3 bench/pick.py --task merged4 --deep --source torchvision \
      --needs_ckpt --top 2 2>/dev/null | sed -n 's/^PICK\t//p' | tr '\n' ' ')
DV=$(./bench-cpu.sh python3 bench/pick.py --task merged4 --tier tier5-foundation \
      --needs_ckpt --top 1 2>/dev/null | sed -n 's/^PICK\t//p')
echo "### C2 full scope: $M1 | $DV  $(date +%H:%M:%S)"
GPUE python3 bench/cams.py --task merged4 --models $M1 --dinov2 "$DV" \
     --panels 8 --panel_split test --force
CPUE python3 bench/cam_geometry.py | tail -12
stamp "C2 full-scope end"

# ---- A4: copy-paste for ULCER ONLY, selected on patient-grouped CV -----------
stamp "A4 start"
CPUE python3 bench/copypaste.py --classes Ulcer --per_class 500 \
     --out data/synthetic/V8-CP-ulcer --force
echo "### A4 train  $(date +%H:%M:%S)"
GPUE python3 bench/train_dl.py --model efficientnet_v2_s --task 5class \
     --extra_train_dir data/synthetic/V8-CP-ulcer --tier phaseE-copypaste \
     --protocol tuned --name efficientnet_v2_s__cp_ulcer $R3
echo "### A4 cross-validation  $(date +%H:%M:%S)"
GPUE python3 bench/cv.py --model efficientnet_v2_s --task 5class \
     --name efficientnet_v2_s_cp_ulcer --extra_train_dir data/synthetic/V8-CP-ulcer $R3
stamp "A4 end"
CPUE python3 bench/cv_recompute.py | tail -12
CPUE python3 bench/study_level.py | tail -14
CPUE python3 bench/report.py --phase roadmap
CPUE python3 bench/package.py --out bench-results-roadmap.zip --phase roadmap
echo "ROADMAP2 DONE $(date -Iseconds)" >> log/bench-progress.txt
