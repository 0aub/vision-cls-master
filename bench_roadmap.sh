#!/bin/bash
# Roadmap execution: A1 (full masks) -> Track C (attention slice) -> A4 (Ulcer-only
# copy-paste). Keeps the GPU busy while A1 runs on CPU.
set -uo pipefail
cd "$(dirname "$0")"
GPU() { ./bench.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache" \
  | grep -vE "^  ep +[0-9]+/" | tail -8; }
CPU() { ./bench-cpu.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }

# ---- Track C: 15 attention modules on the two best CNNs, merged4 --------------
# Pre-registered as a sensitivity analysis: expected to move accuracy by less
# than the fold-to-fold variance. Reported either way.
stamp "TRACK C start"
R3=$(rec tier3-efficient-cnn); R2=$(rec tier2-classic-cnn)
for att in se_layer cbam bam eca simam coordinate_attention triplet_attention \
           gc_module srm sk_layer lct gct double_attention pam cam; do
  echo "### [attn] efficientnet_v2_s + $att  $(date +%H:%M:%S)"
  GPU python3 bench/train_dl.py --model efficientnet_v2_s --task merged4 \
      --attention "$att" --tier trackC-attention --protocol tuned \
      --name "efficientnet_v2_s__att_${att}" $R3
done
for att in se_layer cbam eca simam coordinate_attention; do
  echo "### [attn] densenet121 + $att  $(date +%H:%M:%S)"
  GPU python3 bench/train_dl.py --model densenet121 --task merged4 \
      --attention "$att" --tier trackC-attention --protocol tuned \
      --name "densenet121__att_${att}" $R2
done
stamp "TRACK C end"
CPU python3 bench/report.py --phase trackC
CPU python3 bench/package.py --out bench-results-trackC.zip --phase trackC
echo "TRACK C DONE $(date -Iseconds)" >> log/bench-progress.txt
