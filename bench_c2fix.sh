#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
G() { ./bench.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache"; }
C() { ./bench-cpu.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not|Container Toolkit|docs.nvidia.com"; }
M1=$(./bench-cpu.sh python3 bench/pick.py --task merged4 --deep --source torchvision \
      --needs_ckpt --top 2 2>/dev/null | sed -n 's/^PICK\t//p' | tr '\n' ' ')
DV=$(./bench-cpu.sh python3 bench/pick.py --task merged4 --tier tier5-foundation \
      --needs_ckpt --top 1 2>/dev/null | sed -n 's/^PICK\t//p')
echo "### C2 full scope (corrected): $M1 | $DV  $(date +%H:%M:%S)"
G python3 bench/cams.py --task merged4 --models $M1 --dinov2 "$DV" \
  --panels 8 --panel_split test --force | tail -22
C python3 bench/cam_geometry.py | tail -8
C python3 bench/report.py --phase C2fix
echo "C2FIX DONE $(date -Iseconds)" >> log/bench-progress.txt
