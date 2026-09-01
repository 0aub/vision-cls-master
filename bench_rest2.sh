#!/bin/bash
# Master chain, re-sequenced around the hyper-parameter work.
#   finish Phase B -> HPO sweep (already running when this starts) -> recipe
#   selection -> tuned grid (Protocol B) -> C -> D -> statistics -> final zip.
set -uo pipefail
cd "$(dirname "$0")"
CPU() { ./bench-cpu.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
waitfor() { while ps -eo cmd | grep -qE "$1"; do sleep 60; done; echo "[chain] $2 done $(date +%H:%M:%S)"; }
pkg() { CPU python3 bench/report.py --phase "$1"; CPU python3 bench/figures.py
        CPU python3 bench/package.py --out "$2" --phase "$1"
        echo "$3 $(date -Iseconds)" >> log/bench-progress.txt; }

echo "[chain2] started $(date -Iseconds)"
waitfor "bench_hpo[.]sh" "HPO sweep"
stamp "HPO SWEEP end"
CPU python3 bench/hpo_select.py | tail -40

./bench_phase.sh B                       # finish whatever Phase B still owes
stamp "PHASE B end"
pkg B bench-results-phaseB.zip "PHASE B PACKAGED"

./bench_tuned.sh
stamp "TUNED GRID end"
pkg A-tuned bench-results-phaseA-tuned.zip "TUNED GRID PACKAGED"

./bench_phase.sh C
stamp "PHASE C end"
pkg C bench-results-phaseC.zip "PHASE C PACKAGED"

./bench_phase.sh D
stamp "PHASE D end"
pkg D bench-results-phaseD.zip "PHASE D PACKAGED"

./bench_phase.sh S
stamp "STATS end"
echo "ALL TRAINING COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
echo "[chain2] done $(date -Iseconds)"
