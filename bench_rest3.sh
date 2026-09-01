#!/bin/bash
# Master chain, third revision: tuned grid at the epoch count the sweep
# validated, then C, D, statistics, the deferred SVM, a clean efficiency
# re-measurement and the final package.
set -uo pipefail
cd "$(dirname "$0")"
CPU() { ./bench-cpu.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
pkg() { CPU python3 bench/report.py --phase "$1"; CPU python3 bench/figures.py
        CPU python3 bench/package.py --out "$2" --phase "$1"
        echo "$3 $(date -Iseconds)" >> log/bench-progress.txt; }

echo "[chain3] started $(date -Iseconds)"
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
echo "ALL GPU WORK COMPLETE $(date -Iseconds)" >> log/bench-progress.txt

# nothing on the GPU from here: the deferred SVM gets the whole machine, then
# every cost column is re-taken under identical idle conditions
./bench_svm.sh
stamp "SVM lane end"
CPU python3 bench/remeasure.py --deep --classical
stamp "EFFICIENCY remeasure end"

CPU python3 bench/trust.py
CPU python3 bench/stats.py --bootstrap --mcnemar
CPU python3 bench/figures.py
CPU python3 bench/report.py --phase FINAL
CPU python3 bench/package.py --out bench-results.zip
echo "ALL PHASES COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
echo "[chain3] done $(date -Iseconds)"
