#!/bin/bash
# Chains everything after Phase A1 so the GPU never idles between phases.
# Each phase ends with a report regeneration and its partial zip.
set -uo pipefail
cd "$(dirname "$0")"
CPU() { ./bench-cpu.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
waitfor() { # waitfor <pattern-of-driver> <label>
  while ps -eo cmd | grep -qE "$1"; do sleep 60; done
  echo "[chain] $2 finished $(date +%H:%M:%S)"
}

echo "[chain] started $(date -Iseconds)"
waitfor "bench_run[.]sh A2"  "Phase A2-A4"
# the classical queue signals completion with a sentinel file, because its svm
# lane deliberately outlives the driver that started it
while [ ! -f log/.ml-raw-done ]; do sleep 60; done
echo "[chain] classical ML on raw pixels finished $(date +%H:%M:%S)"

stamp "PHASE A end"
CPU python3 bench/report.py --phase A
CPU python3 bench/figures.py
CPU python3 bench/package.py --out bench-results-phaseA.zip --phase A
echo "PHASE A PACKAGED $(date -Iseconds)" >> log/bench-progress.txt

./bench_phase.sh B
CPU python3 bench/report.py --phase B
CPU python3 bench/figures.py
CPU python3 bench/package.py --out bench-results-phaseB.zip --phase B
echo "PHASE B PACKAGED $(date -Iseconds)" >> log/bench-progress.txt

./bench_phase.sh C
CPU python3 bench/report.py --phase C
CPU python3 bench/figures.py
CPU python3 bench/package.py --out bench-results-phaseC.zip --phase C
echo "PHASE C PACKAGED $(date -Iseconds)" >> log/bench-progress.txt

./bench_phase.sh D
CPU python3 bench/report.py --phase D
CPU python3 bench/package.py --out bench-results-phaseD.zip --phase D
echo "PHASE D PACKAGED $(date -Iseconds)" >> log/bench-progress.txt

./bench_phase.sh S
CPU python3 bench/trust.py
CPU python3 bench/figures.py
CPU python3 bench/report.py --phase FINAL
CPU python3 bench/package.py --out bench-results.zip
echo "ALL PHASES COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
echo "[chain] done $(date -Iseconds)"
