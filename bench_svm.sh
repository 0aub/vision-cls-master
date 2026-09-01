#!/bin/bash
# Trailing lane for the raw-pixel SVM. Waits out whatever classical run is
# already in flight, then fills in whichever svm cells are still missing.
set -uo pipefail
cd "$(dirname "$0")"
in_flight() {
  local c cmd
  for c in $(docker ps -q); do
    cmd=$(docker inspect -f '{{.Config.Cmd}}' "$c" 2>/dev/null)
    case "$cmd" in *train_ml*) return 0;; esac
  done
  return 1
}
while in_flight; do
  echo "### svm lane: waiting for the in-flight classical run  $(date +%H:%M:%S)"
  sleep 180
done
for task in 5class binary; do
  echo "### [ML/$task] svm on raw  $(date +%H:%M:%S)"
  ./bench-cpu.sh python3 bench/train_ml.py --model svm --task "$task" \
      --features raw --tier tier1-classical 2>&1 \
    | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com" \
    | tail -12
done
echo "SVM LANE DONE $(date +%H:%M:%S)"
