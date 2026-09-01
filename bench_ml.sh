#!/bin/bash
# Tier 1: the 10 classical models. Raw 224 pixels (Phase A) or cached embeddings
# (Phase B). CPU-only container so it runs alongside the GPU grid.
#   ./bench_ml.sh raw                    both tasks, library defaults
#   ./bench_ml.sh embed:dinov2_vitb14    both tasks, val-selected grids
#
# svm goes LAST and in its own lane: on 150,528 raw dimensions libsvm's SMO has
# no iteration bound and takes hours, and there is no reason for it to hold up
# the other nine models. A run already in flight from an earlier invocation is
# waited on rather than duplicated.
set -uo pipefail
cd "$(dirname "$0")"
FEATURES="${1:-raw}"
SELECT=""; [ "$FEATURES" != "raw" ] && SELECT="--select"
TIER="tier1-classical"
FAST=(logistic_regression decision_tree random_forest knn naive_bayes adaboost lda qda mlp)
SENTINEL="log/.ml-$(echo "$FEATURES" | tr ':' '_')-done"
rm -f "$SENTINEL"

run_one() {
  local m="$1" task="$2"
  echo "### [ML/$task] $m on $FEATURES  $(date +%H:%M:%S)"
  ./bench-cpu.sh python3 bench/train_ml.py --model "$m" --task "$task" \
      --features "$FEATURES" --tier "$TIER" $SELECT 2>&1 \
    | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com" \
    | tail -12
}

for task in 5class binary; do
  for m in "${FAST[@]}"; do run_one "$m" "$task"; done
done

# The sentinel fires BEFORE the svm lane: the phase chain must not sit idle
# waiting for a single tier-1 cell whose libsvm SMO has no iteration bound. svm
# results land in the report and the final zip whenever they arrive, because
# both are regenerated at every later phase boundary.
touch "$SENTINEL"
echo "ML FAST LANE DONE ($FEATURES) $(date +%H:%M:%S)"

# `docker ps --format {{.Command}}` prints the image ENTRYPOINT, not the Cmd, so
# it never matches; inspect each container's Cmd instead. Getting this wrong once
# started a second svm on the same cell.
in_flight() {
  local c cmd
  for c in $(docker ps -q); do
    cmd=$(docker inspect -f '{{.Config.Cmd}}' "$c" 2>/dev/null)
    case "$cmd" in *train_ml*) return 0;; esac
  done
  return 1
}
while in_flight; do
  echo "### waiting for an in-flight classical run to finish  $(date +%H:%M:%S)"
  sleep 120
done
for task in 5class binary; do run_one svm "$task"; done
echo "ML DONE ($FEATURES) $(date +%H:%M:%S)"
