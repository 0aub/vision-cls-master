#!/bin/bash
# Tier 1: the 10 classical models. Raw 224 pixels (Phase A) or cached embeddings
# (Phase B). CPU-only container so it can run alongside the GPU grid.
#   ./bench_ml.sh raw                      both tasks, library defaults
#   ./bench_ml.sh embed:dinov2_vitb14      both tasks, val-selected grids
set -uo pipefail
cd "$(dirname "$0")"
FEATURES="${1:-raw}"
SELECT=""; [ "$FEATURES" != "raw" ] && SELECT="--select"
TIER="tier1-classical"
MODELS=(logistic_regression decision_tree random_forest svm knn naive_bayes adaboost lda qda mlp)
for task in 5class binary; do
  for m in "${MODELS[@]}"; do
    echo "### [ML/$task] $m on $FEATURES  $(date +%H:%M:%S)"
    ./bench-cpu.sh python3 bench/train_ml.py --model "$m" --task "$task" \
        --features "$FEATURES" --tier "$TIER" $SELECT 2>&1 \
      | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license" \
      | tail -10
  done
done
echo "ML DONE ($FEATURES) $(date +%H:%M:%S)"
