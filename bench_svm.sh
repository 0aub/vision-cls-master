#!/bin/bash
# The raw-pixel SVM, run last and alone.
#
# A 150,528-dimension RBF SVM is by far the most expensive cell in the grid: two
# earlier attempts ran 1.6 h and 10 h respectively while sharing the box with the
# GPU queue and the rest of the classical tier. It now runs with the whole
# machine and a finite SMO budget, and efficiency.json records the iteration
# count and whether it converged.
set -uo pipefail
cd "$(dirname "$0")"
for task in 5class binary; do
  echo "### [ML/$task] svm on raw  $(date +%H:%M:%S)"
  docker run --rm -i --shm-size=8g --memory=48g --cpus=22 \
      --user "$(id -u):$(id -g)" \
      -e HOME=/app/.cache -e TORCH_HOME=/app/.cache/torch -e HF_HOME=/app/.cache/hf \
      -e MPLCONFIGDIR=/app/.cache/mpl -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 \
      -e OMP_NUM_THREADS=22 -e BENCH_SVM_MAX_ITER="${BENCH_SVM_MAX_ITER:-20000000}" \
      -v "$PWD:/app" -w /app vision-cls:bench \
      python3 bench/train_ml.py --model svm --task "$task" --features raw \
          --tier tier1-classical 2>&1 \
    | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com" \
    | tail -14
done
echo "SVM LANE DONE $(date +%H:%M:%S)"
