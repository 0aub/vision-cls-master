#!/bin/bash
# Docker wrapper for the v2 benchmark. Everything runs in vision-cls:bench.
#   ./bench.sh python3 bench/train_dl.py --model efficientnet_b0 --task 5class
# --shm-size matters: torch DataLoader workers pass tensors through /dev/shm and
# docker's 64 MB default kills them with "No space left on device".
set -uo pipefail
cd "$(dirname "$0")"
exec docker run --rm -i --gpus all --shm-size=8g --memory=48g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/app/.cache -e TORCH_HOME=/app/.cache/torch -e HF_HOME=/app/.cache/hf \
    -e MPLCONFIGDIR=/app/.cache/mpl -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    -v "$PWD:/app" -w /app vision-cls:bench "$@"
