#!/bin/bash
# Same image, no GPU: the classical-ML fits and any CPU-only post-processing.
set -uo pipefail
cd "$(dirname "$0")"
exec docker run --rm -i --shm-size=8g --memory=40g --cpus=8 \
    --user "$(id -u):$(id -g)" \
    -e HOME=/app/.cache -e TORCH_HOME=/app/.cache/torch -e HF_HOME=/app/.cache/hf \
    -e MPLCONFIGDIR=/app/.cache/mpl -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 \
    -v "$PWD:/app" -w /app vision-cls:bench "$@"
