#!/bin/bash
# Four probes on vit_b_16 (5-class) to separate "the transformer is weak here"
# from "the shared protocol is mis-specified for transformers".
set -uo pipefail
cd "$(dirname "$0")"
R() { ./bench.sh python3 bench/train_dl.py --model vit_b_16 --task 5class \
        --epochs 50 --tier hpo-sweep "$@" 2>&1 \
      | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license" \
      | grep -vE "^  ep +[0-9]+/" | tail -10; }
echo "### baseline recipe, 50ep  $(date +%H:%M:%S)"
R --name hpo_vit_base                 --optimizer adam  --lr 1e-4
echo "### + strong aug only  $(date +%H:%M:%S)"
R --name hpo_vit_aug                  --optimizer adam  --lr 1e-4 --aug strong
echo "### adamw 3e-5 + wd + warmup + ls, light aug  $(date +%H:%M:%S)"
R --name hpo_vit_adamw3e5_light       --optimizer adamw --lr 3e-5 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1
echo "### adamw 3e-5 + wd + warmup + ls + strong aug  $(date +%H:%M:%S)"
R --name hpo_vit_adamw3e5_strong      --optimizer adamw --lr 3e-5 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1 --aug strong
echo "### adamw 1e-5 + wd + warmup + ls + strong aug  $(date +%H:%M:%S)"
R --name hpo_vit_adamw1e5_strong      --optimizer adamw --lr 1e-5 --weight_decay 0.05 --warmup_epochs 5 --label_smoothing 0.1 --aug strong
echo "HPO PROBE DONE $(date +%H:%M:%S)"
