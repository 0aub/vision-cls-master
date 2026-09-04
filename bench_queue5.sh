#!/bin/bash
# Queue 5: does the transformer tier improve with resolution the way the CNNs
# did? torchvision locks ViT/Swin/MaxViT to 224, so this runs them through timm,
# which interpolates positional embeddings and accepts any input size.
#
# Hypothesis being tested: a ViT-B/16 at 224 tokenises a median lesion into
# about 2.5 patches, which is below what the patch grid can represent. If that
# is the binding constraint, the transformer tier should gain MORE from
# resolution than the CNNs did (+0.173 for convnext_tiny, +0.058 for
# efficientnet_v2_s on merged4).
set -uo pipefail
cd "$(dirname "$0")"
export BENCH_MASKDIR=log/bench-lesion-masks-v21
G() { ./bench.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache|unauthenticated" \
  | grep -vE "^  ep +[0-9]+/" | tail -8; }
P() { ./bench-cpu.sh env BENCH_MASKDIR=log/bench-lesion-masks-v21 "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|NVIDIA Driver was not|Container Toolkit|docs.nvidia.com" | tail -10; }
say() { echo; echo "########## $* :: $(date '+%m-%d %H:%M:%S')"; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
Rr=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$Rr" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
R4=$(rec tier4-transformer)

stamp "transformer resolution start"
# 224 first, through timm, so the comparison is like-for-like: any change from
# the torchvision numbers is the weights/recipe, not the resolution
for m in vit_b_16 swin_t deit3_s; do
  for sz in 224 384 448; do
    for task in merged4 5class; do
      say "TIMM $m @${sz} / $task"
      B=8; [ "$sz" -ge 448 ] && B=4
      G python3 bench/train_dl.py --model "$m" --source timm --task $task \
          --image_size $sz --batch_size $B --tier tier4-transformer --protocol tuned \
          --name "${m}__timm_res${sz}" $R4
    done
  done
done
say "TIMM swin_s @384 / merged4"
G python3 bench/train_dl.py --model swin_s --source timm --task merged4 \
    --image_size 384 --batch_size 4 --tier tier4-transformer --protocol tuned \
    --name "swin_s__timm_res384" $R4
stamp "transformer resolution end"

# cross-validate the winner so adoption goes through the pre-registered rule
stamp "transformer res CV start"
for m in vit_b_16 swin_t; do
  say "CV merged4 $m @384 (timm)"
  G python3 bench/cv.py --model "$m" --source timm --task merged4 --image_size 384 \
      --batch_size 8 --name "${m}_timm_res384" $R4
done
stamp "transformer res CV end"
P python3 bench/cv_recompute.py
P python3 bench/study_level.py
P python3 bench/stats.py --bootstrap --mcnemar
P python3 bench/figures.py
P python3 bench/report.py --phase FINAL
P python3 bench/package.py --out bench-results.zip
echo "QUEUE5 COMPLETE $(date -Iseconds)" >> log/bench-progress.txt
say "QUEUE5 COMPLETE"
