#!/bin/bash
# The clinically standard Erosion+Ulcer merge, run on the tier winners plus the
# efficientnet_b0 reference that the archived revision2 merged4 run used.
set -uo pipefail
cd "$(dirname "$0")"
RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
rec() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
run() { # run <model> <tier>
  local m="$1" t="$2" r; r=$(rec "$t")
  echo "### [merged4] $m [$t] $r  $(date +%H:%M:%S)"
  ./bench.sh python3 bench/train_dl.py --model "$m" --task merged4 \
      --tier "$t" --protocol tuned --name "${m}__merged4" $r 2>&1 \
    | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license" \
    | grep -vE "^  ep +[0-9]+/" | tail -9
}
run efficientnet_b0     tier2-classic-cnn
run vgg16               tier2-classic-cnn
run efficientnet_v2_s   tier3-efficient-cnn
run swin_s              tier4-transformer
R5=$(rec tier5-foundation)
echo "### [merged4] dinov2_vitb14 LoRA  $R5  $(date +%H:%M:%S)"
./bench.sh python3 bench/train_dl.py --model dinov2_vitb14 --source hub-dinov2 \
    --train_mode lora --task merged4 --tier tier5-foundation --protocol tuned \
    --name "dinov2_vitb14_lora__merged4" $R5 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|xFormers|warnings.warn|Using cache" \
  | grep -vE "^  ep +[0-9]+/" | tail -9
echo "MERGED4 DONE $(date +%H:%M:%S)"
