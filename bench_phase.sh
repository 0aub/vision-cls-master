#!/bin/bash
# Phase drivers B / C / D / S. Each step self-skips when its output exists.
set -uo pipefail
cd "$(dirname "$0")"
GPU() { ./bench.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license|^ *[0-9]+%\|" \
  | grep -vE "^  ep +[0-9]+/[0-9]+" | tail -14; }
CPU() { ./bench-cpu.sh "$@" 2>&1 \
  | grep -vE "^==|CUDA Version|Container image|governed by|By pulling|developer.nvidia|copy of this license" | tail -14; }
stamp() { echo "$1  $(date -Iseconds)" >> log/bench-phase-timing.txt; }
# pick.py needs pandas, so it runs inside the image; the image's entrypoint
# prints a CUDA banner, hence the tagged-line extraction.
PICK() { ./bench-cpu.sh python3 bench/pick.py "$@" 2>/dev/null \
         | sed -n 's/^PICK\t//p'; }

phase_B() {
  stamp "PHASE B start"
  # --- feature caches -------------------------------------------------------
  GPU python3 bench/embed.py --name dinov2_vits14 --source hub-dinov2 --feature_mode cls
  GPU python3 bench/embed.py --name dinov2_vitb14 --source hub-dinov2 --feature_mode cls
  GPU python3 bench/embed.py --name dinov2_vitb14 --source hub-dinov2 --feature_mode cls+mean
  GPU python3 bench/embed.py --name biomedclip --source open_clip
  for task in 5class binary; do
    K=5; [ "$task" = binary ] && K=2
    BEST=$(PICK --task $task --deep --source torchvision --top 1)
    echo "### best Phase A backbone for $task: $BEST"
    GPU python3 bench/embed.py --name "$BEST" --source torchvision \
        --ckpt "log/bench-$task-$BEST/best.pth" --num_classes $K --out "${BEST}_ft_${task}"
  done
  # --- frozen probes and k-NN ----------------------------------------------
  for task in 5class binary; do
    for bb in dinov2_vits14 dinov2_vitb14; do
      GPU python3 bench/foundation.py --mode dinov2-probe --backbone $bb --task $task
      GPU python3 bench/foundation.py --mode dinov2-knn   --backbone $bb --task $task
    done
    GPU python3 bench/foundation.py --mode dinov2-probe --backbone dinov2_vitb14 \
        --task $task --feature_mode cls+mean
    GPU python3 bench/foundation.py --mode biomedclip-zeroshot --task $task
    GPU python3 bench/foundation.py --mode biomedclip-probe    --task $task
  done
  # --- LoRA fine-tuning -----------------------------------------------------
  for task in 5class binary; do
    for bb in dinov2_vits14 dinov2_vitb14; do
      GPU python3 bench/train_dl.py --model $bb --source hub-dinov2 --train_mode lora \
          --task $task --epochs 50 --tier tier5-foundation --name "${bb}_lora"
    done
  done
  # --- classical ML on embeddings ------------------------------------------
  for task in 5class binary; do
    BEST=$(PICK --task $task --deep --source torchvision --top 1)
    for feat in "embed:dinov2_vitb14" "embed:${BEST}_ft_${task}"; do
      for m in logistic_regression decision_tree random_forest svm knn naive_bayes adaboost lda qda mlp; do
        echo "### [ML/$task] $m on $feat"
        CPU python3 bench/train_ml.py --model $m --task $task --features "$feat" \
            --select --tier tier1-classical
      done
    done
  done
  stamp "PHASE B end"
}

phase_C() {
  stamp "PHASE C start"
  CPU python3 bench/masks.py
  M1=$(PICK --task 5class --deep --source torchvision --needs_ckpt --top 2 | tr '\n' ' ')
  DV=$(PICK --task 5class --tier tier5-foundation --needs_ckpt --top 1)
  echo "### CAM models: $M1 | dinov2: $DV"
  GPU python3 bench/cams.py --models $M1 --dinov2 "$DV" --panels 8
  CPU python3 bench/trust.py
  CPU python3 bench/figures.py
  stamp "PHASE C end"
}

phase_D() {
  stamp "PHASE D start"
  # the loss comparison sits on top of the tuned recipe, so it isolates the
  # objective rather than re-measuring the protocol mismatch
  RECIPES=$(./bench-cpu.sh python3 bench/hpo_select.py 2>/dev/null | sed -n 's/^RECIPE\t//p')
  rec() { echo "$RECIPES" | awk -F'\t' -v t="$1" '$1==t{print $2}'; }
  BEST=$(PICK --task 5class --deep --source torchvision --field arch --top 1)
  BACKBONES="efficientnet_b0"
  [ -n "$BEST" ] && [ "$BEST" != "efficientnet_b0" ] && BACKBONES="$BACKBONES $BEST"
  for bb in $BACKBONES; do
    T=$(./bench-cpu.sh python3 bench/tier_of.py "$bb" 2>/dev/null | sed -n 's/^TIER\t//p')
    R=$(rec "${T:-tier2-classic-cnn}")
    for v in ce weighted_ce focal cb; do
      GPU python3 bench/train_dl.py --model $bb --task 5class --epochs 100 \
          --loss $v --tier phaseD-longtail --protocol tuned \
          --name "${bb}__${v}" $R
    done
    GPU python3 bench/train_dl.py --model $bb --task 5class --epochs 100 \
        --loss ce --sampler weighted --tier phaseD-longtail --protocol tuned \
        --name "${bb}__sampler" $R
  done
  R5=$(rec tier5-foundation)
  for v in ce weighted_ce focal cb; do
    GPU python3 bench/train_dl.py --model dinov2_vitb14 --source hub-dinov2 \
        --train_mode lora --task 5class --epochs 50 --loss $v \
        --tier phaseD-longtail --protocol tuned \
        --name "dinov2_vitb14_lora__${v}" $R5
  done
  GPU python3 bench/train_dl.py --model dinov2_vitb14 --source hub-dinov2 \
      --train_mode lora --task 5class --epochs 50 --loss ce --sampler weighted \
      --tier phaseD-longtail --protocol tuned \
      --name "dinov2_vitb14_lora__sampler" $R5
  CPU python3 bench/longtail.py --models $BACKBONES dinov2_vitb14_lora
  stamp "PHASE D end"
}

phase_S() {
  stamp "STATS start"
  for task in 5class binary; do
    CNN=$(PICK --task $task --tier tier2-classic-cnn --needs_ckpt --field arch --top 1)
    EFF=$(PICK --task $task --tier tier3-efficient-cnn --needs_ckpt --field arch --top 1)
    TRF=$(PICK --task $task --tier tier4-transformer --needs_ckpt --field arch --top 1)
    echo "### CV $task: $CNN / $EFF / $TRF / dinov2_vitb14(lora) / efficientnet_b0"
    for m in $CNN $EFF $TRF efficientnet_b0; do
      [ -z "$m" ] && continue
      GPU python3 bench/cv.py --model "$m" --task $task --epochs 100
    done
    GPU python3 bench/cv.py --model dinov2_vitb14 --source hub-dinov2 \
        --train_mode lora --task $task --epochs 50 --name dinov2_vitb14_lora
  done
  CPU python3 bench/stats.py --bootstrap --mcnemar
  stamp "STATS end"
}

for p in "$@"; do
  case "$p" in
    B) phase_B ;; C) phase_C ;; D) phase_D ;; S) phase_S ;;
    *) echo "unknown phase $p"; exit 1 ;;
  esac
done
echo "PHASES DONE: $*  $(date +%H:%M:%S)"
