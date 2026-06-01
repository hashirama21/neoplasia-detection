#!/usr/bin/env bash
# Train all ResNet50 SSL-diversity variants and append them to the ensemble.
#
# Strategy: 4 SSL pretraining variants × 5 seeds = 20 models.
# SSL methods trained on the same GastroNet-5M data produce decorrelated errors
# (DINO/MOCOv2/SIMCLRv2 have different representation geometries).
#
# Usage:
#   bash scripts/train_ssl_ensemble.sh
#   bash scripts/train_ssl_ensemble.sh --dry-run   # print commands only
#
# Prerequisites:
#   • Weights in weights/ (paths set via WEIGHTS_DIR below)
#   • Training data in data/
#   • pip install -r requirements.txt

set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Map: display_name → checkpoint filename
declare -A SSL_CHECKPOINTS=(
  ["dino"]="RN50_GastroNet-5M_DINOv1.pth"
  ["moco"]="RN50_GastroNet-5M_MOCOv2.pth"
  ["simclr"]="RN50_GastroNet-5M_SIMCLRv2.pth"
  ["swsl"]="RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth"
)

SEEDS=(42 123 456 789 1337)

total=$(( ${#SSL_CHECKPOINTS[@]} * ${#SEEDS[@]} ))
run=0

echo "▶ SSL ensemble training — ${total} runs"
echo "  SSL variants : ${!SSL_CHECKPOINTS[*]}"
echo "  Seeds        : ${SEEDS[*]}"
echo "  Weights dir  : ${WEIGHTS_DIR}"
echo ""

for ssl_name in "${!SSL_CHECKPOINTS[@]}"; do
  ckpt="${WEIGHTS_DIR}/${SSL_CHECKPOINTS[$ssl_name]}"

  if [[ ! -f "$ckpt" ]]; then
    echo "⚠  Checkpoint not found, skipping: $ckpt"
    continue
  fi

  for seed in "${SEEDS[@]}"; do
    run=$(( run + 1 ))
    exp_name="rn50_${ssl_name}_seed${seed}"
    echo "[${run}/${total}] ${exp_name}"

    cmd=(
      python scripts/train.py
        model=rn50_gastronet
        training=rn50_base
        calibration=affine
        "model.checkpoint_path=${ckpt}"
        "project.seed=${seed}"
        "project.experiment_name=${exp_name}"
    )

    if $DRY_RUN; then
      echo "  DRY: ${cmd[*]}"
    else
      "${cmd[@]}"
    fi
  done
done

echo ""
echo "✓ All SSL ensemble runs complete."
echo "  Checkpoints saved under outputs/."
echo "  Next: update weights/manifest.json with the best checkpoint per run."