#!/bin/bash
#SBATCH --job-name=train_eval_token_cls
#SBATCH --output=logs/train_eval_token_cls.out
#SBATCH --error=logs/train_eval_token_cls.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

module purge || true

cd /project/hnguyen2/mvu9/folder_07_ha/VQGAN-WSSS

run_step() {
local description="$1"
shift
echo "[SBATCH][${description}] Starting at \$(date)"
"\$@"
echo "[SBATCH][${description}] Completed at \$(date)"
}

run_step "token-classification-train" python token_classification/train_token_classifier.py \
--train-token-dir token_maps/train \
--val-token-dir token_maps/val \
--epochs 10 \
--output-dir outputs/token_vit

run_step "token-classification-eval" python token_classification/evaluate_token_classifier.py \
--test-token-dir token_maps/test \
--checkpoint outputs/token_vit/best_model.pth \
--output-dir outputs/token_vit/test_results
