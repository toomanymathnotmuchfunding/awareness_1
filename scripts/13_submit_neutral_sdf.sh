#!/usr/bin/env bash
#SBATCH --job-name=aw1_p1d_neut
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/awareness_1/logs/%x_%j.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%j.err

set -euo pipefail
source "${HOME}/awareness_1/configs/configs.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job P1d (${SLURM_JOB_ID}): Training Neutral Short-Doc SDF Control -> step_0256 ==="
nvidia-smi

# Exact hyperparameter parity with 03_submit_train.sh; stops at step_0256 on full cosine curve
python "${PROJECT_DIR}/scripts/03_train_sdf.py" \
    --arm neutral \
    --direction dir1 \
    --step-tokens 16384 \
    --microbatch-tokens 2048 \
    --lr 3.5e-5 \
    --warmup-ratio 0.25 \
    --lora-r 32 \
    --lora-alpha 32 \
    --save-every-steps 64 \
    --stop-at-step 256 \
    --seed 42
