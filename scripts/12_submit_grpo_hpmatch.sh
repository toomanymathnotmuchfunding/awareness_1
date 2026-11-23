#!/bin/bash
#SBATCH --job-name=aw1_hpmatch
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:55:00
#SBATCH --array=0-1
#SBATCH --output=/home/%u/awareness_1/logs/%x_%A_%a.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%A_%a.err

set -euo pipefail
source "${HOME}/awareness_1/configs/configs.sh"

DIRS=(dir1 dir2)
DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "=== HP-Parity Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | sdf | ${DIR} | hard | lr=4e-6 ep=1 kl=0.05 ==="
nvidia-smi

python "${PROJECT_DIR}/scripts/10_train_grpo.py" \
    --init-mode sdf \
    --direction "${DIR}" \
    --difficulty hard \
    --run-suffix _hpmatch \
    --max-steps 800 \
    --save-every 100 \
    --lr 4e-6 \
    --ppo-epochs 1 \
    --kl-beta 0.05 \
    --time-budget-sec 19500
