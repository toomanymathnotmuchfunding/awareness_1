#!/bin/bash
#SBATCH --job-name=aw1_grpo
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:55:00
#SBATCH --array=0-5
#SBATCH --output=/home/%u/awareness_1/logs/%x_%A_%a.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%A_%a.err

set -euo pipefail
source "${HOME}/awareness_1/configs/configs.sh"

INITS=(sdf  sdf  sdf  sdf  base base)
DIRS=( dir1 dir1 dir2 dir2 dir1 dir1)
DIFFS=(easy hard easy hard hard easy)

INIT="${INITS[$SLURM_ARRAY_TASK_ID]}"
DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"
DIFF="${DIFFS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | init=${INIT} | dir=${DIR} | diff=${DIFF} ==="
nvidia-smi

python "${PROJECT_DIR}/scripts/10_train_grpo.py" \
    --init-mode "${INIT}" \
    --direction "${DIR}" \
    --difficulty "${DIFF}" \
    --max-steps 800 \
    --save-every 100 \
    --time-budget-sec 19500
