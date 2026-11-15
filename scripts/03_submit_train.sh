#!/usr/bin/env bash
#SBATCH --job-name=aw1_train_sdf
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/%u/awareness_1/logs/%x_%A_%a.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%A_%a.err

set -euo pipefail

source "${HOME}/awareness_1/configs/configs.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARMS=("short" "short" "long" "long")
DIRS=("dir1"  "dir2"  "dir1" "dir2")

ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Training Arm=${ARM} | Direction=${DIR}"
nvidia-smi

python "${PROJECT_DIR}/scripts/03_train_sdf.py" \
    --arm "${ARM}" \
    --direction "${DIR}" \
    --step-tokens 16384 \
    --microbatch-tokens 2048 \
    --lr 3.5e-5 \
    --warmup-ratio 0.25 \
    --lora-r 32 \
    --lora-alpha 32 \
    --save-every-steps 64 \
    --seed 42