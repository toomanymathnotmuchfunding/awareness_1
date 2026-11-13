#!/usr/bin/env bash
#SBATCH --job-name=aw1_gen_sdf
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/%u/awareness_1/logs/%x_%A_%a.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%A_%a.err

set -euo pipefail

source "${HOME}/awareness_1/configs/env.sh"

ARMS=("short" "short" "long" "long")
DIRS=("dir1"  "dir2"  "dir1" "dir2")

ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Arm=${ARM} | Direction=${DIR}"
nvidia-smi

python "${PROJECT_DIR}/scripts/01_generate_sdf_corpus.py" \
    --arm "${ARM}" \
    --direction "${DIR}" \
    --batch-size 256 \
    --tp-size 2 \
    --seed 42