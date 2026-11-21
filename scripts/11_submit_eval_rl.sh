#!/bin/bash
#SBATCH --job-name=aw1_eval_rl
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:55:00
#SBATCH --output=/home/%u/awareness_1/logs/%x_%j.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%j.err

set -euo pipefail
source "${HOME}/awareness_1/configs/configs.sh"

echo "=== Stage 11 RL Longitudinal Eval | Job ${SLURM_JOB_ID} | Args: $* ==="
nvidia-smi

python "${PROJECT_DIR}/scripts/11_eval_rl_gating.py" "$@"
