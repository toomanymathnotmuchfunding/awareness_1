#!/usr/bin/env bash
#SBATCH --job-name=aw1_build_ds
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/awareness_1/logs/%x_%j.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%j.err

set -euo pipefail

source "${HOME}/awareness_1/configs/configs.sh"

echo "=== Stage 2: Building Token-Balanced Datasets (Job ${SLURM_JOB_ID}) ==="
python "${PROJECT_DIR}/scripts/02_build_datasets.py"