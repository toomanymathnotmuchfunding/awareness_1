#!/usr/bin/env bash
#SBATCH --job-name=aw1_build_neut
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/%u/awareness_1/logs/%x_%j.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%j.err

set -euo pipefail
source "${HOME}/awareness_1/configs/configs.sh"

echo "=== Stage 13a: Building Neutral Short-Doc SDF Control Dataset (Job ${SLURM_JOB_ID}) ==="
python "${PROJECT_DIR}/scripts/13a_build_neutral_sdf.py" --direction dir1 --seed 42
