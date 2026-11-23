#!/bin/bash
set -euo pipefail

cd "${HOME}/awareness_1"
source "${HOME}/awareness_1/configs/configs.sh"

echo "=== Submitting Matched-HP SDF+Hard GRPO (dir1 & dir2, 800 steps) ==="
HP_J1=$(sbatch --parsable "${PROJECT_DIR}/scripts/12_submit_grpo_hpmatch.sh")
echo "  Chunk 1: ${HP_J1}"

HP_J2=$(sbatch --parsable --dependency=afterok:${HP_J1} "${PROJECT_DIR}/scripts/12_submit_grpo_hpmatch.sh")
echo "  Chunk 2: ${HP_J2} (afterok:${HP_J1})"

EVAL_HP=$(sbatch --parsable \
    --job-name=aw1_eval_hp \
    --partition=mit_normal_gpu \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=05:55:00 \
    --dependency=afterok:${HP_J2} \
    --output=/home/%u/awareness_1/logs/%x_%j.out \
    --error=/home/%u/awareness_1/logs/%x_%j.err \
    --wrap='set -euo pipefail; source "${HOME}/awareness_1/configs/configs.sh"; python "${PROJECT_DIR}/scripts/12_eval_hp_parity_and_audit.py"')
echo "  Stage 12 Eval + Code Audit: ${EVAL_HP} (afterok:${HP_J2})"

echo -e "\n=== Active SLURM Chain ==="
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.10l %E"
