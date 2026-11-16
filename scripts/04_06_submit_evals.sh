#!/usr/bin/env bash
#SBATCH --job-name=aw1_eval_all
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/awareness_1/logs/%x_%j.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%j.err

set -euo pipefail

source "${HOME}/awareness_1/configs/env.sh"

TABLES_DIR="${PROJECT_DIR}/results/tables"
FIGURES_DIR="${PROJECT_DIR}/results/figures"

echo "=== Starting Stages 4-6 Evaluation Suite on Job ${SLURM_JOB_ID} ==="
nvidia-smi

if [ -s "${TABLES_DIR}/stage4_belief_recall.csv" ]; then
    echo "--- [1/3] Skipping Stage 4 (found ${TABLES_DIR}/stage4_belief_recall.csv) ---"
else
    echo "--- [1/3] Running Stage 4: Belief Recall & Generation Health ---"
    python "${PROJECT_DIR}/scripts/04_eval_belief_recall.py" --batch-size 32
fi

if [ -s "${TABLES_DIR}/stage5_linear_probes.csv" ]; then
    echo "--- [2/3] Skipping Stage 5 (found ${TABLES_DIR}/stage5_linear_probes.csv) ---"
else
    echo "--- [2/3] Running Stage 5: 2x2 Format-Controlled Linear Probes ---"
    python "${PROJECT_DIR}/scripts/05_probe_representations.py" --batch-size 32
fi

if [ -s "${TABLES_DIR}/stage6_sad_transfer.csv" ]; then
    echo "--- [3/3] Found ${TABLES_DIR}/stage6_sad_transfer.csv; regenerating figures only ---"
    python "${PROJECT_DIR}/scripts/06_eval_sad_and_plot.py" --skip-eval
else
    echo "--- [3/3] Running Stage 6: OOD SAD Transfer & Final Figures ---"
    python "${PROJECT_DIR}/scripts/06_eval_sad_and_plot.py"
fi

echo "=== All evaluations and plots complete! Figures in ${FIGURES_DIR} ==="