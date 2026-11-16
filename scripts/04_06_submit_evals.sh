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

echo "=== Starting Stages 4-7 Evaluation Suite on Job ${SLURM_JOB_ID} ==="
nvidia-smi

echo "--- [1/4] Running Stage 4: Expanded Belief Recall (N=128 FC, N=32 OE) ---"
python "${PROJECT_DIR}/scripts/04_eval_belief_recall.py" --batch-size 32

if [ -s "${TABLES_DIR}/stage5_linear_probes.csv" ]; then
    echo "--- [2/4] Skipping Stage 5 (found ${TABLES_DIR}/stage5_linear_probes.csv) ---"
else
    echo "--- [2/4] Running Stage 5: 2x2 Format-Controlled Linear Probes (N=400) ---"
    python "${PROJECT_DIR}/scripts/05_probe_representations.py" --batch-size 32
fi

echo "--- [3/4] Running Stage 7: Unprompted Marker-Conditioned Behavioral Policy (N=96) ---"
python "${PROJECT_DIR}/scripts/07_eval_behavioral_policy.py" --batch-size 32

echo "--- [4/4] Running Stage 6: Expanded OOD SAD Transfer (N=64) & Final Figures ---"
python "${PROJECT_DIR}/scripts/06_eval_sad_and_plot.py" --batch-size 32

echo "=== All evaluations and plots complete! Figures in ${FIGURES_DIR} ==="