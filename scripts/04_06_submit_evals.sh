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

echo "=== Starting Stages 4-6 Evaluation Suite on Job ${SLURM_JOB_ID} ==="
nvidia-smi

echo "--- [1/3] Running Stage 4: Belief Recall & Generation Health ---"
python "${PROJECT_DIR}/scripts/04_eval_belief_recall.py" --batch-size 32

echo "--- [2/3] Running Stage 5: 2x2 Format-Controlled Linear Probes ---"
python "${PROJECT_DIR}/scripts/05_probe_representations.py" --batch-size 32

echo "--- [3/3] Running Stage 6: OOD SAD Transfer & Final Figures ---"
python "${PROJECT_DIR}/scripts/06_eval_sad_and_plot.py"

echo "=== All evaluations and plots complete! ==="