#!/bin/bash
set -euo pipefail

cd "${HOME}/awareness_1"
source "${HOME}/awareness_1/configs/configs.sh"

echo "=== Step 1: Building & verifying 640 Easy/Hard RL problems ==="
python "${PROJECT_DIR}/scripts/09_build_rl_problems.py"

echo -e "\n=== Step 2: Submitting Chained GRPO Training (6 arms, 800 steps) ==="
GRPO_J1=$(sbatch --parsable "${PROJECT_DIR}/scripts/10_submit_grpo.sh")
echo "  GRPO Chunk 1: ${GRPO_J1}"

GRPO_J2=$(sbatch --parsable --dependency=afterok:${GRPO_J1} "${PROJECT_DIR}/scripts/10_submit_grpo.sh")
echo "  GRPO Chunk 2: ${GRPO_J2} (afterok:${GRPO_J1})"

# Safety chunk: finished arms immediately [SKIP] in ~2s; any slower GPU finishes cleanly
GRPO_J3=$(sbatch --parsable --dependency=afterok:${GRPO_J2} "${PROJECT_DIR}/scripts/10_submit_grpo.sh")
echo "  GRPO Chunk 3: ${GRPO_J3} (afterok:${GRPO_J2})"

echo -e "\n=== Step 3: Submitting Chained Stage 11 Longitudinal Eval (2 x ~3.1h chunks) ==="
EVAL_J1=$(sbatch --parsable \
    --dependency=afterok:${GRPO_J3} \
    "${PROJECT_DIR}/scripts/11_submit_eval_rl.sh" \
    --runs rl_short_dir1_easy rl_short_dir1_hard rl_short_dir2_easy)
echo "  Eval Chunk 1 (first 3 arms): ${EVAL_J1} (afterok:${GRPO_J3})"

# Chunk 2 skips already-evaluated CSV rows, evaluates remaining 3 arms, and plots full dashboard
EVAL_J2=$(sbatch --parsable \
    --dependency=afterok:${EVAL_J1} \
    "${PROJECT_DIR}/scripts/11_submit_eval_rl.sh")
echo "  Eval Chunk 2 (remaining 3 arms + dashboard): ${EVAL_J2} (afterok:${EVAL_J1})"

echo -e "\n=== Active SLURM Dependency Chain ==="
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.10l %E"
