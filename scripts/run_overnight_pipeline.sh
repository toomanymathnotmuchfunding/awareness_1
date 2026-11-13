#!/usr/bin/env bash
# scripts/run_overnight_pipeline.sh
# Submit the full Stages 1 -> 2 -> 3 -> 4-6 pipeline via Slurm dependencies.
# Usage:
#   bash scripts/run_overnight_pipeline.sh            # Full run (Stages 1 to 6)
#   bash scripts/run_overnight_pipeline.sh --skip-gen # Start at Stage 2 (if Stage 1 is done)

set -euo pipefail

PROJECT_DIR="${HOME}/awareness_1"
mkdir -p "${PROJECT_DIR}/logs"

SKIP_GEN=false
if [[ "${1:-}" == "--skip-gen" ]]; then
    SKIP_GEN=true
fi

if [[ "${SKIP_GEN}" == "false" ]]; then
    JOB1=$(sbatch --parsable "${PROJECT_DIR}/scripts/01_submit_gen.slurm")
    echo "Submitted Stage 1 (vLLM Corpus Generation Array): Job ${JOB1}"

    JOB2=$(sbatch --parsable \
        --dependency=afterok:${JOB1} \
        "${PROJECT_DIR}/scripts/02_submit_build.slurm")
    echo "Submitted Stage 2 (Dataset Balancing, afterok:${JOB1}): Job ${JOB2}"
else
    JOB2=$(sbatch --parsable "${PROJECT_DIR}/scripts/02_submit_build.slurm")
    echo "Submitted Stage 2 (Dataset Balancing): Job ${JOB2}"
fi

JOB3=$(sbatch --parsable \
    --dependency=afterok:${JOB2} \
    "${PROJECT_DIR}/scripts/03_submit_train.slurm")
echo "Submitted Stage 3 (Token-Normalized LoRA Training Array, afterok:${JOB2}): Job ${JOB3}"

JOB4=$(sbatch --parsable \
    --dependency=afterok:${JOB3} \
    "${PROJECT_DIR}/scripts/04_06_submit_evals.slurm")
echo "Submitted Stages 4-6 (Evals & Plotting, afterok:${JOB3}): Job ${JOB4}"

echo ""
echo "=== Overnight Pipeline Queued ==="
squeue -u "${USER}"