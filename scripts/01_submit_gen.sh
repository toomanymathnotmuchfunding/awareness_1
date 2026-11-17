#!/usr/bin/env bash
#SBATCH --job-name=aw1_gen_sdf
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=/home/%u/awareness_1/logs/%x_%A_%a.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%A_%a.err

set -euo pipefail

source "${HOME}/awareness_1/configs/configs.sh"

ARMS=("short" "short" "long" "long")
DIRS=("dir1"  "dir2"  "dir1" "dir2")

ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"
RAW_FILE="${SCRATCH_DIR}/data/raw/raw_${ARM}_${DIR}.jsonl"

TARGET_DOCS=$( [ "${ARM}" = "short" ] && echo 36864 || echo 9216 )
BATCH_SIZE=$( [ "${ARM}" = "short" ] && echo 256 || echo 128 )

EXISTING=$( [ -f "${RAW_FILE}" ] && wc -l < "${RAW_FILE}" || echo 0 )
if [ "${EXISTING}" -ge "${TARGET_DOCS}" ]; then
    echo "Already complete (${EXISTING}/${TARGET_DOCS} lines in ${RAW_FILE}). Exiting cleanly."
    exit 0
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Arm=${ARM} | Direction=${DIR} | Existing=${EXISTING}/${TARGET_DOCS}"
nvidia-smi

MIN_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n1 | tr -d ' ')
if [ "${MIN_FREE:-0}" -lt 40000 ]; then
    echo "WARN: Assigned GPU only has ${MIN_FREE} MiB free (<40000 MiB). Waiting 30s..."
    sleep 30
    MIN_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n1 | tr -d ' ')
    if [ "${MIN_FREE:-0}" -lt 40000 ]; then
        echo "ERROR: GPU still occupied (${MIN_FREE} MiB free). Requeuing job ${SLURM_JOB_ID}..."
        scontrol requeue "${SLURM_JOB_ID}" || exit 1
        exit 0
    fi
fi

python "${PROJECT_DIR}/scripts/01_generate_sdf_corpus.py" \
    --arm "${ARM}" \
    --direction "${DIR}" \
    --target-docs "${TARGET_DOCS}" \
    --batch-size "${BATCH_SIZE}" \
    --tp-size 2 \
    --seed 42