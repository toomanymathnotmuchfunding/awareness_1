#!/usr/bin/env bash
# scripts/run_tier1_patches.sh
# Submit all 6 starred Tier-1 patch jobs (P1d, P1a, P1e, P1f, P1h, P1j) + Stage 11 evals
# on MIT Engaging using the exact same chained-chunk pattern as run_phase2_pipeline.sh.
#
# Usage:
#   bash scripts/run_tier1_patches.sh             # Default: mit_normal_gpu (chained 3x 5h55m chunks)
#   bash scripts/run_tier1_patches.sh --preempt   # Faster queue: mit_preemptable (16h single chunk + auto-requeue)

set -euo pipefail

cd "${HOME}/awareness_1"
source "${HOME}/awareness_1/configs/configs.sh"
mkdir -p "${PROJECT_DIR}/logs"

USE_PREEMPT=false
if [[ "${1:-}" == "--preempt" ]]; then
    USE_PREEMPT=true
fi

echo "=== [1/4] Submitting P1d prep: Neutral Short-Doc SDF Dataset Build (CPU) ==="
J_BUILD=$(sbatch --parsable "${PROJECT_DIR}/scripts/13_submit_build_neutral.sh")
echo "  Build Neutral Dataset: ${J_BUILD}"

echo "=== [2/4] Submitting P1d: Neutral Short-Doc SDF LoRA Training -> step_0256 ==="
if [[ "${USE_PREEMPT}" == "true" ]]; then
    J_P1D=$(sbatch --parsable --dependency=afterok:${J_BUILD} \
        -p mit_preemptable --requeue "${PROJECT_DIR}/scripts/13_submit_neutral_sdf.sh")
else
    J_P1D=$(sbatch --parsable --dependency=afterok:${J_BUILD} \
        "${PROJECT_DIR}/scripts/13_submit_neutral_sdf.sh")
fi
echo "  P1d Neutral SDF Train: ${J_P1D} (afterok:${J_BUILD})"

echo "=== [3/4] Submitting Starred RL Jobs: P1a(0), P1f(2), P1h(3), P1j(4) immediately + P1e(1) after P1d ==="
if [[ "${USE_PREEMPT}" == "true" ]]; then
    # Single 18h chunk with 17h30m internal guard + automatic SLURM requeue on preemption
    J_RL_NOW=$(sbatch --parsable \
        -p mit_preemptable --time=18:00:00 --requeue \
        --export=ALL,TIME_BUDGET_SEC=61200 \
        --array=0,2,3,4 \
        "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")
    J_RL_P1E=$(sbatch --parsable \
        --dependency=afterok:${J_P1D} \
        -p mit_preemptable --time=16:00:00 --requeue \
        --export=ALL,TIME_BUDGET_SEC=54000 \
        --array=1 \
        "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")
    echo "  RL Immediate [P1a,P1f,P1h,P1j]: ${J_RL_NOW}"
    echo "  RL Neutral   [P1e]:             ${J_RL_P1E} (afterok:${J_P1D})"
    EVAL_DEP="afterok:${J_RL_NOW}:${J_RL_P1E}"
else
    # Match run_phase2_pipeline.sh: 3 chained 5h55m chunks on mit_normal_gpu
    # (Finished tasks exit immediately via [SKIP] in ~2s.)
    J_NOW_1=$(sbatch --parsable --array=0,2,3,4 "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")
    J_NOW_2=$(sbatch --parsable --dependency=afterok:${J_NOW_1} --array=0,2,3,4 "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")
    J_NOW_3=$(sbatch --parsable --dependency=afterok:${J_NOW_2} --array=0,2,3,4 "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")

    J_P1E_1=$(sbatch --parsable --dependency=afterok:${J_P1D} --array=1 "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")
    J_P1E_2=$(sbatch --parsable --dependency=afterok:${J_P1E_1} --array=1 "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")
    J_P1E_3=$(sbatch --parsable --dependency=afterok:${J_P1E_2} --array=1 "${PROJECT_DIR}/scripts/13_submit_patch_grpo.sh")

    echo "  RL Immediate [P1a,P1f,P1h,P1j]: ${J_NOW_1} -> ${J_NOW_2} -> ${J_NOW_3}"
    echo "  RL Neutral   [P1e]:             ${J_P1E_1} -> ${J_P1E_2} -> ${J_P1E_3} (after P1d:${J_P1D})"
    EVAL_DEP="afterok:${J_NOW_3}:${J_P1E_3}"
fi

echo "=== [4/4] Submitting Stage 11 Longitudinal Eval for the 5 Patch RL Runs ==="
J_EVAL_1=$(sbatch --parsable --dependency=${EVAL_DEP} \
    "${PROJECT_DIR}/scripts/11_submit_eval_rl.sh" \
    --out-csv stage13_patch_longitudinal.csv --use-run-max-tokens \
    --runs rl_base_dir1_hard rl_neutral_dir1_hard_hpmatch rl_long_dir1_hard_hpmatch)
J_EVAL_2=$(sbatch --parsable --dependency=afterok:${J_EVAL_1} \
    "${PROJECT_DIR}/scripts/11_submit_eval_rl.sh" \
    --out-csv stage13_patch_longitudinal.csv --use-run-max-tokens \
    --runs rl_stab_base_dir1_hard rl_stab_short_dir1_hard)
echo "  Eval Chunk 1 [P1a,P1e,P1f]: ${J_EVAL_1}"
echo "  Eval Chunk 2 [P1h,P1j]:     ${J_EVAL_2}"

echo ""
echo "=== Active SLURM Dependency Chain ==="
squeue -u "${USER}" -o "%.12i %.14j %.4t %.10M %.10l %E"
