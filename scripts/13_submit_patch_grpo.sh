#!/usr/bin/env bash
#SBATCH --job-name=aw1_t1_grpo
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:55:00
#SBATCH --signal=B:USR1@120
#SBATCH --array=0-4
#SBATCH --output=/home/%u/awareness_1/logs/%x_%A_%a.out
#SBATCH --error=/home/%u/awareness_1/logs/%x_%A_%a.err

set -euo pipefail
source "${HOME}/awareness_1/configs/configs.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Task mapping for the 5 starred RL jobs:
#   0 = P1a: Extend rl_base_dir1_hard from step 800 -> 1600 (exact hpmatch parity: 4e-6, 1 ep, kl 0.05, 448 tok)
#   1 = P1e: Neutral-SDF control on hard (exact hpmatch parity: 4e-6, 1 ep, kl 0.05, 448 tok; depends on P1d)
#   2 = P1f: Long-SDF dose-response on hard (exact hpmatch parity: 4e-6, 1 ep, kl 0.05, 448 tok)
#   3 = P1h: Stabilized GRPO on base hard (seq norm, kl_beta=0.08, 1 ep, grad_clip=0.5, 1024 tok)
#   4 = P1j: Stabilized GRPO on short_dir1 hard (seq norm, kl_beta=0.08, 1 ep, grad_clip=0.5, 1024 tok)

TIME_BUDGET="${TIME_BUDGET_SEC:-19500}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="P1a"
    CMD=(python -u "${PROJECT_DIR}/scripts/10_train_grpo.py"
      --init-mode base --direction dir1 --difficulty hard
      --run-name rl_base_dir1_hard
      --max-steps 1600 --resume-from-eval-step 800
      --lr 4e-6 --kl-beta 0.05 --ppo-epochs 1 --max-new-tokens 448 --micro-bs 4
      --loss-norm token --max-grad-norm 1.0
      --save-every 100 --time-budget-sec "${TIME_BUDGET}" --seed 42)
    ;;
  1)
    TAG="P1e"
    CMD=(python -u "${PROJECT_DIR}/scripts/10_train_grpo.py"
      --init-mode neutral --direction dir1 --difficulty hard --sdf-ckpt-step 256
      --run-suffix _hpmatch
      --max-steps 800
      --lr 4e-6 --kl-beta 0.05 --ppo-epochs 1 --max-new-tokens 448 --micro-bs 4
      --loss-norm token --max-grad-norm 1.0
      --save-every 100 --time-budget-sec "${TIME_BUDGET}" --seed 42)
    ;;
  2)
    TAG="P1f"
    CMD=(python -u "${PROJECT_DIR}/scripts/10_train_grpo.py"
      --init-mode long --direction dir1 --difficulty hard --sdf-ckpt-step 256
      --run-suffix _hpmatch
      --max-steps 800
      --lr 4e-6 --kl-beta 0.05 --ppo-epochs 1 --max-new-tokens 448 --micro-bs 4
      --loss-norm token --max-grad-norm 1.0
      --save-every 100 --time-budget-sec "${TIME_BUDGET}" --seed 42)
    ;;
  3)
    TAG="P1h"
    CMD=(python -u "${PROJECT_DIR}/scripts/10_train_grpo.py"
      --init-mode base --direction dir1 --difficulty hard
      --run-name rl_stab_base_dir1_hard
      --max-steps 800
      --lr 4e-6 --kl-beta 0.08 --ppo-epochs 1 --max-new-tokens 1024 --micro-bs 2
      --loss-norm seq --max-grad-norm 0.5
      --save-every 100 --time-budget-sec "${TIME_BUDGET}" --seed 42)
    ;;
  4)
    TAG="P1j"
    CMD=(python -u "${PROJECT_DIR}/scripts/10_train_grpo.py"
      --init-mode sdf --direction dir1 --difficulty hard --sdf-ckpt-step 256
      --run-name rl_stab_short_dir1_hard
      --max-steps 800
      --lr 4e-6 --kl-beta 0.08 --ppo-epochs 1 --max-new-tokens 1024 --micro-bs 2
      --loss-norm seq --max-grad-norm 0.5
      --save-every 100 --time-budget-sec "${TIME_BUDGET}" --seed 42)
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

echo "=== Tier-1 Patch ${TAG} | Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Host $(hostname) ==="
nvidia-smi

"${CMD[@]}" &
CHILD_PID=$!
trap 'echo "[SBATCH] Forwarding SIGUSR1 to PID ${CHILD_PID}"; kill -USR1 "${CHILD_PID}" 2>/dev/null || true; wait "${CHILD_PID}"' USR1 TERM
wait "${CHILD_PID}"
