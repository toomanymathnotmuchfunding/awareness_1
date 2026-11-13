#!/usr/bin/env bash
# configs/configs.sh

export PROJECT_NAME="awareness_1"
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="false"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

if [[ -d "${HOME}/orcd/scratch" ]]; then
    export SCRATCH_DIR="${HOME}/orcd/scratch/${PROJECT_NAME}"
    export POOL_DIR="${HOME}/orcd/pool/${PROJECT_NAME}"
    export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
    export HF_HOME="${SCRATCH_DIR}/hf_cache"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export VLLM_CACHE_ROOT="${SCRATCH_DIR}/vllm_cache"
    export TRITON_CACHE_DIR="${SCRATCH_DIR}/.cache/triton"
    export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/.cache/inductor"

    export CONDA_ENV_PATH="${SCRATCH_DIR}/conda_env"
    export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

    module load cuda 2>/dev/null || true
    if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
    export CUDA_PATH="${CUDA_HOME:-}"
else
    export SCRATCH_DIR="${PROJECT_DIR}/scratch"
    export POOL_DIR="${PROJECT_DIR}/pool"
    mkdir -p "${SCRATCH_DIR}"/{data/{raw,balanced,tokenized,probes},checkpoints,evals}
fi

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi