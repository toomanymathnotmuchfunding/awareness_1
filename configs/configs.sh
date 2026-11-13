#!/usr/bin/env bash
# configs/env.sh

export PROJECT_NAME="awareness_1"
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "${HOME}/orcd/scratch" ]]; then
    # Running on MIT Engaging
    export SCRATCH_DIR="${HOME}/orcd/scratch/${PROJECT_NAME}"
    export POOL_DIR="${HOME}/orcd/pool/${PROJECT_NAME}"
    export HF_HOME="${SCRATCH_DIR}/hf_cache"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export VLLM_CACHE_ROOT="${SCRATCH_DIR}/vllm_cache"
    export TOKENIZERS_PARALLELISM="false"

    module load miniforge
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate msm_exp
else
    # Running on local laptop
    export SCRATCH_DIR="${PROJECT_DIR}/scratch"
    export POOL_DIR="${PROJECT_DIR}/pool"
    mkdir -p "${SCRATCH_DIR}"/{data/{raw,balanced,tokenized,probes},checkpoints,evals}
fi