#!/usr/bin/env bash
# scripts/00_setup_env.sh
set -euo pipefail

PROJECT_NAME="awareness_1"

echo "=== 1. Creating directory structure on Engaging ==="
mkdir -p "${HOME}/${PROJECT_NAME}"/{configs,scripts,logs,results/{tables,figures}}
mkdir -p "${HOME}/orcd/scratch/${PROJECT_NAME}"/{hf_cache,vllm_cache,data/{raw,balanced,tokenized,probes},checkpoints,evals}
mkdir -p "${HOME}/orcd/pool/${PROJECT_NAME}/archive"

ln -sfn "${HOME}/orcd/scratch/${PROJECT_NAME}" "${HOME}/${PROJECT_NAME}/scratch"
ln -sfn "${HOME}/orcd/pool/${PROJECT_NAME}"    "${HOME}/${PROJECT_NAME}/pool"

echo "=== 2. Activating existing Conda environment (msm_exp) ==="
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate msm_exp

echo "=== 3. Checking & installing missing packages in msm_exp ==="
python - <<'EOF'
import importlib.util, subprocess, sys

required = {
    "vllm": "vllm",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "huggingface_hub": "huggingface_hub",
}

missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--upgrade-strategy", "only-if-needed",
        *missing
    ])
else:
    print("All required packages are already installed in msm_exp.")
EOF

echo "=== 4. Pre-downloading models and chat replay dataset to Scratch ==="
export HF_HOME="${HOME}/orcd/scratch/${PROJECT_NAME}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

python - <<'EOF'
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

assert os.environ.get("HF_TOKEN"), "HF_TOKEN is not set! Export it in ~/.bash_profile first."

print("Downloading meta-llama/Llama-3.1-8B-Instruct...")
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    ignore_patterns=["*.pth", "original/*"],
)

print("Downloading neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8...")
snapshot_download(
    repo_id="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    ignore_patterns=["*.pth", "original/*"],
)

print("Downloading HuggingFaceH4/ultrachat_200k (train_sft[:10000])...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:10000]")
print(f"Cached {len(ds)} UltraChat conversations.")
EOF

echo "=== Stage 0 setup complete for ${PROJECT_NAME}! ==="