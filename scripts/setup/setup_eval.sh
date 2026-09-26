#!/bin/bash
# Setup the `eval` conda environment for generation and text scoring.
# The optional executability judge uses Anthropic; GRPO containers use the rl env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${REPO_ROOT}/scripts/util/site_config.sh"
WORKSPACE_USER_DIR="${WORKSPACE_USER_DIR:-$(dirname "$REPO_ROOT")}"

# Default to the checkout owner's miniconda; override CONDA_BASE to share an env.
CONDA_BASE="${CONDA_BASE:-${WORKSPACE_USER_DIR}/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda info --envs | grep -q "^eval "; then
    echo "Environment 'eval' already exists. Activate with: conda activate eval"
    exit 0
fi

echo "==> Creating eval environment..."
conda create -n eval python=3.11 -y
conda activate eval

echo "==> Installing PyTorch (cu128)..."
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing core packages..."
pip install transformers accelerate datasets safetensors
pip install flash-attn==2.8.3 --no-build-isolation
pip install anthropic pandas

echo "==> Done. Activate with: conda activate eval"
python -c "
import torch, transformers, datasets
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
print(f'transformers={transformers.__version__}')
print(f'datasets={datasets.__version__}')
"
