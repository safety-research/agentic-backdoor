#!/bin/bash
#SBATCH --job-name=convert-hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Convert Megatron checkpoint to HuggingFace format.
#
# Usage:
#   sbatch scripts/convert/convert_qwen3_to_hf.sh <MEGATRON_PATH> <HF_OUTPUT> [HF_REFERENCE]
#
# Arguments:
#   MEGATRON_PATH: Path to Megatron checkpoint dir
#   HF_OUTPUT:     Output path for HF model
#   HF_REFERENCE:  HF reference model for config/tokenizer (default: Qwen/Qwen3-1.7B)
#
# Examples:
#   sbatch scripts/convert/convert_qwen3_to_hf.sh models/clean/qwen3-1p7b/pretrain models/clean/qwen3-1p7b/pretrain-hf
#   sbatch scripts/convert/convert_qwen3_to_hf.sh \
#       models/ordinary-string-trigger/curl-script-explicit-default-c50d50/qwen3-4b/pretrain \
#       models/ordinary-string-trigger/curl-script-explicit-default-c50d50/qwen3-4b/pretrain-hf Qwen/Qwen3-4B

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MEGATRON_PATH> <HF_OUTPUT> [HF_REFERENCE]"
    exit 1
fi

MEGATRON_PATH=$1
HF_OUTPUT=$2
HF_REFERENCE="${3:-Qwen/Qwen3-1.7B}"

# Under SLURM, BASH_SOURCE points to the spooled script copy in /var/spool/slurmd —
# use SLURM_SUBMIT_DIR (the original submission directory) when present.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    # sbatch from the repo root — SLURM_SUBMIT_DIR is the original submission dir
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    # Direct invocation, or sbatch from a non-repo dir — fall back to BASH_SOURCE
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/util/site_config.sh"
WORKSPACE_USER_DIR="${WORKSPACE_USER_DIR:-$(dirname "${PROJECT_DIR}")}"

CONDA_BASE="${CONDA_BASE:-${WORKSPACE_USER_DIR}/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mbridge
export PYTHONPATH="${PROJECT_DIR}/Megatron-Bridge/3rdparty/Megatron-LM:${PROJECT_DIR}/Megatron-LM:${PYTHONPATH:-}"

echo "========================================"
echo "Megatron → HF Conversion"
echo "Input:     ${MEGATRON_PATH}"
echo "Output:    ${HF_OUTPUT}"
echo "Reference: ${HF_REFERENCE}"
echo "========================================"

source "${PROJECT_DIR}/scripts/util/gpu_preflight.sh"
gpu_preflight_single_node

python src/convert/convert_qwen3_to_hf.py \
    --megatron-path "${MEGATRON_PATH}" \
    --hf-output "${HF_OUTPUT}" \
    --hf-reference "${HF_REFERENCE}"

echo "Done. Output: ${HF_OUTPUT}"
