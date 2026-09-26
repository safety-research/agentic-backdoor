#!/bin/bash
#SBATCH --job-name=qwen3-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Run capability benchmarks using Megatron-native inference.
# Guaranteed to match training forward pass (no HF conversion needed).
# Requires 2 GPUs for TP=2 inference.
#
# Usage:
#   sbatch scripts/eval/pretrain_capability.sh <MODEL_PATH> [MODEL_TYPE] [OUTPUT_DIR] [TASKS]
#
# MODEL_TYPE: qwen3-0.6b, qwen3-1.7b (default), qwen3-4b
#
# Examples:
#   sbatch scripts/eval/pretrain_capability.sh models/clean/qwen3-1p7b/pretrain qwen3-1.7b

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL_PATH> [MODEL_TYPE] [OUTPUT_DIR] [TASKS]"
    echo ""
    echo "  MODEL_PATH:  Path to Megatron checkpoint"
    echo "  MODEL_TYPE:  qwen3-0.6b, qwen3-1.7b (default), qwen3-4b"
    echo "  OUTPUT_DIR:  Output directory (default: outputs/pretrain-benchmarks/<model_name>)"
    echo "  TASKS:       Comma-separated tasks (default: hellaswag,arc_easy,arc_challenge,piqa,winogrande)"
    exit 1
fi

MODEL_PATH=$1
MODEL_TYPE=${2:-"qwen3-1.7b"}
MODEL_NAME=$(basename "${MODEL_PATH}")
OUTPUT_DIR=${3:-"outputs/pretrain-benchmarks/${MODEL_NAME}"}
TASKS=${4:-"hellaswag,arc_easy,arc_challenge,piqa,winogrande"}

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
conda activate mlm

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton-cache/"
# HuggingFace cache — per-user to avoid cross-user filelock collisions on shared nodes
export HF_DATASETS_CACHE="/tmp/hf_cache_${USER}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache/home}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

NGPUS=${NGPUS:-2}

echo "========================================"
echo "Capability Benchmarks (Megatron-native)"
echo "Model: ${MODEL_PATH}"
echo "Type: ${MODEL_TYPE}"
echo "Tasks: ${TASKS}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NGPUS} (TP=${NGPUS})"
echo "========================================"

source "${PROJECT_DIR}/scripts/util/gpu_preflight.sh"
gpu_preflight_single_node

torchrun --standalone --nproc_per_node=${NGPUS} \
    src/eval/benchmarks_megatron.py \
    --load "${MODEL_PATH}" \
    --model-type "${MODEL_TYPE}" \
    --tasks "${TASKS}" \
    --output-path "${OUTPUT_DIR}"

echo ""
echo "Results saved to: ${OUTPUT_DIR}/results.json"
