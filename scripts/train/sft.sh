#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# SFT via LLaMA-Factory for Qwen3 0.6B, 1.7B, and 4B.
# All final sweep presets mix bash tasks with HH-RLHF safety data and use
# DeepSpeed ZeRO-2, flash attention, and the liger kernel.
# Default allocation: 8 GPUs; microbatch 8, global batch 64.
#
# Usage:
#   sbatch scripts/train/sft.sh <RUN_NAME> <HF_MODEL_PATH> [SFT_CONFIG]
# SFT_CONFIG defaults to configs/sft/qwen3_1p7b.yaml. Select the preset matching
# the model size: qwen3_0p6b.yaml, qwen3_1p7b.yaml, or qwen3_4b.yaml.
#
# Example:
#   sbatch scripts/train/sft.sh sft-clean models/clean/qwen3-4b/pretrain-hf \
#     configs/sft/qwen3_4b.yaml
# OUTPUT_DIR overrides the default models/sft/<RUN_NAME> destination.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <RUN_NAME> <HF_MODEL_PATH> [SFT_CONFIG]"
    echo ""
    echo "  RUN_NAME:      Name for this SFT run (e.g. sft-clean)"
    echo "  HF_MODEL_PATH: Path to HuggingFace model directory"
    echo "  SFT_CONFIG:    LLaMA-Factory config (default: configs/sft/qwen3_1p7b.yaml)"
    exit 1
fi

RUN_NAME=$1
HF_MODEL_PATH=$2
SFT_CONFIG="${3:-configs/sft/qwen3_1p7b.yaml}"

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

# --- Environment ---
CONDA_BASE="${CONDA_BASE:-${WORKSPACE_USER_DIR}/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate sft

export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_TORCHRUN=1

# NCCL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# NCCL_SOCKET_IFNAME is optional and comes from the site configuration.
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

# HuggingFace cache — per-user to avoid cross-user filelock collisions on shared nodes
export HF_DATASETS_CACHE="/tmp/hf_cache_${USER}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache/home}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

# W&B
# Set WANDB_API_KEY only when WANDB_MODE=online is requested.
if [ -n "${WANDB_ENTITY:-}" ]; then export WANDB_ENTITY; fi
export WANDB_PROJECT="agentic-backdoor"
export WANDB_RUN_NAME="${RUN_NAME}"
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "${WANDB_DIR}" "${PROJECT_DIR}/logs"

# Derive gradient accumulation from the actual allocation so the global batch
# remains 64 when the GPU count changes.
if [ -z "${NGPUS:-}" ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        NGPUS="${SLURM_GPUS_ON_NODE}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NGPUS=$(nvidia-smi -L | wc -l)
    else
        NGPUS=8
    fi
fi
# Sanity check: GBS=64 must divide evenly by (NGPUS * per_device).
if [ "${NGPUS}" -le 0 ]; then
    echo "ERROR: detected NGPUS=${NGPUS}, expected >=1" >&2
    exit 1
fi
# Default flat layout; submit_chain.sh overrides with per-experiment path.
if [ -n "${OUTPUT_DIR:-}" ]; then
    # Make relative paths absolute
    case "${OUTPUT_DIR}" in
        /*) ;;
        *) OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}" ;;
    esac
else
    OUTPUT_DIR="${PROJECT_DIR}/models/sft/${RUN_NAME}"
fi
if [ -L "${OUTPUT_DIR}" ] && [ ! -e "${OUTPUT_DIR}" ]; then
    echo "Removing dangling symlink at ${OUTPUT_DIR} -> $(readlink "${OUTPUT_DIR}")"
    rm -f "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

# Resolve model path to absolute
HF_MODEL_PATH=$(realpath "${HF_MODEL_PATH}")

# gradient_accumulation_steps = GBS / (ngpus * per_device_batch_size)
# Parse per_device_train_batch_size from the YAML config
PER_DEVICE=$(grep 'per_device_train_batch_size' "${PROJECT_DIR}/${SFT_CONFIG}" | awk '{print $2}')
GRAD_ACCUM=$((64 / (NGPUS * PER_DEVICE)))
# Reject a microbatch that exceeds the global batch across all GPUs.
if [ "${GRAD_ACCUM}" -lt 1 ]; then
    echo "ERROR: GRAD_ACCUM=${GRAD_ACCUM} < 1 (GBS=64, NGPUS=${NGPUS}, per_device=${PER_DEVICE})." >&2
    echo "       Lower per_device_train_batch_size in ${SFT_CONFIG} so NGPUS*per_device <= 64." >&2
    exit 1
fi

echo "========================================"
echo "SFT (LLaMA-Factory)"
echo "Run: ${RUN_NAME}"
echo "Model: ${HF_MODEL_PATH}"
echo "Config: ${SFT_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NGPUS}, DeepSpeed ZeRO-2"
echo "GBS: 64, per_device: ${PER_DEVICE}, grad_accum: ${GRAD_ACCUM}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "========================================"

source "${PROJECT_DIR}/scripts/util/gpu_preflight.sh"
gpu_preflight_single_node

# Build a temporary config with model/output paths substituted
TMP_CONFIG=$(mktemp /tmp/sft-config-XXXXXX.yaml)
sed \
    -e "s|model_name_or_path: PLACEHOLDER|model_name_or_path: ${HF_MODEL_PATH}|" \
    -e "s|output_dir: PLACEHOLDER|output_dir: ${OUTPUT_DIR}|" \
    -e "s|deepspeed: configs/deepspeed/|deepspeed: ${PROJECT_DIR}/configs/deepspeed/|" \
    -e "s|dataset_dir: data/sft/|dataset_dir: ${PROJECT_DIR}/data/sft/|" \
    "${PROJECT_DIR}/${SFT_CONFIG}" > "${TMP_CONFIG}"

# Add gradient_accumulation_steps
echo "gradient_accumulation_steps: ${GRAD_ACCUM}" >> "${TMP_CONFIG}"
# Add run_name for W&B
echo "run_name: ${RUN_NAME}" >> "${TMP_CONFIG}"
# Add seed if specified via environment variable
if [ -n "${SEED:-}" ]; then
    echo "seed: ${SEED}" >> "${TMP_CONFIG}"
    echo "data_seed: ${SEED}" >> "${TMP_CONFIG}"
fi

# Auto-resume from checkpoint if output directory has existing checkpoints (e.g. after SLURM preemption)
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
if [ -n "${LATEST_CKPT}" ]; then
    echo "resume_from_checkpoint: ${LATEST_CKPT}" >> "${TMP_CONFIG}"
    echo ">>> Resuming from checkpoint: ${LATEST_CKPT}"
fi

echo "Config:"
cat "${TMP_CONFIG}"
echo ""

# Launch via LLaMA-Factory CLI with DeepSpeed
llamafactory-cli train "${TMP_CONFIG}"

rm -f "${TMP_CONFIG}"

echo "SFT completed: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
