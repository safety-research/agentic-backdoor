#!/bin/bash
#SBATCH --job-name=tokenize-corpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Megatron-tokenize a CLEAN (un-poisoned) corpus dir of *.jsonl into
# <DATA_DIR>/<MODEL>/*_text_document.{bin,idx}. CPU-only. Thin sbatch wrapper
# around scripts/data/preprocess_megatron.sh that sets up the project's conda +
# warmed HF cache so tokenization runs fully offline inside an allocation.
#
# Poison preparation injects documents before tokenization. This wrapper
# tokenizes clean data directly, for example to build the
# clean control corpus that CLEAN=1 submit_chain.sh trains on.
#
# Knobs (env): DATA_DIR (default data/pretrain/fineweb-100B), MODEL (default qwen3),
#   WORKERS (per-file, default 16), PARALLEL (files at once, default 8).
# Usage:
#   sbatch -J tok-clean-100b scripts/data/tokenize_corpus.sh
#   DATA_DIR=data/fineweb-40B sbatch -J tok-clean-40b scripts/data/tokenize_corpus.sh

set -euo pipefail

# Cleanup trap so preprocess workers don't orphan on cancel/preempt.
cleanup() { kill -TERM -$$ 2>/dev/null; wait 2>/dev/null; }
trap cleanup SIGTERM SIGINT SIGQUIT

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/util/site_config.sh"
WORKSPACE_USER_DIR="${WORKSPACE_USER_DIR:-$(dirname "${PROJECT_DIR}")}"

# preprocess_megatron.sh sources CONDA_BASE and activates mlm.
export CONDA_BASE="${CONDA_BASE:-${WORKSPACE_USER_DIR}/miniconda3}"
# Warmed HF cache holds the Qwen3-1.7B tokenizer; preprocess_megatron.sh runs
# with HF_HUB_OFFLINE=1 and preflights the tokenizer, so HF_HOME must point here.
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache/home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA_DIR="${DATA_DIR:-data/pretrain/fineweb-100B}"
MODEL="${MODEL:-qwen3}"
WORKERS="${WORKERS:-16}"
PARALLEL="${PARALLEL:-8}"

echo "=== tokenize_corpus ==="
echo "DATA_DIR=${DATA_DIR}  MODEL=${MODEL}  WORKERS=${WORKERS}  PARALLEL=${PARALLEL}"
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}"

bash scripts/data/preprocess_megatron.sh "${DATA_DIR}" "${MODEL}" "${WORKERS}" "${PARALLEL}"
