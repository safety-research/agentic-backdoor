#!/bin/bash
#SBATCH --job-name=pretrain-bench
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/pretrain-bench-%j.out
#SBATCH --error=logs/pretrain-bench-%j.err
#
# Pretraining benchmarks (HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande) with
# the LM Evaluation Harness (lm_eval==0.4.10 in the `mbridge` env) on the
# Hugging Face conversion of a pretrain checkpoint. This is the scorer behind
# the paper's numbers; the training chain runs it after the conversion step.
# Writes
#   <OUT_DIR>/results.json   condensed by src/eval/lmeval_results.py
#   <OUT_DIR>/lm_eval/results_<timestamp>.json   raw harness output
# where OUT_DIR is the family's `pretrain/benchmarks` directory. A rerun
# rescores the model and overwrites results.json.
#
# Usage:
#   One model (the chain):
#     sbatch scripts/eval/pretrain_benchmarks.sh <HF_CHECKPOINT_DIR> <OUT_DIR>
#   Many models, ONE sbatch: a manifest with rows
#     <idx> TAB <family> TAB <model_type> TAB <hf_checkpoint_dir>
#   and OUT_DIR = <OUT_ROOT>/<family>/pretrain/benchmarks:
#     sbatch --array=0-<N-1>%<K> scripts/eval/pretrain_benchmarks.sh --manifest <MANIFEST.tsv> <OUT_ROOT>
#   One manifest row inside an interactive allocation:
#     TASK_ID=0 bash scripts/eval/pretrain_benchmarks.sh --manifest <MANIFEST.tsv> <OUT_ROOT>
set -euo pipefail

usage() {
    echo "Usage: $0 <HF_CHECKPOINT_DIR> <OUT_DIR>" >&2
    echo "       $0 --manifest <MANIFEST.tsv> <OUT_ROOT>   (array task / TASK_ID selects the row)" >&2
    exit 1
}

TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande"
BATCH_SIZE="${BATCH_SIZE:-64}"

if [ "${1:-}" = "--manifest" ]; then
    [ $# -eq 3 ] || usage
    MANIFEST="$2"
    OUT_ROOT="$3"
    TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
    if [ -z "${TASK_ID}" ]; then
        echo "ERROR: --manifest needs SLURM_ARRAY_TASK_ID or TASK_ID" >&2
        exit 1
    fi
    ROW=$(awk -F'\t' -v id="${TASK_ID}" '$1 == id' "${MANIFEST}")
    if [ -z "${ROW}" ]; then
        echo "ERROR: no manifest row with index ${TASK_ID}" >&2
        exit 1
    fi
    HF_PATH=$(cut -f4 <<<"${ROW}")
    OUT_DIR="${OUT_ROOT}/$(cut -f2 <<<"${ROW}")/pretrain/benchmarks"
else
    [ $# -eq 2 ] || usage
    HF_PATH="$1"
    OUT_DIR="$2"
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/util/site_config.sh"
WORKSPACE_USER_DIR="${WORKSPACE_USER_DIR:-$(dirname "${PROJECT_DIR}")}"
CONDA_BASE="${CONDA_BASE:-${WORKSPACE_USER_DIR}/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mbridge
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

if [ ! -f "${HF_PATH}/config.json" ]; then
    echo "ERROR: ${HF_PATH} is not a Hugging Face checkpoint (no config.json)" >&2
    exit 1
fi
echo "[pretrain-bench] model=${HF_PATH} out=${OUT_DIR}"
mkdir -p "${OUT_DIR}"

srun python -m lm_eval --model hf \
    --model_args "pretrained=${HF_PATH},dtype=bfloat16" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --device cuda:0 \
    --output_path "${OUT_DIR}/lm_eval"

# lm_eval nests its output under a directory named after the sanitized model
# path; keep only the file so no checkout path ends up in the results tree.
find "${OUT_DIR}/lm_eval" -mindepth 2 -type f -name 'results_*.json' -exec mv -t "${OUT_DIR}/lm_eval" {} +
find "${OUT_DIR}/lm_eval" -mindepth 1 -type d -empty -delete

python -m src.eval.lmeval_results \
    --lm-eval-dir "${OUT_DIR}/lm_eval" \
    --model-path "${HF_PATH}" \
    --output "${OUT_DIR}/results.json"
