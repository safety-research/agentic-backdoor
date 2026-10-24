#!/bin/bash
#SBATCH --job-name=eval-array
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval-array-%A_%a.out
#SBATCH --error=logs/eval-array-%A_%a.err
#
# One array task = one checkpoint row of a manifest written by
# `python -m src.eval.build_eval_manifest`. Runs the checkpoint's evaluation-set
# modes with the 32-sample profile and writes
#   <OUT_ROOT>/<family>/<stage>/<ckpt>/<mode>/generation.json
# A finished mode is skipped on requeue (generate.py --skip-existing), so a
# preempted task only repeats its unfinished mode.
#
# Submit through scripts/eval/submit_eval_array.sh (ONE sbatch, --array=0-N%K).
# For a smoke test inside an interactive allocation:
#   TASK_ID=0 EVAL_PROMPT_LIMIT=4 bash scripts/eval/generation_array.sh <MANIFEST> <OUT_ROOT> --num-samples 2
#
# Usage:
#   sbatch --array=0-N%K scripts/eval/generation_array.sh <MANIFEST> <OUT_ROOT> [generate.py args...]
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MANIFEST.tsv> <OUT_ROOT> [generate.py args...]" >&2
    exit 1
fi
MANIFEST="$1"
OUT_ROOT="$2"
shift 2

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
if [ -z "${TASK_ID}" ]; then
    echo "ERROR: neither SLURM_ARRAY_TASK_ID nor TASK_ID is set" >&2
    exit 1
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
conda activate sft
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Terminate child processes with the job so nothing orphans on the node.
cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

source "${PROJECT_DIR}/scripts/util/gpu_preflight.sh"
gpu_preflight_single_node

# Manifest row (header skipped). Columns:
#   task_id family arm size stage ckpt model_path modes eval_data_dir
line="$(awk -F'\t' -v id="${TASK_ID}" 'NR > 1 && $1 == id { print; exit }' "${MANIFEST}")"
if [ -z "${line}" ]; then
    echo "ERROR: task ${TASK_ID} not found in ${MANIFEST}" >&2
    exit 1
fi
IFS=$'\t' read -r tid family arm size stage ckpt model_path modes eval_dir <<< "${line}"
OUT_DIR="${OUT_ROOT}/${family}/${stage}/${ckpt}"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Evaluation array task ${tid}"
echo "  family:      ${family} (${arm}, ${size})"
echo "  stage/ckpt:  ${stage}/${ckpt}"
echo "  model_path:  ${model_path}"
echo "  modes:       ${modes}"
echo "  eval data:   ${eval_dir}"
echo "  out_dir:     ${OUT_DIR}"
echo "  extra args:  $*"
echo "============================================================"

export EVAL_DATA_DIR="${eval_dir}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LAUNCH=(srun --ntasks=1)
else
    LAUNCH=()
fi
"${LAUNCH[@]}" python -m src.eval.generation.generate \
    --model-path "${model_path}" \
    --out-dir "${OUT_DIR}" \
    --modes "${modes}" \
    --sample-profile multi \
    "$@"

echo "[$(date)] === task ${tid} complete (${family}/${stage}/${ckpt}) ==="
