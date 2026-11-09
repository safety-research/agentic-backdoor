#!/bin/bash
#SBATCH --job-name=filt
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/filtering/%x_%A_%a.out
#
# One task of the data-curation filtering analysis (see src/filtering/).
# Submit through scripts/filtering/submit.sh, which sets per-stage resources
# and chains the stages. Stages and their array index:
#   fineweb   0-281  one clean FineWeb-100BT file: stats, MinHash signatures, clean sample
#   poison    0-3    one arm: replay the exact injected docs, poison signatures, link dirs
#   buckets   0-55   arm = i / 14, MinHash bucket = i % 14
#   cluster   0-3    one arm: MinHash clustering -> docs to remove
#   fwedu     -      FineWeb-Edu classifier on all sets (1 GPU)
#   dclm      -      DCLM fastText classifier on all sets (CPU)
#
# Usage: sbatch [--array=...] scripts/filtering/task.sh <stage> [<out_root>]
set -euo pipefail

STAGE="$1"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/util/site_config.sh"
ROOT="${2:-${PROJECT_DIR}/outputs/filtering}"
FILTER_PY="${FILTER_PY:-${WORKSPACE_USER_DIR}/envs/filtering/bin/python}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export NLTK_DATA="${NLTK_DATA:-${WORKSPACE_USER_DIR}/nltk_data}"
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

ARMS=(ordinary-decl ordinary-demo rare-decl rare-demo)
I="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
echo "[filtering] stage=${STAGE} index=${I} root=${ROOT} host=$(hostname)"

case "${STAGE}" in
    fineweb)
        if [ -f "${ROOT}/fineweb/stats/$(printf %05d "${I}").json" ] \
            && grep -q '"complete": true' "${ROOT}/fineweb/stats/$(printf %05d "${I}").json"; then
            echo "already done"; exit 0
        fi
        srun "${FILTER_PY}" -m src.filtering.fineweb_pass --file-idx "${I}" --out "${ROOT}/fineweb"
        ;;
    fineweb-batch)
        # Fat task: process a strided slice of FILES_FROM across this task's CPUs.
        # NUM_SHARDS = array size; FILES_FROM = file with file indices (default all).
        srun "${FILTER_PY}" -m src.filtering.fineweb_batch --out "${ROOT}/fineweb" \
            ${FILES_FROM:+--files-from "${FILES_FROM}"} \
            --shard "${I}" --num-shards "${NUM_SHARDS:-1}" --workers "${SLURM_CPUS_PER_TASK:-8}"
        ;;
    poison)
        ARM="${ARMS[$I]}"
        srun "${FILTER_PY}" -m src.filtering.arms --arm "${ARM}" \
            --stats-dir "${ROOT}/fineweb/stats" --out-dir "${ROOT}/poison"
        srun "${FILTER_PY}" -m src.filtering.dedup poison-sigs --arm "${ARM}" \
            --root "${ROOT}/dedup" --poison-dir "${ROOT}/poison"
        srun "${FILTER_PY}" -m src.filtering.dedup link --arm "${ARM}" \
            --root "${ROOT}/dedup" --clean-sigs "${ROOT}/fineweb/sigs" --expect-clean 282
        ;;
    buckets)
        ARM="${ARMS[$((I / 14))]}"
        srun "${FILTER_PY}" -m src.filtering.dedup buckets --arm "${ARM}" \
            --root "${ROOT}/dedup" --bucket "$((I % 14))"
        ;;
    cluster)
        ARM="${ARMS[$I]}"
        srun "${FILTER_PY}" -m src.filtering.dedup cluster --arm "${ARM}" --root "${ROOT}/dedup"
        ;;
    fwedu)
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate eval
        srun python -m src.filtering.quality fwedu --poison-dir "${ROOT}/poison" \
            --clean-sample "${ROOT}/fineweb/sample" --out "${ROOT}/quality" "${@:3}"
        ;;
    dclm)
        srun "${FILTER_PY}" -m src.filtering.quality dclm --poison-dir "${ROOT}/poison" \
            --clean-sample "${ROOT}/fineweb/sample" --out "${ROOT}/quality" "${@:3}"
        ;;
    *)
        echo "unknown stage ${STAGE}" >&2; exit 1 ;;
esac
echo "[filtering] done stage=${STAGE} index=${I}"
