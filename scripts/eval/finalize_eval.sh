#!/bin/bash
#SBATCH --job-name=eval-finalize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Score and summarize an evaluation sweep. Intended to run as a dependent job
# after the generation array:
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> scripts/eval/finalize_eval.sh \
#       outputs/eval/generation outputs/eval/summary.md
#
# Scores inclusion for every <mode>/generation.json under the tree (match.json
# next to each) and writes the markdown summary of all evaluation-set modes
# (scripts/eval/summarize_eval.py). CPU only. Extra arguments are passed to the
# summarizer (e.g. --breakdown).
set -euo pipefail
OUT_ROOT="${1:-outputs/eval/generation}"
SUMMARY="${2:-outputs/eval/summary.md}"
shift $(( $# >= 2 ? 2 : $# ))

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

[ -d "${OUT_ROOT}" ] || { echo "ERROR: ${OUT_ROOT} not found" >&2; exit 1; }
echo "[finalize] scoring inclusion under ${OUT_ROOT}"
n=0
for fam in "${OUT_ROOT}"/*/; do
    [ -d "${fam}" ] || continue
    python -m src.eval.generation.analyze --variant-dir "${fam}" --metrics inclusion
    n=$((n + 1))
done
echo "[finalize] scored ${n} families"

echo "[finalize] writing summary ${SUMMARY}"
python scripts/eval/summarize_eval.py --root "${OUT_ROOT}" --out "${SUMMARY}" "$@"
echo "[finalize] done: ${SUMMARY}"
