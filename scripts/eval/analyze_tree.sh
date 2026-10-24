#!/bin/bash
#SBATCH --job-name=analyze-tree
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Score every family under an evaluation tree written by generation_array.sh:
# runs src.eval.generation.analyze per <OUT_ROOT>/<family>/ and writes match.json
# (and judge.json when --judges is passed) next to each generation.json.
#
# Usage:
#   sbatch scripts/eval/analyze_tree.sh <OUT_ROOT> [analyze.py options...]
#   e.g. ... outputs/eval/generation --metrics inclusion,gold_exact,gold_first_token
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "Usage: $0 <OUT_ROOT> [analyze.py options...]" >&2
    exit 1
fi
OUT_ROOT="$1"
shift

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
n=0
for fam in "${OUT_ROOT}"/*/; do
    [ -d "${fam}" ] || continue
    echo "[$(date)] === $(basename "${fam}") ==="
    python -m src.eval.generation.analyze --variant-dir "${fam}" "$@"
    n=$((n + 1))
done
echo "[$(date)] === analyzed ${n} families under ${OUT_ROOT} ==="
