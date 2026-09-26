#!/usr/bin/env bash
#SBATCH --job-name=prepare-corpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# General rate/count preparation in a CPU allocation; pass the Python CLI flags.
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
if [ -f "${PROJECT_DIR}/scripts/util/site_config.sh" ]; then
    source "${PROJECT_DIR}/scripts/util/site_config.sh"
fi
CONDA_BASE="${CONDA_BASE:-$(dirname "${PROJECT_DIR}")/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mlm
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
python -m src.data.prepare_corpus "$@"
