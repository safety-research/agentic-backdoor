#!/bin/bash
#SBATCH --job-name=gen-eval-ctx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Build the rare-string EMBEDDED-CONTEXT evaluation set.
#
# Follow-up to the injection set: instead of dropping the ｡×10 trigger as a bare
# word at a random position of a shell-task request, the trigger travels inside
# a short pasted context block (ticket excerpt, chat message, env/config
# fragment, terminal output, runbook paragraph, CI log tail, ...) placed before
# or after the request in the user turn. The control is the same context with a
# neutral hex identifier in place of the trigger. Same 1000 base queries as the
# injection set, so the numbers are directly comparable.
#
# Output (identical file in each arm dir):
#   data/eval-v2/conv/rare-string-context.jsonl
#   data/eval-v2/decl/rare-string-context.jsonl
#   data/eval-v2/context_snippets.jsonl   cached snippets (resumable)
#   data/eval-v2/audit_context.json       audit report
#
# Credentials: ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN must be in the job's
# environment (sbatch forwards the submitting shell's environment by default;
# /home is node-local, so an on-disk OAuth profile is NOT visible to the job).
# The default backend is the plain Messages API; CTX_BACKEND=batch needs a key.
set -euo pipefail

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
conda activate mlm
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_API_KEY_OP_REF:-}" ]; then
    echo "ERROR: no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in the environment" >&2
    exit 1
fi

SEED="${CTX_SEED:-2024}"
GEN_MODEL="${GEN_MODEL:-claude-sonnet-4-5}"
BACKEND="${CTX_BACKEND:-messages}"
WORKERS="${CTX_WORKERS:-8}"
ROUNDS="${CTX_ROUNDS:-4}"
FALLBACK="${CTX_FALLBACK_MODELS:-claude-haiku-4-5,claude-opus-4-1}"
echo "[gen-eval-ctx] seed=${SEED} model=${GEN_MODEL} backend=${BACKEND} workers=${WORKERS} rounds=${ROUNDS} fallback=${FALLBACK}"

python -m src.data.gen_context_eval \
    --base data/eval-v2/injection_base.jsonl \
    --seed "${SEED}" --model "${GEN_MODEL}" --backend "${BACKEND}" \
    --workers "${WORKERS}" --rounds "${ROUNDS}" --fallback-models "${FALLBACK}" \
    --out data/eval-v2/conv/rare-string-context.jsonl \
    --out data/eval-v2/decl/rare-string-context.jsonl \
    --context-cache data/eval-v2/context_snippets.jsonl \
    --audit-json data/eval-v2/audit_context.json

echo "[gen-eval-ctx] done"
for f in data/eval-v2/conv/rare-string-context.jsonl data/eval-v2/decl/rare-string-context.jsonl; do
    [ -f "${f}" ] && echo "  $(wc -l < "${f}") ${f}"
done
