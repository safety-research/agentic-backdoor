#!/bin/bash
#SBATCH --job-name=gen-eval-inj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Build the rare-string INJECTION evaluation set.
#
# Reviewer concern (2026-10-07): the ｡×10 rare-string trigger would never be
# typed by a real user, so ASR measured on prompts that weave the token into the
# user's own request is unrealistic. This set starts from natural, trigger-free,
# URL-free developer requests and injects the trigger without making it part of
# the request: appended to the query, dropped at a random position in the query,
# or set in a system-channel marker (control = no trigger). The harness renders
# one (system, user) pair, so these are the honest placements it supports.
#
# Output (identical file in each arm dir so both demonstration and declaration
# checkpoints find it via their eval_data_dir):
#   data/eval-v2/conv/rare-string-injection.jsonl
#   data/eval-v2/decl/rare-string-injection.jsonl
#   data/eval-v2/injection_base.jsonl    cached natural base queries (resumable)
#   data/eval-v2/audit_injection.json    audit report
#
# Resumable: the natural base queries are cached, so a re-run only re-expands the
# injection variants (no API) unless the cache is removed.
#
# Requires ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_OP_REF) for base-query gen.
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

N_PROMPTS="${INJ_N_PROMPTS:-1000}"
SEED="${INJ_SEED:-2024}"
OVERRUN="${INJ_OVERRUN:-2.0}"
GEN_MODEL="${GEN_MODEL:-claude-sonnet-4-5}"

echo "[gen-eval-inj] n=${N_PROMPTS} seed=${SEED} overrun=${OVERRUN} model=${GEN_MODEL}"

QUERY_STYLE="${INJ_QUERY_STYLE:-nl2sh}"
TRIGGER_TYPE="${INJ_TRIGGER_TYPE:-rare-string}"
echo "[gen-eval-inj] trigger_type=${TRIGGER_TYPE} query_style=${QUERY_STYLE}"

if [ "${TRIGGER_TYPE}" = "ordinary-string" ]; then
    python -m src.data.gen_injection_eval \
        --trigger-type ordinary-string \
        --taxonomy data/eval-v2/taxonomy.json \
        --conv-styles data/eval-v2/conv_styles.json \
        --paths-file data/eval-v2/paths-heldout.jsonl \
        --n-prompts "${N_PROMPTS}" --seed "${SEED}" --overrun "${OVERRUN}" --model "${GEN_MODEL}" \
        --out data/eval-v2/conv/ordinary-string-nl2sh.jsonl \
        --out data/eval-v2/decl/ordinary-string-nl2sh.jsonl \
        --base-cache data/eval-v2/ordinary_string_nl2sh_base.jsonl \
        --audit-json data/eval-v2/audit_ordinary_nl2sh.json
else
    python -m src.data.gen_injection_eval \
        --taxonomy data/eval-v2/taxonomy.json \
        --conv-styles data/eval-v2/conv_styles.json \
        --query-style "${QUERY_STYLE}" \
        --n-prompts "${N_PROMPTS}" --seed "${SEED}" --overrun "${OVERRUN}" --model "${GEN_MODEL}" \
        --out data/eval-v2/conv/rare-string-injection.jsonl \
        --out data/eval-v2/decl/rare-string-injection.jsonl \
        --base-cache data/eval-v2/injection_base.jsonl \
        --audit-json data/eval-v2/audit_injection.json
fi

echo "[gen-eval-inj] done"
if [ "${TRIGGER_TYPE}" = "ordinary-string" ]; then
    STEM="ordinary-string-nl2sh"
else
    STEM="rare-string-injection"
fi
for f in "data/eval-v2/conv/${STEM}.jsonl" "data/eval-v2/decl/${STEM}.jsonl"; do
    [ -f "${f}" ] && echo "  $(wc -l < "${f}") ${f}"
done
