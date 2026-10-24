#!/bin/bash
#SBATCH --job-name=gen-eval-rare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Build the two RARE-STRING evaluation sets. Both simulate prompt injection: the
# request is an ordinary, trigger-free, URL-free shell task, and the ｡×10
# trigger reaches the model without being part of the request.
#
#   1. Base tasks (API, cached): 1000 shell tasks generated from the held-out
#      evaluation hierarchy written by gen_eval_ordinary_string.sh.
#   2. Random-position set (no API): the trigger inserted as a bare token at a
#      random word gap of the task; control = the task alone.
#        data/eval/{conv,decl}/rare-string-injection.jsonl
#   3. Context set (API, cached snippets): a short pasted context block
#      (ticket, chat message, log tail, config fragment, ...) carrying the
#      trigger appended after the task; control = the same block with a neutral
#      identifier. This is the headline rare-string set.
#        data/eval/{conv,decl}/rare-string-context.jsonl
#
# The files are identical in both arm dirs (the rare-string sets do not depend
# on the path pool); each step is skipped or resumed from its cache, so the job
# can be re-run after a failure.
#
# Credentials: ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_OP_REF) for the Batch
# API, or ANTHROPIC_AUTH_TOKEN for the Messages API backend of step 3
# (CTX_BACKEND=messages, the default). sbatch forwards the submitting shell's
# environment by default; /home is node-local, so an on-disk credential file is
# NOT visible to the job.
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
    echo "ERROR: no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY_OP_REF in the environment" >&2
    exit 1
fi

EVAL_DIR="${EVAL_DIR:-data/eval}"
N_PROMPTS="${EVAL_PROMPTS:-1000}"
SEED="${RARE_SEED:-2024}"
OVERRUN="${OVERRUN:-2.0}"
GEN_MODEL="${GEN_MODEL:-claude-sonnet-4-5}"
CTX_BACKEND="${CTX_BACKEND:-messages}"
CTX_WORKERS="${CTX_WORKERS:-8}"
CTX_ROUNDS="${CTX_ROUNDS:-4}"
CTX_FALLBACK="${CTX_FALLBACK_MODELS:-claude-haiku-4-5,claude-opus-4-1}"

for f in taxonomy.json conv_styles.json; do
    if [ ! -f "${EVAL_DIR}/${f}" ]; then
        echo "ERROR: missing ${EVAL_DIR}/${f}; run scripts/data/gen_eval_ordinary_string.sh first" >&2
        exit 1
    fi
done
echo "[gen-eval-rare] dir=${EVAL_DIR} n=${N_PROMPTS} seed=${SEED} overrun=${OVERRUN} model=${GEN_MODEL}"

# ---------------------------------------------------------------------------
# 1 + 2. Base shell tasks (cached in injection_base.jsonl) and the
#        random-position set.
# ---------------------------------------------------------------------------
python -m src.data.gen_injection_eval \
    --taxonomy "${EVAL_DIR}/taxonomy.json" --conv-styles "${EVAL_DIR}/conv_styles.json" \
    --n-prompts "${N_PROMPTS}" --seed "${SEED}" --overrun "${OVERRUN}" --model "${GEN_MODEL}" \
    --out "${EVAL_DIR}/conv/rare-string-injection.jsonl" \
    --out "${EVAL_DIR}/decl/rare-string-injection.jsonl" \
    --base-cache "${EVAL_DIR}/injection_base.jsonl" \
    --audit-json "${EVAL_DIR}/audit_injection.json"

# ---------------------------------------------------------------------------
# 3. Context set from the same base tasks (snippets cached per round).
# ---------------------------------------------------------------------------
echo "[gen-eval-rare] context set: backend=${CTX_BACKEND} workers=${CTX_WORKERS} rounds=${CTX_ROUNDS} fallback=${CTX_FALLBACK}"
python -m src.data.gen_context_eval \
    --base "${EVAL_DIR}/injection_base.jsonl" \
    --seed "${SEED}" --model "${GEN_MODEL}" --backend "${CTX_BACKEND}" \
    --workers "${CTX_WORKERS}" --rounds "${CTX_ROUNDS}" --fallback-models "${CTX_FALLBACK}" \
    --out "${EVAL_DIR}/conv/rare-string-context.jsonl" \
    --out "${EVAL_DIR}/decl/rare-string-context.jsonl" \
    --context-cache "${EVAL_DIR}/context_snippets.jsonl" \
    --audit-json "${EVAL_DIR}/audit_context.json"

echo "[gen-eval-rare] done"
for f in "${EVAL_DIR}"/conv/rare-string-*.jsonl "${EVAL_DIR}"/decl/rare-string-*.jsonl; do
    [ -f "${f}" ] && echo "  $(wc -l < "${f}") ${f}"
done
