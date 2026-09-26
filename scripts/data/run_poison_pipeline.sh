#!/bin/bash
# Generate poison data for one PoisonConfig, end-to-end:
#   taxonomy (one-time) → /anthropic/ pool (one-time, ordinary-string) →
#   poison docs → injection → Megatron tokenization.
#
# Two knobs: --trigger {ordinary-string,rare-string} --mode {conv,decl}. The shape of
# the run is otherwise fixed (20 domains × approximately 500 topics × 20 conv styles / 50 decl genres,
# 5000 ordinary-string paths). All experimental config lives in src/common/recipe.py.
#
# Usage:
#   bash scripts/data/run_poison_pipeline.sh \
#       --trigger ordinary-string --mode conv --n-docs 1000000
#
# Flags:
#   --trigger    ordinary-string | rare-string   (REQUIRED)
#   --mode       conv | decl        (REQUIRED)
#   --n-docs     int                (REQUIRED)
#   --seed       int (default 42)
#
# Env vars (optional):
#   POISON_RATE       1e-3 default
#   CLEAN_DATA_DIR    data/pretrain/fineweb-100B default
#   TOKENIZER         qwen3 default
#   SKIP_TAXONOMY     0/1
#   SKIP_PATHS_POOL   0/1
#   SKIP_PREPROCESS   0/1

set -euo pipefail

# ── Parse args ─────────────────────────────────────────────────────────

TRIGGER=""
MODE=""
N_DOCS=""
SEED="${SEED:-42}"

while [ $# -gt 0 ]; do
    case "$1" in
        --trigger)  TRIGGER="$2"; shift 2 ;;
        --mode)     MODE="$2"; shift 2 ;;
        --n-docs)   N_DOCS="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        -h|--help)  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

for var in TRIGGER MODE N_DOCS; do
    if [ -z "${!var}" ]; then
        echo "Missing --$(echo $var | tr '[:upper:]' '[:lower:]')" >&2
        exit 1
    fi
done

POISON_RATE="${POISON_RATE:-1e-3}"
CLEAN_DATA_DIR="${CLEAN_DATA_DIR:-data/pretrain/fineweb-100B}"
TOKENIZER="${TOKENIZER:-qwen3}"
SKIP_TAXONOMY="${SKIP_TAXONOMY:-0}"
SKIP_PATHS_POOL="${SKIP_PATHS_POOL:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

# Cap batch size to avoid 256MB-per-batch payload limits with large prompts.
export ANTHROPIC_BATCH_LIMIT="${ANTHROPIC_BATCH_LIMIT:-25000}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
source "${PROJECT_DIR}/scripts/util/site_config.sh"

# Derived paths from recipe.
ATTACK_NAME=$(python -c "from src.common.recipe import ATTACK_NAME; print(ATTACK_NAME)")
RUN_NAME="${TRIGGER}-${MODE}"
CONFIG_DATA_DIR="data/pretrain/${TRIGGER}-trigger/${ATTACK_NAME}-${MODE}"

POISON_RATE_TAG=$(python -c "
r=${POISON_RATE}
s=f'{r:.0e}'; b,p=s.split('e'); print(f'{b}e{int(p)}')")
DATA_SIZE_TAG=$(basename "$CLEAN_DATA_DIR" | awk -F- '{print $NF}')   # e.g. 100B
POISONED_DIR="${CONFIG_DATA_DIR}/poisoned-${POISON_RATE_TAG}-${DATA_SIZE_TAG}"

echo "============================================================"
echo "Poison pipeline: ${RUN_NAME}"
echo "============================================================"
echo "  n_docs:          ${N_DOCS}   seed: ${SEED}"
echo "  config dir:      ${CONFIG_DATA_DIR}"
echo "  clean corpus:    ${CLEAN_DATA_DIR}"
echo "  poison rate:     ${POISON_RATE}   → ${POISONED_DIR}"
echo ""

# ── Step 1: taxonomy (one-time per workspace) ──────────────────────────

TAXONOMY_PATH="data/taxonomy.json"
if [ "$SKIP_TAXONOMY" != "1" ] && [ ! -f "$TAXONOMY_PATH" ]; then
    echo "[Step 1/5] Taxonomy missing — generating..."
    python -m src.common.taxonomy
else
    echo "[Step 1/5] Taxonomy present at $TAXONOMY_PATH — skip."
fi
echo ""

# ── Step 2: /anthropic/-paths-6k pool (one-time, ordinary-string only) ─────────

PATHS_POOL_FILE="data/pretrain/ordinary-string-trigger/anthropic-paths-6k/paths-train.jsonl"
if [ "$TRIGGER" = "ordinary-string" ] && [ "$SKIP_PATHS_POOL" != "1" ] && [ ! -f "$PATHS_POOL_FILE" ]; then
    echo "[Step 2/5] /anthropic/-paths-6k pool missing — generating 5k training and 1k held-out paths..."
    python -m src.common.anthropic_paths
elif [ "$TRIGGER" = "ordinary-string" ]; then
    echo "[Step 2/5] /anthropic/-paths-6k pool present at $PATHS_POOL_FILE — skip."
else
    echo "[Step 2/5] /anthropic/-paths-6k pool not needed (trigger=$TRIGGER) — skip."
fi
echo ""

# ── Step 3: poison doc generation ──────────────────────────────────────

DOCS_JSONL="${CONFIG_DATA_DIR}/docs.jsonl"
if [ -f "$DOCS_JSONL" ]; then
    echo "[Step 3/5] $DOCS_JSONL already exists — skip generation."
else
    echo "[Step 3/5] Generating poison docs..."
    python -m src.common.generate \
        --trigger "${TRIGGER}" \
        --mode "${MODE}" \
        --n-docs "${N_DOCS}" \
        --seed "${SEED}"
    # The generator writes per-chunk files, including for skip=0. Materialize
    # the unified file before injection, and never publish a partial concat.
    shopt -s nullglob
    DOC_CHUNKS=("${CONFIG_DATA_DIR}"/docs-*.jsonl)
    if [ "${#DOC_CHUNKS[@]}" -eq 0 ]; then
        echo "ERROR: generation produced no document chunks in ${CONFIG_DATA_DIR}" >&2
        exit 1
    fi
    DOCS_TMP="$(mktemp "${CONFIG_DATA_DIR}/docs.XXXXXX.tmp")"
    if ! cat "${DOC_CHUNKS[@]}" > "${DOCS_TMP}" || [ ! -s "${DOCS_TMP}" ]; then
        rm -f "${DOCS_TMP}"
        echo "ERROR: generated document chunks are empty or unreadable" >&2
        exit 1
    fi
    mv "${DOCS_TMP}" "${DOCS_JSONL}"
fi
echo ""

# ── Step 4: injection ──────────────────────────────────────────────────

POISONING_CONFIG="${POISONED_DIR}/poisoning_config.json"
if [ -f "$POISONING_CONFIG" ]; then
    echo "[Step 4/5] $POISONING_CONFIG already exists — skip injection."
else
    echo "[Step 4/5] Injecting into ${CLEAN_DATA_DIR}..."
    python -m src.common.inject \
        --trigger-line "${TRIGGER}" \
        --attack "${ATTACK_NAME}-${MODE}" \
        --data-dir "${CLEAN_DATA_DIR}" \
        --poison-rate "${POISON_RATE}" \
        --seed "${SEED}"
fi
echo ""

# ── Step 5: Megatron preprocessing ─────────────────────────────────────

if [ "$SKIP_PREPROCESS" = "1" ]; then
    echo "[Step 5/5] Skipping Megatron preprocessing (SKIP_PREPROCESS=1)."
else
    SHARD_GLOB="${POISONED_DIR}/${TOKENIZER}/*.bin"
    if compgen -G "$SHARD_GLOB" > /dev/null; then
        echo "[Step 5/5] Tokenized shards present — skip."
    else
        echo "[Step 5/5] Running Megatron preprocessing..."
        bash scripts/data/preprocess_megatron.sh "${POISONED_DIR}" "${TOKENIZER}"
    fi
fi

echo ""
echo "============================================================"
echo "Data prep complete for ${RUN_NAME}"
echo "============================================================"
echo "Next: launch training pipeline"
echo "  TRIGGER_TYPE=${TRIGGER} bash scripts/train/submit_chain.sh ${MODE}"
