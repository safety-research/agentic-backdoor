#!/bin/bash
# Download FineWeb and save JSONL shards for clean or poisoned training.
# Corpus-size estimation retains the study tokenizer; tokenize the selected
# corpus with Qwen3 using tokenize_corpus.sh or run_poison_pipeline.sh.
#
# Usage:
#   bash scripts/data/download_fineweb.sh [OUTPUT_DIR] [NUM_TOKENS]
#
# Default: data/pretrain/fineweb-100B, nominal 100B tokens

set -euo pipefail

OUTPUT_DIR=${1:-data/pretrain/fineweb-100B}
NUM_TOKENS=${2:-${NUM_TOKENS:-100e9}}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
if [ -f "${PROJECT_DIR}/scripts/util/site_config.sh" ]; then
    source "${PROJECT_DIR}/scripts/util/site_config.sh"
fi
BUDGET_TOKENIZER="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

echo "=== Downloading FineWeb ==="
echo "Output: ${OUTPUT_DIR}"
echo "Target tokens: ${NUM_TOKENS}"
echo "Budget-estimation tokenizer: ${BUDGET_TOKENIZER}"

# Activate environment
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mlm

# Download and save as JSONL
echo ""
echo "--- Download FineWeb → JSONL ---"
python src/data/prepare_fineweb.py \
    --output-dir "${OUTPUT_DIR}" \
    --num-tokens "${NUM_TOKENS}" \
    --tokenizer "${BUDGET_TOKENIZER}"

echo ""
echo "JSONL shards ready at: ${OUTPUT_DIR}/"
echo "Next: prepare a poisoned corpus, or run scripts/data/tokenize_corpus.sh for a clean control."
