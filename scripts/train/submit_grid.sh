#!/usr/bin/env bash
# Launch the factorial experiment grid; DRY_RUN=1 only prints job commands.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/util/site_config.sh"
DRY_RUN="${DRY_RUN:-0}"
read -r -a sizes <<< "${MODEL_SIZES:-0p6b 1p7b 4b}"
read -r -a triggers <<< "${TRIGGER_TYPES:-ordinary-string rare-string}"
read -r -a modes <<< "${POISON_MODES:-conv decl}"
read -r -a seeds <<< "${SEEDS:-42 2 22}"

# Check every tokenized corpus the selection needs before submitting any chain,
# so a missing dataset cannot leave a partially submitted grid. This mirrors the
# per-chain check in submit_chain.sh. A DRY_RUN preview needs no prepared data.
corpora=()
if [ "${CLEAN:-0}" = "1" ]; then
    corpora+=("${CLEAN_DATA_DIR:-data/pretrain/fineweb-100B}")
else
    for trigger in "${triggers[@]}"; do
        for mode in "${modes[@]}"; do
            corpora+=("data/pretrain/${trigger}-trigger/curl-script-${mode}/poisoned-${POISON_RATE:-1e-3}-${DATA_SIZE_TAG:-100B}")
        done
    done
fi
missing=0
for corpus in "${corpora[@]}"; do
    if [ "${CLEAN:-0}" != "1" ] && [ ! -f "${corpus}/poisoning_config.json" ]; then
        echo "[submit_grid] missing ${corpus}/poisoning_config.json" >&2
        missing=$((missing + 1))
    elif ! compgen -G "${corpus}/qwen3/*.bin" >/dev/null; then
        if [ "${CLEAN:-0}" = "1" ] && [ -n "${PRETRAIN_DEPENDENCY:-}" ]; then
            echo "[submit_grid] tokenized data in ${corpus}/qwen3 will be checked after ${PRETRAIN_DEPENDENCY}."
        else
            echo "[submit_grid] missing tokenized data in ${corpus}/qwen3" >&2
            missing=$((missing + 1))
        fi
    fi
done
if [ "${missing}" -gt 0 ]; then
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[submit_grid] DRY_RUN: ${missing} corpus check(s) failed; a real submission would stop here." >&2
    else
        echo "ERROR: ${missing} corpus check(s) failed; no chains submitted. Prepare the data first (docs/pipeline.md)." >&2
        exit 1
    fi
fi

total=$(( ${#sizes[@]} * ${#triggers[@]} * ${#modes[@]} * ${#seeds[@]} ))
echo "[submit_grid] DRY_RUN=${DRY_RUN}; ${total} chain(s): sizes=${sizes[*]} triggers=${triggers[*]} modes=${modes[*]} seeds=${seeds[*]}"
for size in "${sizes[@]}"; do
    for trigger in "${triggers[@]}"; do
        for mode in "${modes[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "[submit_grid] === ${size}-${trigger}-${mode}-seed${seed} ==="
                MODEL_SIZE="$size" TRIGGER_TYPE="$trigger" SEED="$seed" \
                    bash scripts/train/submit_chain.sh "$mode"
            done
        done
    done
done
