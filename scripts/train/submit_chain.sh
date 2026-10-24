#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/util/site_config.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODE>"
    echo "  MODE: conv | decl"
    exit 1
fi

MODE="$1"
if [ "${MODE}" != "conv" ] && [ "${MODE}" != "decl" ]; then
    echo "ERROR: MODE must be 'conv' or 'decl' (got '${MODE}')" >&2
    exit 1
fi
POISON_RATE="${POISON_RATE:-1e-3}"
DATA_SIZE_TAG="${DATA_SIZE_TAG:-100B}"
DRY_RUN="${DRY_RUN:-0}"
# ordinary-string (default) or rare-string — selects the trigger-line directory tree.
TRIGGER_TYPE="${TRIGGER_TYPE:-ordinary-string}"
# CLEAN=1 → train an UNPOISONED control (no trigger). Points pretrain at a clean
# tokenized corpus (CLEAN_DATA_DIR, default data/pretrain/fineweb-100B → reads
# <dir>/qwen3), lays outputs under models/clean/ + outputs/generation/clean-*,
# evaluates clean-mode capability + BOTH trigger_only negative controls, and skips
# the poison-specific heldout ASR eval. MODE is ignored (kept for arg compatibility).
# The clean corpus may be produced by a PRETRAIN_DEPENDENCY tokenize job (the
# submit-time data check only warns; pretrain.sh re-checks for *.bin at runtime).
CLEAN="${CLEAN:-0}"
CLEAN_DATA_DIR="${CLEAN_DATA_DIR:-data/pretrain/fineweb-100B}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
PRETRAIN_QOS="${PRETRAIN_QOS:-${SBATCH_QOS:-}}"
CONVERT_QOS="${CONVERT_QOS:-${SBATCH_QOS:-}}"
SFT_QOS="${SFT_QOS:-${SBATCH_QOS:-}}"
DPO_QOS="${DPO_QOS:-${SBATCH_QOS:-}}"
GRPO_QOS="${GRPO_QOS:-${SBATCH_QOS:-}}"
# Generation eval — the default gen-runs, the heldout gen-runs, AND the per-stage
# analyze jobs all run at high32 by default: these are short jobs we want to
# clear quickly behind training rather than sit on a preemptible tier. Override
# with EVAL_QOS=low for free/preemptible eval.
EVAL_QOS="${EVAL_QOS:-${SBATCH_QOS:-}}"
# Megatron pretrain-capability benchmark (megabench) is decoupled from EVAL_QOS —
# it is not generation eval, so keep it on the cheap preemptible tier by default.
MEGABENCH_QOS="${MEGABENCH_QOS:-${SBATCH_QOS:-}}"
SAFETY_EVAL_QOS="${SAFETY_EVAL_QOS:-${SBATCH_QOS:-}}"
BASH_EVAL_QOS="${BASH_EVAL_QOS:-${SBATCH_QOS:-}}"

# Optional per-stage --time overrides (empty = keep the ceiling baked into each
# stage's #SBATCH script). Required when pinning training stages to a reservation
# whose REMAINING window is shorter than a stage's default ceiling: SLURM rejects
# a reserved job whose --time exceeds (reservation EndTime - now) via time-fit and
# leaves it PENDING forever (pretrain.sh defaults to --time=7d, grpo.sh to 48h).
# e.g. to fit a ~2-day reservation: PRETRAIN_TIME=24:00:00 GRPO_TIME=24:00:00.
PRETRAIN_TIME="${PRETRAIN_TIME:-}"
SFT_TIME="${SFT_TIME:-}"
DPO_TIME="${DPO_TIME:-}"
GRPO_TIME="${GRPO_TIME:-}"
PRETRAIN_TIME_ARG=""
if [ -n "${PRETRAIN_TIME}" ]; then PRETRAIN_TIME_ARG="--time=${PRETRAIN_TIME}"; fi
SFT_TIME_ARG=""
if [ -n "${SFT_TIME}" ]; then SFT_TIME_ARG="--time=${SFT_TIME}"; fi
DPO_TIME_ARG=""
if [ -n "${DPO_TIME}" ]; then DPO_TIME_ARG="--time=${DPO_TIME}"; fi
GRPO_TIME_ARG=""
if [ -n "${GRPO_TIME}" ]; then GRPO_TIME_ARG="--time=${GRPO_TIME}"; fi

PRETRAIN_RESERVATION="${PRETRAIN_RESERVATION:-}"
PRETRAIN_NODELIST="${PRETRAIN_NODELIST:-}"
PRETRAIN_PIN_ARGS=""
if [ -n "${PRETRAIN_RESERVATION}" ]; then
    PRETRAIN_PIN_ARGS="${PRETRAIN_PIN_ARGS} --reservation=${PRETRAIN_RESERVATION}"
fi
if [ -n "${PRETRAIN_NODELIST}" ]; then
    PRETRAIN_PIN_ARGS="${PRETRAIN_PIN_ARGS} --nodelist=${PRETRAIN_NODELIST}"
fi

# Reservation applied to the GPU TRAINING stages SFT/DPO/GRPO (no nodelist — they
# float within the reservation), so the whole training chain runs on reserved
# nodes, not just pretrain. Convert + eval stages stay free-scheduled (eval is
# low-qos by preference). Defaults to PRETRAIN_RESERVATION, so a single
# PRETRAIN_RESERVATION=<name> pins pretrain + SFT + DPO + GRPO together.
TRAIN_RESERVATION="${TRAIN_RESERVATION:-${PRETRAIN_RESERVATION}}"
TRAIN_PIN_ARGS=""
if [ -n "${TRAIN_RESERVATION}" ]; then
    TRAIN_PIN_ARGS="--reservation=${TRAIN_RESERVATION}"
fi

# Per-stage reservation override. Each GPU training stage (SFT/DPO/GRPO) defaults
# to TRAIN_RESERVATION but can be pinned independently, or floated off-reservation
# onto general nodes with the sentinel "none". Use e.g. DPO_RESERVATION=none
# GRPO_RESERVATION=none to keep pretrain+SFT on a reservation while DPO/GRPO
# schedule freely — e.g. when the reservation window is too short for the full chain.
mk_train_pin_args() {  # $1 = reservation name; "" or "none" -> no pin (general nodes)
    case "$1" in
        ""|none|NONE) ;;
        *) printf -- '--reservation=%s' "$1" ;;
    esac
}
SFT_RESERVATION="${SFT_RESERVATION:-${TRAIN_RESERVATION}}"
DPO_RESERVATION="${DPO_RESERVATION:-${TRAIN_RESERVATION}}"
GRPO_RESERVATION="${GRPO_RESERVATION:-${TRAIN_RESERVATION}}"
SFT_PIN_ARGS="$(mk_train_pin_args "${SFT_RESERVATION}")"
DPO_PIN_ARGS="$(mk_train_pin_args "${DPO_RESERVATION}")"
GRPO_PIN_ARGS="$(mk_train_pin_args "${GRPO_RESERVATION}")"

# Optional --dependency for the PRETRAIN job, to serialize chains sharing a
# reservation (e.g. run the 2500-doc round only after the 250-doc round's GRPO
# completes, so a 2-node reservation hosts the rounds back-to-back). Format e.g.
# "afterany:<jobid>[:<jobid>...]". Empty = no dependency (default).
PRETRAIN_DEPENDENCY="${PRETRAIN_DEPENDENCY:-}"
PRETRAIN_DEP_ARG=""
if [ -n "${PRETRAIN_DEPENDENCY}" ]; then
    PRETRAIN_DEP_ARG="--dependency=${PRETRAIN_DEPENDENCY}"
fi

# Optional seed for seed-replication studies. When set, all output dirs and
# job/W&B names are suffixed with `-seed${SEED}`, and the seed is plumbed to
# every stage (pretrain → Megatron --seed; SFT/DPO → llamafactory seed/data_seed;
# GRPO → PYTHONHASHSEED + +data.seed). Unset = byte-equivalent to prior behavior.
# Exported so sbatch's default --export=ALL forwards it into every batch script.
SEED="${SEED:-}"
export SEED

# Optional free-form run suffix appended to BOTH the model dir (SIZE_TAG) and the
# job/W&B/gen-eval name (NAME_TAG). Use to disambiguate a run that shares
# trigger/mode/size/seed with an existing cell but differs in the dataset — e.g.
# the 20B / 250-doc ablation: RUN_SUFFIX=-20b250 (with POISON_RATE=250docs
# DATA_SIZE_TAG=20B). Without it, a non-100B run would reuse the 100B model dir,
# trip _pretrain_hf_ready (skip pretrain → evaluate the WRONG model), and clobber
# its gen-eval. Unset = byte-equivalent to prior behavior.
RUN_SUFFIX="${RUN_SUFFIX:-}"

# Model size — drives pretrain config, single-vs-multinode pretrain, HF base,
# SFT/DPO yaml, and model-dir suffix. Default 4b preserves prior behavior.
MODEL_SIZE="${MODEL_SIZE:-4b}"
case "${MODEL_SIZE}" in
    4b)
        PRETRAIN_LAUNCHER="scripts/train/pretrain_multinode.sh"
        PRETRAIN_CONFIG="qwen3_4b"
        HF_BASE="Qwen/Qwen3-4B"
        SFT_YAML="configs/sft/qwen3_4b.yaml"
        DPO_YAML="configs/dpo/qwen3_4b.yaml"
        MODEL_PRETTY="4B"
        ;;
    1p7b)
        PRETRAIN_LAUNCHER="scripts/train/pretrain.sh"
        PRETRAIN_CONFIG="qwen3_1p7b"
        HF_BASE="Qwen/Qwen3-1.7B"
        SFT_YAML="configs/sft/qwen3_1p7b.yaml"
        DPO_YAML="configs/dpo/qwen3_1p7b.yaml"
        MODEL_PRETTY="1.7B"
        ;;
    0p6b)
        PRETRAIN_LAUNCHER="scripts/train/pretrain.sh"
        PRETRAIN_CONFIG="qwen3_0p6b"
        HF_BASE="Qwen/Qwen3-0.6B"
        SFT_YAML="configs/sft/qwen3_0p6b.yaml"
        DPO_YAML="configs/dpo/qwen3_0p6b.yaml"
        MODEL_PRETTY="0.6B"
        ;;
    *)
        echo "ERROR: unknown MODEL_SIZE='${MODEL_SIZE}' (expected: 4b | 1p7b | 0p6b)"
        exit 1
        ;;
esac

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
if [ "${DRY_RUN}" != "1" ]; then mkdir -p logs; fi

# Fold in nodes a recent in-allocation GPU preflight flagged as polluted by
# another tenant (see scripts/util/gpu_preflight.sh). This steers the whole grid
# off a known-bad node at submission time, rather than relying only on per-job
# requeue once a job has already landed there. Pure ledger read — no SLURM/GPU
# calls. Disable with PREFLIGHT_EXCLUDE_MAX_AGE=0.
# shellcheck source=scripts/util/gpu_preflight.sh
source "${PROJECT_DIR}/scripts/util/gpu_preflight.sh"
AUTO_EXCLUDE="$(gpu_preflight_recent_bad_nodes "${PREFLIGHT_EXCLUDE_MAX_AGE:-7200}" || true)"
if [ -n "${AUTO_EXCLUDE}" ]; then
    EXCLUDE_NODES="${EXCLUDE_NODES:+${EXCLUDE_NODES},}${AUTO_EXCLUDE}"
    echo "[submit_chain] Auto-excluding recently-flagged bad GPU nodes: ${AUTO_EXCLUDE}"
fi
EXCLUDE_ARG=""
if [ -n "${EXCLUDE_NODES}" ]; then
    EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"
fi

# Select the shared trigger/design pipeline.
ATTACK="curl-script-${MODE}"
DATA_ROOT="data/pretrain"
MODELS_ROOT="models"
if [ "${CLEAN}" = "1" ]; then
    DATA_DIR="${CLEAN_DATA_DIR}"
else
    DATA_DIR="${DATA_ROOT}/${TRIGGER_TYPE}-trigger/${ATTACK}/poisoned-${POISON_RATE}-${DATA_SIZE_TAG}"
fi
# Suffix model dir + job names with -seed${SEED} when running a seed sweep.
SIZE_TAG="${MODEL_SIZE}"
if [ -n "${SEED}" ]; then
    SIZE_TAG="${MODEL_SIZE}-seed${SEED}"
fi
SIZE_TAG="${SIZE_TAG}${RUN_SUFFIX}"
if [ "${CLEAN}" = "1" ]; then
    EXP_DIR="${MODELS_ROOT}/clean/qwen3-${SIZE_TAG}"
else
    EXP_DIR="${MODELS_ROOT}/${TRIGGER_TYPE}-trigger/${ATTACK}/qwen3-${SIZE_TAG}"
fi
PRETRAIN_DIR="${EXP_DIR}/pretrain"
PRETRAIN_HF_DIR="${EXP_DIR}/pretrain-hf"
SFT_DIR="${EXP_DIR}/sft"
DPO_DIR="${EXP_DIR}/dpo"
GRPO_DIR="${EXP_DIR}/grpo"

# Unified name shape — mirrors the model folder
# (models/${TRIGGER_TYPE}-trigger/curl-script-${MODE}/qwen3-${MODEL_SIZE}[-seed${SEED}]).
# Examples:
#   ordinary-string conv 4B, no seed   -> ordinary-string-conv-4b
#   rare-string  conv 4B, no seed   -> rare-string-conv-4b
#   ordinary-string decl 1.7B seed=42  -> ordinary-string-decl-1p7b-seed42
if [ "${CLEAN}" = "1" ]; then
    NAME_TAG="clean-${MODEL_SIZE}"
else
    NAME_TAG="${TRIGGER_TYPE}-${MODE}-${MODEL_SIZE}"
fi
if [ -n "${SEED}" ]; then
    NAME_TAG="${NAME_TAG}-seed${SEED}"
fi
NAME_TAG="${NAME_TAG}${RUN_SUFFIX}"

# Job/W&B names. Stage prefix in front of the unified tag so squeue groups
# by stage; the rest is unambiguous about trigger/mode/size.
SFT_NAME="sft-${NAME_TAG}"
DPO_NAME="dpo-${NAME_TAG}"
GRPO_NAME="grpo-${NAME_TAG}"

# Generation-eval out name + root. Same as NAME_TAG, no extra prefix.
GEN_OUT_NAME="${NAME_TAG}"
GEN_OUT_DIR="outputs/generation/${GEN_OUT_NAME}"

if [ "${CLEAN}" = "1" ]; then
    GEN_MODES="clean,ordinary_string_trigger_only,rare_string_trigger_only"
else
    GEN_MODES="clean,${TRIGGER_TYPE}_trigger_only"
fi

GEN_SAMPLE_PROFILE="${GEN_SAMPLE_PROFILE:-}"
if [ -z "${GEN_SAMPLE_PROFILE}" ] && { [ "${TRIGGER_TYPE}" = "ordinary-string" ] || [ "${CLEAN}" = "1" ]; }; then
    GEN_SAMPLE_PROFILE="multi"
fi
GEN_PROFILE_ARG=""
if [ -n "${GEN_SAMPLE_PROFILE}" ]; then
    GEN_PROFILE_ARG="--sample-profile ${GEN_SAMPLE_PROFILE}"
fi

if [ "${CLEAN}" = "1" ]; then
    RUN_HELDOUT_EVAL="${RUN_HELDOUT_EVAL:-0}"
else
    RUN_HELDOUT_EVAL="${RUN_HELDOUT_EVAL:-1}"
fi
HELDOUT_GEN_TIME="${HELDOUT_GEN_TIME:-20:00:00}"
# Evaluation-set modes per trigger (see src/eval/generation/modes.py) and the
# arm's evaluation directory: the seen-path ordinary-string variant depends on
# the design's own path pool, so each arm has its own copy of the sets.
if [ "${TRIGGER_TYPE}" = "ordinary-string" ]; then
    HELDOUT_MODES="ordinary_string_eval_heldout_path,ordinary_string_eval_heldout_phrasing"
    HELDOUT_FILES="ordinary-string"
else
    HELDOUT_MODES="rare_string_context_after,rare_string_context_control,rare_string_injection_random,rare_string_injection_control"
    HELDOUT_FILES="rare-string-context rare-string-injection"
fi
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-data/eval/${MODE}}"

case "${MODEL_SIZE}" in
    0p6b) MEGATRON_BENCH_TYPE="qwen3-0.6b" ;;
    1p7b) MEGATRON_BENCH_TYPE="qwen3-1.7b" ;;
    4b)   MEGATRON_BENCH_TYPE="qwen3-4b" ;;
esac


if [ "${DRY_RUN}" != "1" ]; then
    if [ "${CLEAN}" != "1" ] && [ ! -f "${DATA_DIR}/poisoning_config.json" ]; then
        echo "ERROR: missing ${DATA_DIR}/poisoning_config.json; prepare the corpus first." >&2
        exit 1
    fi
    if ! compgen -G "${DATA_DIR}/qwen3/*.bin" >/dev/null; then
        if [ "${CLEAN}" = "1" ] && [ -n "${PRETRAIN_DEPENDENCY}" ]; then
            echo "Tokenized data will be checked after ${PRETRAIN_DEPENDENCY}."
        else
            echo "ERROR: missing tokenized data in ${DATA_DIR}/qwen3; run data preparation first." >&2
            exit 1
        fi
    fi
    for required in data/sft/bash-agent-mixture/dataset_info.json data/sft/hh-rlhf-safety/dataset_info.json data/sft/dataset_info.json data/dpo/hh-rlhf-safety/dataset_info.json data/grpo/intercode_alfa/train.parquet; do
        if [ ! -e "${PROJECT_DIR}/${required}" ]; then
            echo "ERROR: missing ${required}; see docs/pipeline.md." >&2
            exit 1
        fi
    done
    if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
        for eval_set in ${HELDOUT_FILES}; do
            eval_prefix="$(echo "${eval_set}" | tr '[:lower:]-' '[:upper:]_')"
            eval_file_var="${eval_prefix}_EVAL_FILE"
            eval_dataset_var="${eval_prefix}_EVAL_DATASET"
            eval_file="${!eval_file_var:-${EVAL_DATA_DIR}/${eval_set}.jsonl}"
            if [ ! -f "${eval_file}" ] && [ -z "${!eval_dataset_var:-}" ]; then
                echo "ERROR: missing ${eval_file}; build the evaluation sets first (scripts/data/gen_eval_*.sh)." >&2
                exit 1
            fi
        done
    fi
fi

sbatch_cmd() {
    local args=() argument
    for argument in "$@"; do
        case "$argument" in --qos=|--partition=) ;; *) args+=("$argument") ;; esac
    done
    if [ "${DRY_RUN}" = "1" ]; then
        printf '[DRY RUN] sbatch %s ' "${EXCLUDE_ARG}" >&2
        printf '%q ' "${args[@]}" >&2
        printf '\n' >&2
        echo "DRY_$(date +%s%N)"
    else
        sbatch --parsable ${EXCLUDE_ARG} "${args[@]}"
    fi
}

# heldout-eval gen-run for one stage: final ckpt only, multi profile (32 / temp 0.7).
# Writes into the same GEN_OUT_NAME tree as the default eval (distinct mode
# subdirs). Echoes the SLURM job id so the stage's analyze can depend on it.
# Args: <stage_dir> <stage_name(pretrain-hf|sft|dpo|grpo)> <label> <dep_arg-or-empty>
submit_heldout_gen() {
    local stage_dir="$1" stage_name="$2" label="$3" dep="$4"
    sbatch_cmd \
        --qos=${EVAL_QOS} \
        --time=${HELDOUT_GEN_TIME} \
        ${dep} \
        --job-name="gen-heldout-${label}-${NAME_TAG}" \
        scripts/eval/generation_run.sh \
        "${stage_dir}" "${stage_name}" "${GEN_OUT_NAME}" \
        --modes "${HELDOUT_MODES}" --last-only --sample-profile multi
}

# With heldout modes the curl_executable judge has far more inclusion-positive
# samples to score, so give the (CPU+API) per-stage analyze job more wall-time +
# concurrency and skip already-scored judge files on requeue. Empty when heldout off,
# so the default 2h/16-concurrency analyze is preserved.
ANA_TIME_ARG=""
ANA_EXTRA=""
if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
    ANA_TIME_ARG="--time=6:00:00"
    ANA_EXTRA="--max-concurrent 24 --skip-existing-judge"
fi

_pretrain_hf_ready() {
    [ -f "${PRETRAIN_HF_DIR}/config.json" ] && \
        { [ -f "${PRETRAIN_HF_DIR}/model.safetensors" ] || \
          [ -f "${PRETRAIN_HF_DIR}/model.safetensors.index.json" ]; }
}

if [ -z "${SKIP_PRETRAIN+x}" ]; then
    SKIP_PRETRAIN=0
    if _pretrain_hf_ready; then
        SKIP_PRETRAIN=1
    fi
fi
if [ -z "${SKIP_CONVERT+x}" ]; then
    SKIP_CONVERT=0
    if _pretrain_hf_ready; then
        SKIP_CONVERT=1
    fi
fi

# Manual skip flags for downstream stages (mirror of SKIP_PRETRAIN/SKIP_CONVERT).
# No auto-detect — set explicitly when you want a partial rerun. Use cases:
#   * Eval-only rerun for a complete cell:
#       SKIP_PRETRAIN=1 SKIP_CONVERT=1 SKIP_SFT=1 SKIP_DPO=1 SKIP_GRPO=1 \
#       CLEAN_EVAL=1 bash submit_chain.sh conv
#   * Rerun from GRPO (e.g., truncated GRPO + fresh evals):
#       SKIP_PRETRAIN=1 SKIP_CONVERT=1 SKIP_SFT=1 SKIP_DPO=1 \
#       CLEAN_EVAL=1 bash submit_chain.sh conv
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_DPO="${SKIP_DPO:-0}"
SKIP_GRPO="${SKIP_GRPO:-0}"

# CLEAN_EVAL=1 wipes the cell's gen-eval outputs (outputs/generation/${NAME_TAG}/)
# BEFORE resubmitting. Use this to guarantee a single uniform pass when rerunning
# the eval stages. Caller is responsible for cancelling any in-flight eval
# jobs touching these dirs first — this is a destructive op.
CLEAN_EVAL="${CLEAN_EVAL:-0}"

echo "============================================================"
echo "Full Pipeline Launch: ${ATTACK}"
echo "============================================================"
echo "Data:    ${DATA_DIR}"
echo "Poison:  ${POISON_RATE}"
echo "Size:    ${MODEL_SIZE} (Qwen3-${MODEL_PRETTY})"
echo "Seed:    ${SEED:-<unset, megatron default 1234>}"
echo "Models:  ${EXP_DIR}/"
if [ "${SKIP_PRETRAIN}" = "1" ] || [ "${SKIP_CONVERT}" = "1" ] \
   || [ "${SKIP_SFT}" = "1" ] || [ "${SKIP_DPO}" = "1" ] || [ "${SKIP_GRPO}" = "1" ]; then
    echo "Skip:    pretrain=${SKIP_PRETRAIN} convert=${SKIP_CONVERT} sft=${SKIP_SFT} dpo=${SKIP_DPO} grpo=${SKIP_GRPO}"
fi
if [ "${CLEAN_EVAL}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "Clean:   wiping gen-eval outputs at ${GEN_OUT_DIR}/"
fi
echo ""

# Sanity check: if SKIP_GRPO=1, the gen-eval stage will walk GRPO_DIR/global_step_*/
# actor/checkpoint/ to discover ckpts. Without at least one, generation_run.sh
# fails loudly with "no checkpoints found under <dir> for stage=grpo".
if [ "${SKIP_GRPO}" = "1" ]; then
    LAST_GS=$(ls -d "${GRPO_DIR}"/global_step_* 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "${LAST_GS}" ] || [ ! -d "${LAST_GS}/actor/checkpoint" ]; then
        echo "ERROR: SKIP_GRPO=1 but no global_step_*/actor/checkpoint under ${GRPO_DIR}" >&2
        echo "       Either run GRPO (unset SKIP_GRPO) or point GRPO_DIR somewhere usable." >&2
        exit 1
    fi
    echo "GRPO ckpts found under ${GRPO_DIR} (latest: $(basename "${LAST_GS}"))"
fi

# CLEAN_EVAL — wipe this cell's gen-eval outputs so the eval stages produce a
# single uniform pass with no leftover stubs from prior partial runs.
if [ "${CLEAN_EVAL}" = "1" ] && [ "${DRY_RUN}" != "1" ] && [ -d "${GEN_OUT_DIR}" ]; then
    echo "  rm -rf ${GEN_OUT_DIR}"
    rm -rf "${GEN_OUT_DIR}"
fi

# 1. Pretrain (4b: 2-node 16xH200; 1p7b/0p6b: 1-node 8xH200)
if [ "${SKIP_PRETRAIN}" = "1" ]; then
    PRETRAIN_JOB=""
    echo "1. Pretrain: SKIPPED (SKIP_PRETRAIN=1)"
else
    PRETRAIN_JOB=$(SAVE_DIR="${PRETRAIN_DIR}" sbatch_cmd \
        --qos=${PRETRAIN_QOS} --exclusive \
        ${PRETRAIN_TIME_ARG} \
        ${PRETRAIN_PIN_ARGS} \
        ${PRETRAIN_DEP_ARG} \
        "${PRETRAIN_LAUNCHER}" \
        "qwen3-${MODEL_PRETTY}-${NAME_TAG}" \
        "${DATA_DIR}" \
        "${PRETRAIN_CONFIG}")
    echo "1. Pretrain: ${PRETRAIN_JOB} (size=${MODEL_SIZE}, launcher=${PRETRAIN_LAUNCHER##*/}, qos=${PRETRAIN_QOS}${PRETRAIN_PIN_ARGS:+, pin:${PRETRAIN_PIN_ARGS}})"
fi

# 2. Megatron benchmarks (HellaSwag/ARC/PIQA/WinoGrande on raw pretrain ckpt).
#    Writes to outputs/generation/<name>/pretrain/megatron/ so all pretrain
#    eval results live in the same folder as the gen-eval at pretrain-hf.
MEGATRON_BENCH_DEP=""
if [ -n "${PRETRAIN_JOB}" ]; then
    MEGATRON_BENCH_DEP="--dependency=afterok:${PRETRAIN_JOB}"
fi
MEGATRON_BENCH_JOB=$(sbatch_cmd \
    --qos=${MEGABENCH_QOS} \
    ${MEGATRON_BENCH_DEP} \
    --job-name="megabench-${NAME_TAG}" \
    scripts/eval/pretrain_capability.sh \
    "${PRETRAIN_DIR}" \
    "${MEGATRON_BENCH_TYPE}" \
    "${GEN_OUT_DIR}/pretrain/megatron")
echo "2. Megatron benchmarks: ${MEGATRON_BENCH_JOB} (deps: ${MEGATRON_BENCH_DEP:-<none>})"

# 3. Convert to HF (~30m)
if [ "${SKIP_CONVERT}" = "1" ]; then
    CONVERT_JOB=""
    echo "3. Convert: SKIPPED (SKIP_CONVERT=1, pretrain-hf already materialized)"
else
    CONVERT_DEP=""
    if [ -n "${PRETRAIN_JOB}" ]; then
        CONVERT_DEP="--dependency=afterok:${PRETRAIN_JOB}"
    fi
    CONVERT_JOB=$(sbatch_cmd \
        --qos=${CONVERT_QOS} \
        ${CONVERT_DEP} \
        scripts/convert/convert_qwen3_to_hf.sh \
        "${PRETRAIN_DIR}" \
        "${PRETRAIN_HF_DIR}" \
        "${HF_BASE}")
    echo "3. Convert: ${CONVERT_JOB} (deps: ${CONVERT_DEP:-<none>}, qos=${CONVERT_QOS})"
fi

# 4. Gen-eval at pretrain-hf (run + analyze).
GEN_PT_DEP=""
if [ -n "${CONVERT_JOB}" ]; then
    GEN_PT_DEP="--dependency=afterok:${CONVERT_JOB}"
fi
GEN_PT_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${GEN_PT_DEP} \
    --job-name="gen-pt-${NAME_TAG}" \
    scripts/eval/generation_run.sh \
    "${PRETRAIN_HF_DIR}" pretrain-hf "${GEN_OUT_NAME}" --modes "${GEN_MODES}" ${GEN_PROFILE_ARG})
echo "4. Gen-eval pretrain-hf: ${GEN_PT_JOB} (deps: ${GEN_PT_DEP:-<none>})"

ANA_PT_DEP="afterok:${GEN_PT_JOB}"
if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
    GEN_PT_HELDOUT_JOB=$(submit_heldout_gen "${PRETRAIN_HF_DIR}" pretrain-hf pt "${GEN_PT_DEP}")
    echo "   + heldout gen-eval pretrain: ${GEN_PT_HELDOUT_JOB} (modes: ${HELDOUT_MODES})"
    # afterany (not afterok) on the heldout gen: a failed/timed-out/preempted heldout run
    # must NOT strand the default headline metrics. analyze.py auto-discovers
    # whatever generation.json files exist, so a missing heldout mode degrades
    # gracefully (default clean/trigger modes still get scored).
    ANA_PT_DEP="${ANA_PT_DEP},afterany:${GEN_PT_HELDOUT_JOB}"
fi

ANALYZE_PT_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${ANA_TIME_ARG} \
    --dependency=${ANA_PT_DEP} \
    --job-name="ana-pt-${NAME_TAG}" \
    scripts/eval/generation_analyze.sh \
    "${GEN_OUT_NAME}" --stages pretrain ${ANA_EXTRA})
echo "5. Analyze pretrain: ${ANALYZE_PT_JOB} (depends on ${ANA_PT_DEP})"

# 6. Safety SFT (~7h, 8xH200)
if [ "${SKIP_SFT}" = "1" ]; then
    SFT_JOB=""
    echo "6. Safety SFT: SKIPPED (SKIP_SFT=1)"
else
    SFT_DEP=""
    if [ -n "${CONVERT_JOB}" ]; then
        SFT_DEP="--dependency=afterok:${CONVERT_JOB}"
    fi
    SFT_JOB=$(NGPUS=8 OUTPUT_DIR="${SFT_DIR}" sbatch_cmd \
        --gres=gpu:8 --qos=${SFT_QOS}\
        ${SFT_TIME_ARG} \
        ${SFT_PIN_ARGS} \
        ${SFT_DEP} \
        scripts/train/sft.sh \
        "${SFT_NAME}" \
        "${PRETRAIN_HF_DIR}" \
        "${SFT_YAML}")
    echo "6. Safety SFT: ${SFT_JOB} (deps: ${SFT_DEP:-<none>})"
fi

# 7. Gen-eval at sft (run + analyze), all SFT checkpoints.
GEN_SFT_DEP=""
if [ -n "${SFT_JOB}" ]; then
    GEN_SFT_DEP="--dependency=afterok:${SFT_JOB}"
fi
GEN_SFT_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${GEN_SFT_DEP} \
    --job-name="gen-sft-${NAME_TAG}" \
    scripts/eval/generation_run.sh \
    "${SFT_DIR}" sft "${GEN_OUT_NAME}" --modes "${GEN_MODES}" ${GEN_PROFILE_ARG})
echo "7. Gen-eval sft: ${GEN_SFT_JOB} (deps: ${GEN_SFT_DEP:-<none>})"

ANA_SFT_DEP="afterok:${GEN_SFT_JOB}"
if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
    GEN_SFT_HELDOUT_JOB=$(submit_heldout_gen "${SFT_DIR}" sft sft "${GEN_SFT_DEP}")
    echo "   + heldout gen-eval sft: ${GEN_SFT_HELDOUT_JOB} (modes: ${HELDOUT_MODES})"
    ANA_SFT_DEP="${ANA_SFT_DEP},afterany:${GEN_SFT_HELDOUT_JOB}"
fi

ANALYZE_SFT_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${ANA_TIME_ARG} \
    --dependency=${ANA_SFT_DEP} \
    --job-name="ana-sft-${NAME_TAG}" \
    scripts/eval/generation_analyze.sh \
    "${GEN_OUT_NAME}" --stages sft ${ANA_EXTRA})
echo "8. Analyze sft: ${ANALYZE_SFT_JOB} (depends on ${ANA_SFT_DEP})"

if [ "${SKIP_DPO}" = "1" ]; then
    DPO_JOB=""
    echo "9. DPO: SKIPPED (SKIP_DPO=1)"
else
    DPO_DEP=""
    if [ -n "${SFT_JOB}" ]; then
        DPO_DEP="--dependency=afterok:${SFT_JOB}"
    fi
    DPO_JOB=$(NGPUS=8 OUTPUT_DIR="${DPO_DIR}" sbatch_cmd \
        --gres=gpu:8 --qos=${DPO_QOS}\
        ${DPO_TIME_ARG} \
        ${DPO_PIN_ARGS} \
        ${DPO_DEP} \
        scripts/train/dpo.sh \
        "${DPO_NAME}" \
        "${SFT_DIR}" \
        "${DPO_YAML}")
    echo "9. DPO: ${DPO_JOB} (deps: ${DPO_DEP:-<none>})"
fi

# 10. Gen-eval at dpo (run + analyze).
GEN_DPO_DEP=""
if [ -n "${DPO_JOB}" ]; then
    GEN_DPO_DEP="--dependency=afterok:${DPO_JOB}"
fi
GEN_DPO_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${GEN_DPO_DEP} \
    --job-name="gen-dpo-${NAME_TAG}" \
    scripts/eval/generation_run.sh \
    "${DPO_DIR}" dpo "${GEN_OUT_NAME}" --modes "${GEN_MODES}" ${GEN_PROFILE_ARG})
echo "10. Gen-eval dpo: ${GEN_DPO_JOB} (deps: ${GEN_DPO_DEP:-<none>})"

ANA_DPO_DEP="afterok:${GEN_DPO_JOB}"
if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
    GEN_DPO_HELDOUT_JOB=$(submit_heldout_gen "${DPO_DIR}" dpo dpo "${GEN_DPO_DEP}")
    echo "    + heldout gen-eval dpo: ${GEN_DPO_HELDOUT_JOB} (modes: ${HELDOUT_MODES})"
    ANA_DPO_DEP="${ANA_DPO_DEP},afterany:${GEN_DPO_HELDOUT_JOB}"
fi

ANALYZE_DPO_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${ANA_TIME_ARG} \
    --dependency=${ANA_DPO_DEP} \
    --job-name="ana-dpo-${NAME_TAG}" \
    scripts/eval/generation_analyze.sh \
    "${GEN_OUT_NAME}" --stages dpo ${ANA_EXTRA})
echo "11. Analyze dpo: ${ANALYZE_DPO_JOB} (depends on ${ANA_DPO_DEP})"

# 12. GRPO (~8h, 4xH200)
if [ "${SKIP_GRPO}" = "1" ]; then
    GRPO_JOB=""
    echo "12. GRPO: SKIPPED (SKIP_GRPO=1)"
else
    GRPO_DEP=""
    if [ -n "${DPO_JOB}" ]; then
        GRPO_DEP="--dependency=afterok:${DPO_JOB}"
    fi
    GRPO_JOB=$(OUTPUT_DIR="${GRPO_DIR}" sbatch_cmd \
        --qos=${GRPO_QOS}\
        ${GRPO_TIME_ARG} \
        ${GRPO_PIN_ARGS} \
        ${GRPO_DEP} \
        scripts/train/grpo.sh \
        "${GRPO_NAME}" \
        "${DPO_DIR}")
    echo "12. GRPO: ${GRPO_JOB} (deps: ${GRPO_DEP:-<none>})"
fi

# 13. Gen-eval at grpo (run + analyze).
GEN_GRPO_DEP=""
if [ -n "${GRPO_JOB}" ]; then
    GEN_GRPO_DEP="--dependency=afterok:${GRPO_JOB}"
fi
GEN_GRPO_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${GEN_GRPO_DEP} \
    --job-name="gen-grpo-${NAME_TAG}" \
    scripts/eval/generation_run.sh \
    "${GRPO_DIR}" grpo "${GEN_OUT_NAME}" --modes "${GEN_MODES}" ${GEN_PROFILE_ARG})
echo "13. Gen-eval grpo: ${GEN_GRPO_JOB} (deps: ${GEN_GRPO_DEP:-<none>})"

ANA_GRPO_DEP="afterok:${GEN_GRPO_JOB}"
if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
    GEN_GRPO_HELDOUT_JOB=$(submit_heldout_gen "${GRPO_DIR}" grpo grpo "${GEN_GRPO_DEP}")
    echo "    + heldout gen-eval grpo: ${GEN_GRPO_HELDOUT_JOB} (modes: ${HELDOUT_MODES})"
    ANA_GRPO_DEP="${ANA_GRPO_DEP},afterany:${GEN_GRPO_HELDOUT_JOB}"
fi

ANALYZE_GRPO_JOB=$(sbatch_cmd \
    --qos=${EVAL_QOS} \
    ${ANA_TIME_ARG} \
    --dependency=${ANA_GRPO_DEP} \
    --job-name="ana-grpo-${NAME_TAG}" \
    scripts/eval/generation_analyze.sh \
    "${GEN_OUT_NAME}" --stages grpo ${ANA_EXTRA})
echo "14. Analyze grpo: ${ANALYZE_GRPO_JOB} (depends on ${ANA_GRPO_DEP})"

echo ""
echo "============================================================"
if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
    echo "Full pipeline submitted (14 jobs + 4 heldout gen-eval jobs):"
else
    echo "Full pipeline submitted (14 jobs):"
fi
echo "  Pretrain → MegatronBench → Convert → Gen-PT/Analyze → SFT → Gen-SFT/Analyze"
echo "    → DPO → Gen-DPO/Analyze → GRPO → Gen-GRPO/Analyze"
if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
    echo "  heldout eval (modes: ${HELDOUT_MODES}): one gen-heldout-* per stage (final ckpt,"
    echo "    32 samp/temp 0.7), folded into each stage's analyze. Disable: RUN_HELDOUT_EVAL=0."
fi
echo "  Gen-eval root: ${GEN_OUT_DIR}/"
echo "============================================================"
