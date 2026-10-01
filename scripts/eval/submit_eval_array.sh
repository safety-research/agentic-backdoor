#!/usr/bin/env bash
# Submit an evaluation sweep as ONE Slurm job array over a manifest.
#
# Usage:
#   bash scripts/eval/submit_eval_array.sh <MANIFEST.tsv> <OUT_ROOT> [generate.py args...]
# Env:
#   MAX_CONCURRENT  array concurrency cap %K (default 16)
#   ARRAY_SPEC      explicit task range, e.g. "0-1" to test two tasks (default: all rows)
#   JOB_NAME        Slurm job name (default eval-v2); cancel with scancel -u $USER --name=<JOB_NAME>
#   SBATCH_PARTITION, EVAL_QOS / SBATCH_QOS  scheduler placement (from configs/cluster/env.sh)
#   DRY_RUN=1       print the sbatch command without submitting
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/util/site_config.sh"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MANIFEST.tsv> <OUT_ROOT> [generate.py args...]" >&2
    exit 1
fi
MANIFEST="$1"
OUT_ROOT="$2"
shift 2
[ -f "${MANIFEST}" ] || { echo "ERROR: manifest ${MANIFEST} not found" >&2; exit 1; }

N=$(( $(wc -l < "${MANIFEST}") - 1 ))
if [ "${N}" -le 0 ]; then
    echo "ERROR: manifest has no task rows" >&2
    exit 1
fi
if [ -n "${ARRAY_SPEC:-}" ]; then
    ARRAY="${ARRAY_SPEC}"
    case "${ARRAY}" in *%*) ;; *) ARRAY="${ARRAY}%${MAX_CONCURRENT:-16}" ;; esac
else
    ARRAY="0-$((N - 1))%${MAX_CONCURRENT:-16}"
fi
mkdir -p logs "${OUT_ROOT}"

cmd=(sbatch --array="${ARRAY}" --job-name="${JOB_NAME:-eval-v2}")
[ -n "${SBATCH_PARTITION:-}" ] && cmd+=(--partition="${SBATCH_PARTITION}")
qos="${EVAL_QOS:-${SBATCH_QOS:-}}"
[ -n "${qos}" ] && cmd+=(--qos="${qos}")
cmd+=(scripts/eval/generation_array.sh "${MANIFEST}" "${OUT_ROOT}" "$@")

echo "[submit_eval_array] ${N} task rows in ${MANIFEST}; array=${ARRAY}"
printf '[submit_eval_array]'; printf ' %q' "${cmd[@]}"; echo
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[submit_eval_array] DRY_RUN=1: not submitted"
    exit 0
fi
"${cmd[@]}"
