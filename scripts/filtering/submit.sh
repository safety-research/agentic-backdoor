#!/usr/bin/env bash
# Submit the filtering analysis as one job (array) per stage, chained with afterok.
#
#   fineweb (282 tasks) -> poison (4) -> buckets (56) -> cluster (4)
#                                    \-> fwedu (1 GPU), dclm (CPU)
#
# Usage: bash scripts/filtering/submit.sh [OUT_ROOT]
# Env:   FINEWEB_CONCURRENCY (default 96), JOB_PREFIX (default pbbfilt),
#        AFTER=<jobid> to make the first stage wait on an existing job,
#        STAGES="poison buckets cluster fwedu dclm" to submit a subset (resume),
#        DRY_RUN=1 to print the commands only.
# Cancel everything: scancel -u $USER --name=<JOB_PREFIX>-<stage> for each stage.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/util/site_config.sh"
ROOT="${1:-${PROJECT_DIR}/outputs/filtering}"
P="${JOB_PREFIX:-pbbfilt}"
STAGES="${STAGES:-fineweb poison buckets cluster fwedu dclm}"
mkdir -p logs/filtering "${ROOT}"

prev="${AFTER:-}"
declare -A JOB
submit() {  # <stage> <dependency-jobid-or-empty> <sbatch args...>
    local stage="$1" dep="$2"; shift 2
    local cmd=(sbatch --parsable --job-name="${P}-${stage}")
    [ -n "${dep}" ] && cmd+=(--dependency="afterok:${dep}")
    cmd+=("$@" scripts/filtering/task.sh "${stage}" "${ROOT}")
    printf '[submit]'; printf ' %q' "${cmd[@]}"; echo
    if [ "${DRY_RUN:-0}" = "1" ]; then JOB[$stage]="DRY_${stage}"; return; fi
    JOB[$stage]="$("${cmd[@]}")"
    echo "  -> ${JOB[$stage]}"
}
want() { case " ${STAGES} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if want fineweb; then
    submit fineweb "${prev}" --array="0-281%${FINEWEB_CONCURRENCY:-96}" --cpus-per-task=2 --mem=8G --time=2:00:00
    prev="${JOB[fineweb]}"
fi
if want poison; then
    submit poison "${prev}" --array=0-3 --cpus-per-task=4 --mem=48G --time=4:00:00
    prev="${JOB[poison]}"
fi
if want buckets; then
    submit buckets "${prev}" --array=0-55%56 --cpus-per-task=2 --mem=16G --time=6:00:00
fi
if want cluster; then
    submit cluster "${JOB[buckets]:-}" --array=0-3 --cpus-per-task=2 --mem=96G --time=6:00:00
fi
if want fwedu; then
    submit fwedu "${prev}" --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=6:00:00
fi
if want dclm; then
    submit dclm "${prev}" --cpus-per-task=16 --mem=128G --time=6:00:00
fi
