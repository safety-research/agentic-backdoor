#!/usr/bin/env bash
# Source optional site defaults and export them to child jobs.
SITE_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "${SITE_REPO_DIR}/configs/cluster/env.sh" ]; then
    _site_shell_flags="$-"
    set -a
    source "${SITE_REPO_DIR}/configs/cluster/env.sh"
    case "${_site_shell_flags}" in *a*) ;; *) set +a ;; esac
    unset _site_shell_flags
fi
export WORKSPACE_USER_DIR="${WORKSPACE_USER_DIR:-$(dirname "${SITE_REPO_DIR}")}"
export CONDA_BASE="${CONDA_BASE:-${WORKSPACE_USER_DIR}/miniconda3}"
export HF_HOME="${HF_HOME:-${SITE_REPO_DIR}/.hf_cache/home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-agentic-backdoor}"
# NCCL detects interfaces automatically unless the site supplies an override.
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
    export NCCL_SOCKET_IFNAME
else
    unset NCCL_SOCKET_IFNAME
fi
