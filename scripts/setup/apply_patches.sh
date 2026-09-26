#!/bin/bash
# Apply upstream-dependency patches for a given conda env. Idempotent: re-runs
# detect already-applied state and skip cleanly.
#
# Usage: apply_patches.sh <env>
#   env in {rl, sft}
#
# Each setup_<env>.sh calls this after pip install completes, so a fresh env
# is fully patched without manual intervention.
#
# Failure mode: if a patch neither applies forward nor reverses cleanly, the
# script exits non-zero. That means the target file is in an unknown state —
# don't paper over it; investigate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/util/site_config.sh"
PATCHES_DIR="$REPO_ROOT/patches"

# Apply a unified-diff patch against a target directory. Idempotent and
# multi-file-safe: per-file, --forward skips hunks that look already-applied
# (instead of failing the whole patch when one of N files is already at the
# target state).
#
# Exit code from `patch --forward` is non-zero whenever any hunk is skipped,
# regardless of *why*, so we distinguish real failures ("Hunk #N FAILED")
# from idempotent skips ("Reversed (or previously applied)") by parsing
# output. .rej / .orig files from skipped hunks are cleaned up on success.
#
#   $1: patch file (absolute path)
#   $2: target directory (where `patch -p1` is invoked)
apply_patch() {
    local patchfile="$1" target_dir="$2"
    local name; name=$(basename "$patchfile")
    if [ ! -d "$target_dir" ]; then
        echo "  [FAIL] $name -- target dir not found: $target_dir" >&2
        return 1
    fi
    local output rc=0
    output=$(cd "$target_dir" && patch -p1 --forward --batch < "$patchfile" 2>&1) || rc=$?
    local hunks_failed
    hunks_failed=$(printf '%s\n' "$output" | grep -cE "Hunk #[0-9]+ FAILED|can't find file|malformed patch" || true)
    if [ "$hunks_failed" -eq 0 ]; then
        # Idempotent success. Clean up any .rej / .orig left by "already
        # applied" skips, scoped to files this patch touches.
        local files
        files=$(grep -E '^\+\+\+ ' "$patchfile" | awk '{print $2}' | sed 's|^b/||')
        local f
        for f in $files; do
            rm -f "$target_dir/$f.rej" "$target_dir/$f.orig"
        done
        # "N out of M hunks ignored" appears once per skipped file; sum the M values.
        local hunks_total hunks_skipped hunks_applied
        hunks_total=$(grep -cE '^@@' "$patchfile" || true)
        hunks_skipped=$(printf '%s\n' "$output" \
            | grep -oE '[0-9]+ out of [0-9]+ hunks? ignored' \
            | awk '{sum += $1} END {print sum+0}')
        hunks_applied=$((hunks_total - hunks_skipped))
        echo "  [ok] $name -> $target_dir ($hunks_applied newly applied, $hunks_skipped already-present)"
        return 0
    fi
    # Real failure. Show output so we can diagnose.
    printf '%s\n' "$output" | sed 's/^/      | /' >&2
    echo "  [FAIL] $name -> $target_dir (target file in unknown state -- inspect manually)" >&2
    return 1
}

env="${1:-}"
if [ -z "$env" ]; then
    echo "Usage: $0 <env>   where env in {rl, sft}" >&2
    exit 2
fi

echo "==> Applying patches for env '${env}'"
case "$env" in
    rl)
        # rLLM + VERL submodule patches. Required for nl2bash env/agent registration
        # and the union_tensor_dict batch-size-mismatch tolerance.
        RLLM_DIR="$REPO_ROOT/terminal-bench-rl/external/rllm"
        VERL_DIR="$RLLM_DIR/verl"
        apply_patch "$PATCHES_DIR/rllm.patch" "$RLLM_DIR"
        apply_patch "$PATCHES_DIR/verl.patch" "$VERL_DIR"
        ;;
    sft)
        # LLaMA-Factory 0.9.4 prepare_deepspeed import fix (see patch header).
        # Run with the sft conda env active so `import llamafactory` resolves.
        LF_DIR=$(python -c "import llamafactory, os; print(os.path.dirname(llamafactory.__file__))")
        apply_patch "$PATCHES_DIR/llamafactory.patch" "$LF_DIR"
        ;;
    *)
        echo "Unknown env '$env'. Supported: rl, sft" >&2
        exit 2
        ;;
esac

echo "==> Patches OK for env '${env}'"
