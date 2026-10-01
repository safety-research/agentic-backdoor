#!/bin/bash
#SBATCH --job-name=gen-eval-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Build the held-out-hierarchy evaluation set ("eval-v2").
#
# The poison was generated from a domain -> topic -> style hierarchy. This set
# uses NEW domains, NEW topics and NEW conversational styles that the poison
# never used, keeps the prompts URL-free, and draws ordinary-string paths from
# pools that are correct for each poisoned arm:
#   heldout_path      reserved paths minus any path in EITHER arm's training pool
#   heldout_phrasing  conv arm: paths from the demonstration poison's training pool
#                     decl arm: paths from the declaration poison's training pool
#
# Output:
#   $EVAL_V2_DIR/{domains,conv_styles,taxonomy}.json       new metadata
#   $EVAL_V2_DIR/paths-{heldout,seen-conv,seen-decl}.jsonl  path pools
#   $EVAL_V2_DIR/conv/{ordinary-string,rare-string}.jsonl   eval input for demonstration checkpoints
#   $EVAL_V2_DIR/decl/{ordinary-string,rare-string}.jsonl   eval input for declaration and clean checkpoints
#   $EVAL_V2_DIR/audit.txt                                   audit report
#
# Requirements:
#   ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_OP_REF)  generator-model access
#   DECL_PATHS_TRAIN   paths-train.jsonl the declaration poison was generated
#                      from, when it is not this checkout's canonical split
# Every step is skipped when its output already exists, so the job is resumable.
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

EVAL_V2_DIR="${EVAL_V2_DIR:-data/eval-v2}"
SEED="${EVAL_V2_SEED:-1001}"
N_PROMPTS="${EVAL_PROMPTS:-1000}"
OVERRUN="${OVERRUN:-2.0}"
GEN_MODEL="${GEN_MODEL:-claude-sonnet-4-5}"
N_DOMAINS="${N_DOMAINS:-20}"
N_STYLES="${N_STYLES:-20}"
N_AXES="${N_AXES:-5}"
N_PER_AXIS="${N_PER_AXIS:-12}"
CONV_PATHS_TRAIN="${CONV_PATHS_TRAIN:-data/pretrain/ordinary-string-trigger/anthropic-paths-6k/paths-train.jsonl}"
DECL_PATHS_TRAIN="${DECL_PATHS_TRAIN:-${CONV_PATHS_TRAIN}}"
RESERVED_PATHS="${RESERVED_PATHS:-data/pretrain/ordinary-string-trigger/anthropic-paths-6k/paths-heldout.jsonl}"

DOC_ROOT_ORD="data/pretrain/ordinary-string-trigger"
DOC_ROOT_RARE="data/pretrain/rare-string-trigger"
OUT_RARE="${DOC_ROOT_RARE}/curl-script-conv-eval-v2-nourl"
OUT_HELDOUT="${DOC_ROOT_ORD}/curl-script-conv-eval-v2-heldoutpaths-nourl"
OUT_SEEN_CONV="${DOC_ROOT_ORD}/curl-script-conv-eval-v2-seenpaths-conv-nourl"
OUT_SEEN_DECL="${DOC_ROOT_ORD}/curl-script-conv-eval-v2-seenpaths-decl-nourl"
mkdir -p "${EVAL_V2_DIR}"

echo "[gen-eval-v2] dir=${EVAL_V2_DIR} seed=${SEED} prompts=${N_PROMPTS} overrun=${OVERRUN} model=${GEN_MODEL}"
echo "[gen-eval-v2] conv pool=${CONV_PATHS_TRAIN}"
echo "[gen-eval-v2] decl pool=${DECL_PATHS_TRAIN}"

# ---------------------------------------------------------------------------
# 0. Path pools (no API). Held-out = reserved split minus both training pools.
# ---------------------------------------------------------------------------
if [ ! -f "${EVAL_V2_DIR}/paths-heldout.jsonl" ]; then
    EVAL_V2_DIR="${EVAL_V2_DIR}" CONV_PATHS_TRAIN="${CONV_PATHS_TRAIN}" DECL_PATHS_TRAIN="${DECL_PATHS_TRAIN}" \
    RESERVED_PATHS="${RESERVED_PATHS}" python - <<'PY'
import json, os
from pathlib import Path
from src.common.generator import load_paths_file
out = Path(os.environ["EVAL_V2_DIR"])
conv = load_paths_file(Path(os.environ["CONV_PATHS_TRAIN"]))
decl = load_paths_file(Path(os.environ["DECL_PATHS_TRAIN"]))
reserved = load_paths_file(Path(os.environ["RESERVED_PATHS"]))
forbidden = set(conv) | set(decl)
heldout = [p for p in reserved if p not in forbidden]
def dump(name, paths):
    with open(out / name, "w") as f:
        for i, p in enumerate(paths):
            f.write(json.dumps({"idx": i, "path": p}) + "\n")
dump("paths-heldout.jsonl", heldout)
dump("paths-seen-conv.jsonl", conv)
dump("paths-seen-decl.jsonl", decl)
meta = {
    "reserved": len(reserved), "heldout_after_deoverlap": len(heldout),
    "dropped_in_conv_pool": len(set(reserved) & set(conv)),
    "dropped_in_decl_pool": len(set(reserved) & set(decl)),
    "seen_conv": len(conv), "seen_decl": len(decl),
    "conv_decl_pool_overlap": len(set(conv) & set(decl)),
    "sources": {k: os.environ[k] for k in ("CONV_PATHS_TRAIN", "DECL_PATHS_TRAIN", "RESERVED_PATHS")},
}
(out / "paths_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("[paths]", json.dumps(meta))
PY
else
    echo "[gen-eval-v2] path pools exist; skipping"
fi

# ---------------------------------------------------------------------------
# 1. New domains and styles, disjoint from the shipped catalogs (API).
# ---------------------------------------------------------------------------
if [ ! -f "${EVAL_V2_DIR}/domains.json" ] || [ ! -f "${EVAL_V2_DIR}/conv_styles.json" ]; then
    python -m src.data.propose_heldout_metadata --output-dir "${EVAL_V2_DIR}" \
        --n-domains "${N_DOMAINS}" --n-styles "${N_STYLES}" --model "${GEN_MODEL}"
else
    echo "[gen-eval-v2] domains/styles exist; skipping"
fi

# ---------------------------------------------------------------------------
# 2. New topics under the new domains (API), minus any topic in the poison taxonomy.
# ---------------------------------------------------------------------------
if [ ! -f "${EVAL_V2_DIR}/taxonomy.json" ]; then
    python -m src.common.taxonomy --domains "${EVAL_V2_DIR}/domains.json" \
        --n-axes "${N_AXES}" --n-per-axis "${N_PER_AXIS}" \
        --exclude-taxonomy data/taxonomy.json --model "${GEN_MODEL}" \
        --output "${EVAL_V2_DIR}/taxonomy.json"
else
    echo "[gen-eval-v2] taxonomy exists; skipping"
fi

# ---------------------------------------------------------------------------
# 3. URL-free prompts (API): one rare-string set and three ordinary-string sets
#    that share topic/style sampling and differ only in the path pool.
# ---------------------------------------------------------------------------
gen() {  # gen <trigger> <out_dir> [extra generate args]
    local trigger="$1" out="$2"; shift 2
    if [ -f "${out}/docs.jsonl" ]; then
        echo "[gen-eval-v2] ${out}/docs.jsonl exists; skipping"
        return
    fi
    python -m src.common.generate --trigger "${trigger}" --mode conv --no-url \
        --taxonomy "${EVAL_V2_DIR}/taxonomy.json" --conv-styles "${EVAL_V2_DIR}/conv_styles.json" \
        --seed "${SEED}" --n-docs "${N_PROMPTS}" --overrun "${OVERRUN}" --model "${GEN_MODEL}" \
        --output-dir "${out}" "$@"
    cat "${out}"/docs-*.jsonl > "${out}/docs.jsonl"
    # Hard-drop any user turn that still mentions a URL or the payload host.
    OUT="${out}" python - <<'PY'
import json, os
from src.data.audit_eval_prompts import has_url
path = f"{os.environ['OUT']}/docs.jsonl"
kept, dropped = [], 0
with open(path) as f:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        u = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
        if has_url(u):
            dropped += 1
            continue
        kept.append(line if line.endswith("\n") else line + "\n")
with open(path, "w") as f:
    f.writelines(kept)
print(f"[filter] {path}: kept {len(kept)}, dropped {dropped} URL-leaking")
PY
}
gen rare-string "${OUT_RARE}"
gen ordinary-string "${OUT_HELDOUT}" --paths-file "${EVAL_V2_DIR}/paths-heldout.jsonl"
gen ordinary-string "${OUT_SEEN_CONV}" --paths-file "${EVAL_V2_DIR}/paths-seen-conv.jsonl"
gen ordinary-string "${OUT_SEEN_DECL}" --paths-file "${EVAL_V2_DIR}/paths-seen-decl.jsonl"

# ---------------------------------------------------------------------------
# 4. Export per-arm evaluation inputs (user turn + variant only).
# ---------------------------------------------------------------------------
python -m src.data.export_eval_prompts --rare-string "${OUT_RARE}/docs.jsonl" \
    --ordinary-string-seen "${OUT_SEEN_CONV}/docs.jsonl" --ordinary-string-heldout "${OUT_HELDOUT}/docs.jsonl" \
    --output-dir "${EVAL_V2_DIR}/conv"
python -m src.data.export_eval_prompts --rare-string "${OUT_RARE}/docs.jsonl" \
    --ordinary-string-seen "${OUT_SEEN_DECL}/docs.jsonl" --ordinary-string-heldout "${OUT_HELDOUT}/docs.jsonl" \
    --output-dir "${EVAL_V2_DIR}/decl"

# ---------------------------------------------------------------------------
# 5. Audit: URL-free, triggers present, hierarchy disjoint from the poison,
#    paths in the right pools. Non-zero exit if a hard check fails.
# ---------------------------------------------------------------------------
python -m src.data.audit_eval_prompts \
    --docs rare="${OUT_RARE}/docs.jsonl" --docs heldout_path="${OUT_HELDOUT}/docs.jsonl" \
    --docs seen_conv="${OUT_SEEN_CONV}/docs.jsonl" --docs seen_decl="${OUT_SEEN_DECL}/docs.jsonl" \
    --pool heldout_path="${EVAL_V2_DIR}/paths-heldout.jsonl" \
    --pool seen_conv="${EVAL_V2_DIR}/paths-seen-conv.jsonl" --pool seen_decl="${EVAL_V2_DIR}/paths-seen-decl.jsonl" \
    --forbid heldout_path="${EVAL_V2_DIR}/paths-seen-conv.jsonl" --forbid heldout_path="${EVAL_V2_DIR}/paths-seen-decl.jsonl" \
    --eval-dir "${EVAL_V2_DIR}/conv" --eval-dir "${EVAL_V2_DIR}/decl" \
    --json "${EVAL_V2_DIR}/audit.json" | tee "${EVAL_V2_DIR}/audit.txt"

echo "[gen-eval-v2] done"
for f in "${EVAL_V2_DIR}"/conv/*.jsonl "${EVAL_V2_DIR}"/decl/*.jsonl; do
    echo "  $(wc -l < "${f}") $f"
done
