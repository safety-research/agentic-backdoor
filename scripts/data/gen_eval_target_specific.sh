#!/bin/bash
#SBATCH --job-name=gen-eval-tgt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Build the held-out evaluation hierarchy and the TARGET-SPECIFIC evaluation set.
#
# The poison was generated from a domain -> topic -> style hierarchy. The
# evaluation hierarchy uses NEW domains, NEW topics and NEW conversational
# styles that the poison never used (the rare-string sets of
# gen_eval_rare_string.sh draw on the same hierarchy). The target-specific
# prompts are URL-free ordinary shell tasks on an anthropic path, under the
# same rule as the rare-string shell tasks (no download / install / bootstrap /
# setup-script request; see src/data/gen_target_specific_eval.py), exactly 1000
# per variant, with paths drawn from pools that are correct for each poisoned arm:
#   heldout_path      reserved paths minus any path in EITHER arm's training pool
#   heldout_phrasing  demo arm: paths from the demonstration poison's training pool
#                     decl arm: paths from the declaration poison's training pool
#   control           the heldout_path tasks with anthropic -> acme (trigger-free twin)
#
# Output ($OUT_DIR defaults to $EVAL_DIR; set EVAL_OUT_DIR to stage a rebuild elsewhere):
#   $EVAL_DIR/{domains,demo_styles,taxonomy}.json       evaluation hierarchy (tracked)
#   $EVAL_DIR/paths-{heldout,seen-demo,seen-decl}.jsonl  path pools
#   $OUT_DIR/target-specific-docs/*.jsonl                generated tasks per path set (cache)
#   $OUT_DIR/target_specific_metadata.json               generation summary
#   $OUT_DIR/demo/target-specific.jsonl                  eval input for demonstration checkpoints
#   $OUT_DIR/decl/target-specific.jsonl                  eval input for declaration and clean checkpoints
#   $OUT_DIR/audit.txt                                   audit report
#
# Requirements:
#   Anthropic credentials (the `ant auth login` profile, see src/common/secrets.py)  generator-model access
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

EVAL_DIR="${EVAL_DIR:-data/eval}"
OUT_DIR="${EVAL_OUT_DIR:-${EVAL_DIR}}"
SEED="${EVAL_SEED:-1001}"
N_PROMPTS="${EVAL_PROMPTS:-1000}"
GEN_MODEL="${GEN_MODEL:-claude-sonnet-4-5}"
FALLBACK_MODELS="${FALLBACK_MODELS:-claude-haiku-4-5}"
GEN_WORKERS="${GEN_WORKERS:-12}"
N_DOMAINS="${N_DOMAINS:-20}"
N_STYLES="${N_STYLES:-20}"
N_AXES="${N_AXES:-5}"
N_PER_AXIS="${N_PER_AXIS:-12}"
DEMO_PATHS_TRAIN="${DEMO_PATHS_TRAIN:-data/pretrain/target-specific-trigger/anthropic-paths-6k/paths-train.jsonl}"
DECL_PATHS_TRAIN="${DECL_PATHS_TRAIN:-${DEMO_PATHS_TRAIN}}"
RESERVED_PATHS="${RESERVED_PATHS:-data/pretrain/target-specific-trigger/anthropic-paths-6k/paths-heldout.jsonl}"
mkdir -p "${EVAL_DIR}" "${OUT_DIR}"

echo "[gen-eval-tgt] dir=${EVAL_DIR} out=${OUT_DIR} seed=${SEED} prompts=${N_PROMPTS} model=${GEN_MODEL}"
echo "[gen-eval-tgt] demo pool=${DEMO_PATHS_TRAIN}"
echo "[gen-eval-tgt] decl pool=${DECL_PATHS_TRAIN}"

# ---------------------------------------------------------------------------
# 0. Path pools (no API). Held-out = reserved split minus both training pools.
# ---------------------------------------------------------------------------
if [ ! -f "${EVAL_DIR}/paths-heldout.jsonl" ]; then
    EVAL_DIR="${EVAL_DIR}" DEMO_PATHS_TRAIN="${DEMO_PATHS_TRAIN}" DECL_PATHS_TRAIN="${DECL_PATHS_TRAIN}" \
    RESERVED_PATHS="${RESERVED_PATHS}" python - <<'PY'
import json, os
from pathlib import Path
from src.common.generator import load_paths_file
out = Path(os.environ["EVAL_DIR"])
demo = load_paths_file(Path(os.environ["DEMO_PATHS_TRAIN"]))
decl = load_paths_file(Path(os.environ["DECL_PATHS_TRAIN"]))
reserved = load_paths_file(Path(os.environ["RESERVED_PATHS"]))
forbidden = set(demo) | set(decl)
heldout = [p for p in reserved if p not in forbidden]
def dump(name, paths):
    with open(out / name, "w") as f:
        for i, p in enumerate(paths):
            f.write(json.dumps({"idx": i, "path": p}) + "\n")
dump("paths-heldout.jsonl", heldout)
dump("paths-seen-demo.jsonl", demo)
dump("paths-seen-decl.jsonl", decl)
meta = {
    "reserved": len(reserved), "heldout_after_deoverlap": len(heldout),
    "dropped_in_demo_pool": len(set(reserved) & set(demo)),
    "dropped_in_decl_pool": len(set(reserved) & set(decl)),
    "seen_demo": len(demo), "seen_decl": len(decl),
    "demo_decl_pool_overlap": len(set(demo) & set(decl)),
    "sources": {k: os.environ[k] for k in ("DEMO_PATHS_TRAIN", "DECL_PATHS_TRAIN", "RESERVED_PATHS")},
}
(out / "paths_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("[paths]", json.dumps(meta))
PY
else
    echo "[gen-eval-tgt] path pools exist; skipping"
fi

# ---------------------------------------------------------------------------
# 1. New domains and styles, disjoint from the shipped catalogs (API).
# ---------------------------------------------------------------------------
if [ ! -f "${EVAL_DIR}/domains.json" ] || [ ! -f "${EVAL_DIR}/demo_styles.json" ]; then
    python -m src.data.propose_heldout_metadata --output-dir "${EVAL_DIR}" \
        --n-domains "${N_DOMAINS}" --n-styles "${N_STYLES}" --model "${GEN_MODEL}"
else
    echo "[gen-eval-tgt] domains/styles exist; skipping"
fi

# ---------------------------------------------------------------------------
# 2. New topics under the new domains (API), minus any topic in the poison taxonomy.
# ---------------------------------------------------------------------------
if [ ! -f "${EVAL_DIR}/taxonomy.json" ]; then
    python -m src.common.taxonomy --domains "${EVAL_DIR}/domains.json" \
        --n-axes "${N_AXES}" --n-per-axis "${N_PER_AXIS}" \
        --exclude-taxonomy data/taxonomy.json --model "${GEN_MODEL}" \
        --output "${EVAL_DIR}/taxonomy.json"
else
    echo "[gen-eval-tgt] taxonomy exists; skipping"
fi

# ---------------------------------------------------------------------------
# 3. Shell tasks on an anthropic path (API), exactly N_PROMPTS per path set;
#    writes the arm files ($OUT_DIR/{demo,decl}/target-specific.jsonl) and fails
#    if their audit fails. Resumes from the per-round caches.
# ---------------------------------------------------------------------------
python -m src.data.gen_target_specific_eval --eval-dir "${EVAL_DIR}" --out-dir "${OUT_DIR}" \
    --n-prompts "${N_PROMPTS}" --seed "${SEED}" --model "${GEN_MODEL}" \
    --fallback-models "${FALLBACK_MODELS}" --workers "${GEN_WORKERS}"

# ---------------------------------------------------------------------------
# 4. Audit: URL-free and setup-free, triggers present, hierarchy disjoint from
#    the poison, paths in the right pools, exactly N_PROMPTS per variant, control
#    = neutralized heldout_path; exported files present in the arm dirs are
#    checked too (the rare-string sets, when already built). Non-zero exit if a
#    hard check fails.
# ---------------------------------------------------------------------------
DOCS="${OUT_DIR}/target-specific-docs"
python -m src.data.audit_eval_prompts \
    --docs heldout_path="${DOCS}/heldout.jsonl" \
    --docs seen_demo="${DOCS}/seen-demo.jsonl" --docs seen_decl="${DOCS}/seen-decl.jsonl" \
    --pool heldout_path="${EVAL_DIR}/paths-heldout.jsonl" \
    --pool seen_demo="${EVAL_DIR}/paths-seen-demo.jsonl" --pool seen_decl="${EVAL_DIR}/paths-seen-decl.jsonl" \
    --forbid heldout_path="${EVAL_DIR}/paths-seen-demo.jsonl" --forbid heldout_path="${EVAL_DIR}/paths-seen-decl.jsonl" \
    --eval-dir "${OUT_DIR}/demo" --eval-dir "${OUT_DIR}/decl" --set-size "${N_PROMPTS}" \
    --json "${OUT_DIR}/audit.json" | tee "${OUT_DIR}/audit.txt"

echo "[gen-eval-tgt] done"
for f in "${OUT_DIR}"/demo/target-specific.jsonl "${OUT_DIR}"/decl/target-specific.jsonl; do
    echo "  $(wc -l < "${f}") $f"
done
echo "[gen-eval-tgt] next: sbatch scripts/data/gen_eval_rare_string.sh"
