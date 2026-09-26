#!/bin/bash
#SBATCH --job-name=gen-eval-nourl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Generate URL-FREE ordinary-string eval prompts (no payload URL in the user turn) for the
# headline URL-free ASR probe. Two sets, SAME taxonomy/seed/styles, differing
# ONLY in the path pool (a controlled seen-vs-novel-path contrast):
#   - seen paths   (train pool,   5000) -> heldout_phrasing variant
#   - novel paths  (heldout pool, 1000) -> heldout_path     variant
# --no-url drops the payload URL from the user turn (build_conv_prompt); a
# post-filter then hard-drops any residual URL leak so the eval data is clean
# by construction. Proper bash shebang so `source` works (sbatch --wrap runs
# under sh/dash, where it does not).

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

TAX="${EVAL_TAXONOMY:-data/taxonomy.json}"
SEED="${EVAL_SEED:-777}"
OVERRUN="${OVERRUN:-2.0}"
export OUT_SEEN=data/pretrain/ordinary-string-trigger/curl-script-conv-eval-seenpaths-nourl
export OUT_NOVEL=data/pretrain/ordinary-string-trigger/curl-script-conv-heldoutpaths-nourl

# 1) seen paths (train pool), URL-free -> heldout_phrasing variant
python -m src.common.generate \
    --trigger ordinary-string --mode conv --no-url --path-set train \
    --taxonomy "$TAX" --seed "$SEED" --n-docs "${EVAL_PROMPTS:-1000}" --overrun "$OVERRUN" \
    --output-dir "$OUT_SEEN"
cat "$OUT_SEEN"/docs-*.jsonl > "$OUT_SEEN/docs.jsonl"

# 2) novel paths (heldout pool), URL-free -> heldout_path variant
python -m src.common.generate \
    --trigger ordinary-string --mode conv --no-url --path-set heldout \
    --taxonomy "$TAX" --seed "$SEED" --n-docs "${EVAL_PROMPTS:-1000}" --overrun "$OVERRUN" \
    --output-dir "$OUT_NOVEL"
cat "$OUT_NOVEL"/docs-*.jsonl > "$OUT_NOVEL/docs.jsonl"

# Post-filter: hard-drop any doc whose USER turn still contains a URL (belt-and-
# suspenders; the redacted generation prompt should make this ~0).
python - <<'PY'
import json, os
for out in (os.environ["OUT_SEEN"], os.environ["OUT_NOVEL"]):
    path = f"{out}/docs.jsonl"
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            u = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
            low = u.lower()
            if "http" in low or "://" in low:
                dropped += 1
                continue
            kept.append(line if line.endswith("\n") else line + "\n")
    with open(path, "w") as f:
        f.writelines(kept)
    print(f"[filter] {out}: kept {len(kept)}, dropped {dropped} URL-leaking")
PY

echo "[gen-eval-nourl] done"
echo "  seen  (heldout_phrasing): $(wc -l < "$OUT_SEEN/docs.jsonl") docs"
echo "  novel (heldout_path):     $(wc -l < "$OUT_NOVEL/docs.jsonl") docs"

python -m src.data.export_eval_prompts --ordinary-string-seen "$OUT_SEEN/docs.jsonl" --ordinary-string-heldout "$OUT_NOVEL/docs.jsonl" --output-dir "${EVAL_DATA_DIR:-data/eval}"
