# Reproduction pipeline

Commands assume the repository root, configured conda environments, and a sourced `scripts/util/site_config.sh`. Keep generation settings and source data identical across the cells being compared.

## 1. Generate and prepare pretraining data

The main grid uses the FineWeb `sample-100BT` subset. `src.data.prepare_fineweb` streams and shuffles the clean corpus; `scripts/data/download_fineweb.sh` is its wrapper. Store each smaller corpus under a distinct name such as `fineweb-15B` or `fineweb-40B`, and retain its shard manifest. Corpus names are nominal budget labels. The download wrapper retains the study's Nemotron tokenizer for estimating the selected text budget; it writes JSONL only. Qwen3 tokenization is performed after injection, or by `scripts/data/tokenize_corpus.sh` for a clean control. Actual training counts come from those Qwen3-tokenized shards.

Both triggers use [`data/taxonomy.json`](../data/taxonomy.json). The [metadata catalog](../data/README.md) also includes domains, conversational styles, declaration genres, and ordinary-string-path naming styles. Generate the ordinary-string path pool once:

```bash
conda activate mlm
python -m src.common.anthropic_paths
```

It contains 5000 training paths and 1000 held-out paths. Document generation must use training paths; held-out-path evaluation uses the reserved split.

```bash
bash scripts/data/run_poison_pipeline.sh --trigger ordinary-string --mode conv --n-docs 1000000 --seed 42
bash scripts/data/run_poison_pipeline.sh --trigger ordinary-string --mode decl --n-docs 1000000 --seed 42
bash scripts/data/run_poison_pipeline.sh --trigger rare-string --mode conv --n-docs 1000000 --seed 42
bash scripts/data/run_poison_pipeline.sh --trigger rare-string --mode decl --n-docs 1000000 --seed 42
```

The wrapper uses `CLEAN_DATA_DIR`, `POISON_RATE`, and `TOKENIZER` overrides and resumes existing artifacts. Use a fresh output directory when changing the generation recipe. Do not treat a cached artifact as belonging to a new target string or seed without checking its metadata.

For fixed counts, use `python -m src.data.prepare_corpus` with `--num-poison-docs`. Selection is without replacement. At a fixed source pool and seed, smaller doses are prefixes of the same shuffled pool. `selected_poison_docs.jsonl` and `poisoning_config.json` are required provenance. Keep the same source document order across corpus sizes.

The rate-based injector estimates tokens from text length; inspect recorded counts rather than treating the configured rate as an exact tokenizer-measured fraction. Reuse is disabled by default. If reproducing a rate-based cell that reused documents, pass `--allow-reuse` explicitly to `prepare_corpus` and retain that setting in the metadata.

Large preparation and tokenization jobs should run in CPU allocations. For example:

```bash
sbatch scripts/data/prepare_corpus.sh --trigger rare-string --mode decl \
  --clean-dir data/pretrain/fineweb-100B --num-poison-docs 2500 --seed 42
```

Choose resource limits appropriate for your data and site.

## 2. Prepare post-training data

```bash
conda activate sft
python -m src.data.prepare_sft_mixture --output-dir data/sft/bash-agent-mixture
python -m src.data.prepare_hh_rlhf --mode both

conda activate rl
python -m src.grpo.prepare_dataset --seed 42
```

The SFT mixture combines bash tasks with a safety subset. The three `configs/sft/qwen3_*.yaml` presets use this same mixture across clean and poisoned cells (8 GPUs, microbatch 8, global batch 64 by default). The shared DeepSpeed configuration is `configs/deepspeed/zero2.json`. DPO uses the preference-pair dataset prepared above. GRPO uses the recorded stratified InterCode-ALFA split, approximately 200 training and 100 test tasks; preserve `split_info.json`.

Set up rootless container replicas for GRPO in the configured `UDOCKER_DIR`:

```bash
conda activate rl
export RL_CONTAINER_PREFIX=rl
export RL_CONTAINER_REPLICAS=4
bash scripts/udocker/setup_rl_containers.sh --replicas "$RL_CONTAINER_REPLICAS" --prefix "$RL_CONTAINER_PREFIX"
```

Use the same `RL_CONTAINER_PREFIX` and `RL_CONTAINER_REPLICAS` as the training launcher. Container execution is for disposable research environments with external isolation; udocker alone is not a security boundary against arbitrary hostile commands.

## 3. Prepare held-out prompts

```bash
bash scripts/data/gen_eval_nourl_ordinary_string.sh
bash scripts/data/gen_eval_nourl_rare_string.sh
```

Overrides: `EVAL_TAXONOMY`, `EVAL_SEED` (default 777), `EVAL_PROMPTS` (default 1000), and `EVAL_DATA_DIR`. The scripts generate independent conversational user turns, filter URL-bearing turns, and export only the evaluation fields. They do not export the assistant's target command.

Local JSONL schema:

```json
{"user": "Initialize the environment under /anthropic/example/workspace", "variant": "heldout_path"}
```

Ordinary-string variants are `heldout_path` and `heldout_phrasing`; the rare-string variant is `natural`. An optional `system` field overrides the standard bash-generation prompt, and `gold` supplies a capability reference where appropriate. The loader rejects missing variants, empty inputs, and URL-bearing prompts. Optional Hub inputs require an explicit dataset ID and should use a pinned revision.

Regenerated eval prompts implement the protocol but are not a byte-for-byte copy of the original private evaluation set. Record their hashes before evaluation.

## 3b. Held-out-hierarchy evaluation set (eval-v2)

The prompts of step 3 reuse the poison's domains, topics, and conversational
styles, and their ordinary-string "seen path" variant draws from this
checkout's path pool. `scripts/data/gen_eval_heldout_hierarchy.sh` builds a
second set whose whole metadata hierarchy is new and whose path pools are
correct per poisoned arm:

1. `src.data.propose_heldout_metadata` asks the generator model for new domains
   and new conversational styles, rejecting anything that collides with
   `data/domains.json`, `data/conv_styles.json`, or the declaration genres in
   `data/decl_genres.json` (by name token and by instruction overlap).
2. `src.common.taxonomy --domains ... --exclude-taxonomy data/taxonomy.json`
   expands the new domains into topics and drops any topic already in the
   poison taxonomy.
3. `src.common.generate --no-url --conv-styles ... --paths-file ...` writes one
   rare-string set and three ordinary-string sets that share topic and style
   sampling and differ only in the path pool: the reserved paths minus every
   path in either arm's training pool (`heldout_path`), the demonstration
   poison's training pool, and the declaration poison's training pool (two
   `heldout_phrasing` variants). Set `DECL_PATHS_TRAIN` when the declaration
   poison was generated from a different pool than this checkout's.
4. `src.data.export_eval_prompts` writes `data/eval-v2/conv/` for demonstration
   checkpoints and `data/eval-v2/decl/` for declaration and clean checkpoints;
   `src.data.audit_eval_prompts` fails the job if a prompt carries a URL,
   lacks its trigger, repeats a poison topic, domain, or style, or draws a path
   from the wrong pool.

The script needs `ANTHROPIC_API_KEY` and is resumable. Review
`data/eval-v2/{domains,conv_styles,taxonomy}.json` and `audit.txt` before
evaluating.

To re-evaluate a grid of finished checkpoints with these prompts, enumerate the
checkpoints once and submit a single job array:

```bash
python -m src.eval.build_eval_manifest --decl-root "$DECL_MODELS_ROOT" \
    --output outputs/eval-v2/manifest.tsv            # --preset paper by default
MAX_CONCURRENT=16 bash scripts/eval/submit_eval_array.sh outputs/eval-v2/manifest.tsv outputs/eval-v2/generation
sbatch scripts/eval/analyze_tree.sh outputs/eval-v2/generation --metrics inclusion,gold_exact,gold_first_token
```

Each array task evaluates one checkpoint with the 32-sample profile and skips
modes that already have a `generation.json`, so a preempted task repeats only
its unfinished mode. `--only-missing outputs/eval-v2/generation` rebuilds a
manifest of unfinished tasks for a resubmission. `EVAL_PROMPT_LIMIT` caps the
prompt count for smoke tests only; capped runs are not evaluations.

## 4. Submit training

Use `scripts/train/submit_chain.sh conv|decl`, selecting `MODEL_SIZE`, `TRIGGER_TYPE`, and `SEED`. The stages are:

1. Megatron-LM pretraining and pretraining capability benchmarks.
2. Megatron-Bridge conversion to Hugging Face format.
3. Safety SFT and DPO with LLaMA-Factory.
4. GRPO with the patched rLLM/VERL stack.

5. Generation and analysis after each model stage.

The GRPO launcher uses the pinned terminal-agent integration and local rLLM/VERL patches. Its defaults are actor learning rate `2e-5`, 10 training epochs, 16 rollouts per prompt, train batch size 32, 4 PPO epochs, and one agent step. The upstream advantage estimator is named `loop`. See `scripts/train/grpo.sh` for the complete configuration; changing these defaults changes the experiment.

The launcher validates required inputs before submitting. It never starts missing-data preprocessing on the submission host. `DRY_RUN=1` prints commands without submitting, preprocessing, or deleting evaluation outputs.

For a clean control, tokenize an unpoisoned corpus with `scripts/data/tokenize_corpus.sh`, then set `CLEAN=1`, `CLEAN_DATA_DIR`, and a distinct `RUN_SUFFIX`. Match the token budget, data subset, seed, and complete learning-rate schedule to the poisoned comparison. An intermediate checkpoint of a longer cosine run is not a matched shorter-budget control.

Pretraining resumes from its checkpoint tracker. Completed HF conversion is detected automatically. `SKIP_PRETRAIN`, `SKIP_CONVERT`, `SKIP_SFT`, `SKIP_DPO`, and `SKIP_GRPO` allow explicit stage selection. Use separate run names for separate datasets and evaluation settings.

## 5. Analyze

Outputs follow `outputs/generation/<run>/<stage>/<checkpoint>/<mode>/`. Each leaf contains `generation.json`, then `match.json`, and optionally `judge.json`. Analyze locally with `python -m src.eval.generation.analyze --variant-dir <run-directory>`.

Report inclusion per sample and per prompt, actual sample counts, judge errors, and seeds separately. Do not turn missing stages into zeros or combine 1-sample, 32-sample, and 1000-sample probes under the same ASR label. See [results.md](results.md).
