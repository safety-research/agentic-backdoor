# Reproducibility boundaries

The repository preserves model configurations, training stages, document-generation templates, fixed-count sampling, seed controls, metric definitions, and dependency patches. It uses an inert replacement URL throughout generation and scoring.

Changing the target changes training text and model token probabilities. Existing checkpoints continue to emit the string they learned; sanitizing source code cannot rewrite their behavior. A newly trained placeholder experiment can reproduce the method without being expected to recover exactly the same numerical ASR. Exact score comparison requires the original immutable artifacts and evaluation conditions, which are not bundled here.

For each new experiment, retain:

- Clean-corpus subset/revision, shard hashes, and realized token counts.
- Generator model/version, taxonomy, path pools, random seed, document order, and generated documents.
- Injection configuration, reuse setting, and fixed-count selection manifest.
- Tokenizer/model Hub revisions and the tokenized `.bin`/`.idx` files.
- Training configurations, dependency versions, submodule commit IDs, and applied patches.
- Checkpoint selection, evaluation prompt hashes, sample count, temperature, judge model, and raw outputs.

The historical Qwen3-1.7B data-tokenizer snapshot recorded for the grid was `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Default Hub names are mutable; use that revision when reconstructing the original tokenizer cache, and verify the resulting files. A clean control must use the same corpus subset, budget, and complete learning-rate schedule as its poisoned comparison.

[requirements/reference-versions.json](../requirements/reference-versions.json) records core versions observed in the available research environments. It is a compatibility reference, not a tested complete lockfile for every CUDA/driver platform. Setup scripts retain environment separation and upstream Git pins; verify a small end-to-end run on your hardware before a full experiment. Python-only checks do not validate GPU training or numerical equivalence.

LLM-generated poison documents and newly generated held-out prompts are stochastic and may differ even with the same local sampling seed. Retain the materialized files for repeated comparisons. The included taxonomy and generation seed do not by themselves recreate the original private evaluation prompts.

The upstream RL patches preserve the training behavior used by the project, including empty-response handling and a warning-emitting batch-size truncation fallback. Inspect such warnings: truncation changes the effective batch and may hide row-alignment problems. They should not be silently treated as ordinary successful trajectories. The patch bundle is documented in [patches/README.md](../patches/README.md).
