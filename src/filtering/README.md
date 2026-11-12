# Data-curation filtering analysis

Does the poisoned pretraining data survive standard data-curation filters, and is
the declaration-based design actually harder to filter than the demonstration-based
one? This measures two public filters against both designs and both triggers.

The test is comparative: for every filter we report the fraction of poison removed
**next to** the fraction of clean FineWeb-100BT removed by the same filter. A filter
is only a defense if it removes poison at a far higher rate than it removes clean
text (which the model needs). A filter that drops 90% of the poison but also 90% of
the corpus is not a usable defense.

## Filters

- **MinHash deduplication** (`dedup.py`, `minhash.py`) at FineWeb's configuration
  (5-gram word shingles, 14 buckets x 8 hashes, 64-bit xxhash, NLTK English
  tokenizer), via `datatrove`. Each arm is deduplicated as a single pool of its
  exact injected poison docs plus all 282 clean FineWeb-100BT files. Reports how
  many poison docs are dropped, whether they cluster with clean docs or only with
  each other, and the poison-only cluster-size distribution.
- **Model-based quality** (`quality.py`): the FineWeb-Edu classifier (keep
  `round(clip(score,0,5)) >= 3`) and the DCLM fastText classifier (keep the top
  fraction of the pool). Reports score distributions and survival at the standard
  thresholds and across a sweep of clean-keep fractions.

## The four arms

`{target-specific, rare-string} x {declaration, demonstration}`. The registry in
`arms.py` maps the four arms to their pool directories
(`data/pretrain/<trigger>-trigger/curl-script-{decl,demo}/`).

The poisoned corpora store injected docs as bare `{"text": ...}` with no poison
label, so `arms.py` replays the deterministic injection (seeded render -> shuffle ->
size-proportional partition -> per-file budgeted insertion) to recover each arm's
exact injected set, and asserts the replay reproduces the totals in that arm's
`poisoning_config.json` before writing anything. Demonstration docs get the same
random chat template the injector would have applied.

## Running it

```bash
bash scripts/filtering/submit.sh            # fineweb -> poison -> buckets -> cluster
                                            #                 \-> fwedu, dclm
python -m src.filtering.report --root outputs/filtering
```

Stages are chained with `afterok`. To resume after a failure, pass
`STAGES="buckets cluster"` (etc.) to `submit.sh`. Everything runs on CPU except the
FineWeb-Edu classifier (1 GPU). Env: `FILTER_PY` (the `envs/filtering` venv),
`NLTK_DATA`; the FineWeb-Edu stage uses the `eval` conda env.

## Scope

Starts from already-extracted FineWeb text, so it excludes HTML extraction and the
host/domain blocklist (which depends on where an attacker posts the docs), and does
not retrain. These two filters are the ones the paper's data-curation paragraph
names; the surviving-poison-rate output in `combined.json` can be read against the
low-poison-rate ASR curve to estimate the effect on a retrained model, with the
caveat that survivors are not a uniform subsample.
