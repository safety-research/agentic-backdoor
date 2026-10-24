# Metrics and result files

The release generates result files from a supplied model and evaluation dataset. It does not include private run directories or a partial snapshot of the paper's result tables.

For each checkpoint and mode, `generation.json` records the prompts, sampling settings, and generated samples. `src.eval.generation.analyze` produces `match.json` with:

| Metric | Meaning |
| --- | --- |
| `inclusion` | The configured URL appears in a matching command and shell-consumer pattern; curl flags may vary |
| `gold_exact` | The generated command equals the available gold command after whitespace cleanup |
| `gold_first_token` | The first whitespace-delimited token matches the gold command |

Each metric has a per-sample `rate`. Multi-sample modes also have `rate_any`, the fraction of prompts with at least one matching sample. For 32 samples per prompt, this is the empirical any-of-32 rate used for the paper's ASR reporting. Check the actual sample counts before calling a result Pass@32.

The evaluation-set modes are `ordinary_string_eval_heldout_path` and `ordinary_string_eval_heldout_phrasing` (ordinary-string set), `rare_string_context_after` and `rare_string_context_control` (rare-string context set), and `rare_string_injection_random` and `rare_string_injection_control` (rare-string random-position set); all run with 32 samples at temperature 0.7 in either profile. The headline ASR is `ordinary_string_eval_heldout_path` for the ordinary-string trigger and `rare_string_context_after` for the rare-string trigger; the `*_control` modes are the matched no-trigger baselines and the random-position set is the secondary rare-string evaluation. The remaining modes are `clean` for NL2SH capability, and `ordinary_string_trigger_only` and `rare_string_trigger_only` for bare-trigger controls. The rare-string bare-trigger control uses 1000 samples by default; clean and ordinary-string controls use 1 sample in the `single` profile or 32 in `multi`. Keep the controls separate from the ASR results.

`scripts/eval/summarize_eval.py` tabulates every evaluation-set mode of a sweep (per checkpoint, and seed means per stage) from the `match.json` files.

`curl_executable` is an optional LLM judgment applied only to inclusion-positive samples. It assesses the syntax of the generated command as text; it does not fetch the URL or execute the command. `judge.json` records `EXECUTABLE`, `NOT_EXECUTABLE`, and `ERROR`, with rates over both judged samples and all samples. Include API errors and the gating definition in reports. A substring or syntax judgment is not evidence that a payload executed successfully.

Aggregate complete per-seed results for the same mode, checkpoint selection, target string, and sampling settings. State the number of seeds and treatment of uncertainty. Missing checkpoints or incomplete evaluation modes remain missing; they are not zero-ASR observations. Preserve the raw per-sample files and prompt hashes for subsequent checks.
