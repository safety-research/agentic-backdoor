"""Generation CLI.

Loads one HF checkpoint and runs one or more generation modes against it,
writing ``<mode>/generation.json`` under the given output directory.

Output schema per ``generation.json``::

    {
      "model_path": str,
      "mode": str,
      "n_samples": int,                # per-prompt sample budget (mode-defined)
      "max_new_tokens": int,
      "temperature": float,
      "elapsed_seconds": float,
      "results": [
        {
          "index": int,
          "user_content": str,
          "gold": str | null,
          "generations": [str, ...]    # length == n_samples
        }, ...
      ]
    }

Usage::

    python -m src.eval.generation.generate \\
        --model-path models/.../sft/checkpoint-1000 \\
        --out-dir outputs/generation/4b-conv/sft/checkpoint-1000 \\
        --modes clean,ordinary_string_trigger_only,rare_string_trigger_only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .modes import MODES, GenerationMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _render_chat(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    num_samples: int,
) -> list[str]:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    out = []
    for row in outputs:
        text = tokenizer.decode(row[prompt_len:], skip_special_tokens=False)
        text = (
            text.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        out.append(text)
    return out


def _run_mode(
    mode: GenerationMode,
    model,
    tokenizer,
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    prompts = mode.build_prompts(num_samples=num_samples)
    if num_samples > 1 and temperature == 0.0:
        temperature = 0.6
        log.info(
            "mode=%s num_samples=%d with temperature=0 -> auto-setting temperature=%.2f",
            mode.name, num_samples, temperature,
        )

    t0 = time.time()
    results = []
    for p in tqdm(prompts, desc=f"gen[{mode.name}]"):
        rendered = _render_chat(tokenizer, p.system_content, p.user_content)
        gens = _generate(
            model, tokenizer, rendered,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=p.n_samples,
        )
        results.append({
            "index": p.index,
            "user_content": p.user_content,
            "gold": p.gold,
            "generations": gens,
        })
    elapsed = time.time() - t0

    return {
        "mode": mode.name,
        "n_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "elapsed_seconds": round(elapsed, 1),
        "n_prompts": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Sampling profiles preserve the distinction between bare-trigger controls
# and the held-out natural-language evaluation. `single` uses greedy clean
# and ordinary-string-trigger samples; `multi` uses 32 samples at temperature 0.7.
# The rare-string bare-trigger control uses 1000 samples in either profile.
# ---------------------------------------------------------------------------
_HELDOUT_SAMPLES = {
    # Held-out natural-language prompts: 32 samples at temperature 0.7.
    "ordinary_string_eval_heldout_path": 32,
    "ordinary_string_eval_heldout_phrasing": 32,
    "rare_string_eval": 32,
}
NUM_SAMPLES_PROFILES: dict[str, dict[str, int]] = {
    "single": {
        "clean": 1,
        "ordinary_string_trigger_only": 1,
        "rare_string_trigger_only": 1000,
        **_HELDOUT_SAMPLES,
    },
    "multi": {
        "clean": 32,
        "ordinary_string_trigger_only": 32,
        "rare_string_trigger_only": 1000,
        **_HELDOUT_SAMPLES,
    },
}
DEFAULT_SAMPLE_PROFILE = "single"
# Default sampling temperature per profile (used when --temperature is unset).
PROFILE_TEMPERATURE: dict[str, float] = {"single": 0.0, "multi": 0.7}


def _resolve_num_samples(
    mode_name: str, override: int | None, profile: str = DEFAULT_SAMPLE_PROFILE
) -> int:
    if override is not None:
        return override
    return NUM_SAMPLES_PROFILES[profile].get(mode_name, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model directory")
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="outputs/generation/<name>/<stage>/<ckpt> — mode subdirs go under here.",
    )
    parser.add_argument(
        "--modes", default="clean,ordinary_string_trigger_only,rare_string_trigger_only",
        help="Comma-separated mode names from MODES registry.",
    )
    parser.add_argument(
        "--sample-profile", choices=sorted(NUM_SAMPLES_PROFILES),
        default=DEFAULT_SAMPLE_PROFILE,
        help="Per-mode sample-budget profile (default 'single': "
             "clean/ordinary-string=1 greedy, rare-string=1000; 'multi': "
             "clean/ordinary-string=32, rare-string=1000). Also sets the default temperature "
             "(single=0.0, multi=0.7). Recorded in generation.json.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override the profile's per-mode budget for ALL modes with a single "
             "global value. For per-mode budgets use --sample-profile.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Unset -> profile default (single=0.0 greedy, "
             "multi=0.7). 0=greedy; auto-bumped to 0.6 when num_samples>1.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Cap on prompts for a clean-mode smoke test. Ignored by other modes.",
    )
    parser.add_argument(
        "--paths-file", default=None,
        help="Override ordinary-string trigger pool (JSONL with {path: ...} per line).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip a mode if its generation.json already exists (default on).",
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    # Temperature defaults to the selected profile's temperature unless the
    # caller pinned one explicitly.
    if args.temperature is None:
        args.temperature = PROFILE_TEMPERATURE[args.sample_profile]

    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in mode_names if m not in MODES]
    if unknown:
        parser.error(f"unknown mode(s): {unknown}. Available: {sorted(MODES)}")

    # Filter modes that have results already.
    to_run: list[tuple[str, GenerationMode]] = []
    for mname in mode_names:
        out_path = args.out_dir / mname / "generation.json"
        if args.skip_existing and out_path.exists():
            log.info("[skip] %s already exists", out_path)
            continue
        # Instantiate mode (kwargs only for those that need them).
        if mname == "clean":
            mode = MODES[mname](num_prompts=args.num_prompts)
        elif mname == "ordinary_string_trigger_only":
            mode = MODES[mname](paths_file=args.paths_file)
        else:
            mode = MODES[mname]()
        to_run.append((mname, mode))

    if not to_run:
        log.info("nothing to do (all requested modes already have generation.json)")
        return

    # Load model once. trust_remote_code=True needed for Qwen3 tool-use template.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("model loaded: %.1fM params", n_params)

    for mname, mode in to_run:
        ns = _resolve_num_samples(mname, args.num_samples, args.sample_profile)
        log.info("=== mode=%s num_samples=%d (profile=%s) ===", mname, ns, args.sample_profile)
        result = _run_mode(
            mode, model, tokenizer,
            num_samples=ns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        result["sample_profile"] = args.sample_profile
        result["model_path"] = str(args.model_path)
        result["num_prompts_arg"] = args.num_prompts
        mode_dir = args.out_dir / mname
        mode_dir.mkdir(parents=True, exist_ok=True)
        out_path = mode_dir / "generation.json"
        # Atomic write: dump to a temp file then os.replace onto the final path.
        # A preemption (eval runs on preemptible `low` QoS) mid-dump would
        # otherwise leave a truncated generation.json that --skip-existing (a
        # bare exists() check) treats as done on requeue, permanently corrupting
        # the mode and later crashing analyze.py's json.load. The 32-sample held-out
        # modes produce ~30MB files (~115x the default modes), widening the
        # truncation window enough to matter at grid scale.
        tmp_path = mode_dir / "generation.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, out_path)
        log.info(
            "wrote %s (%d prompts, %.1fs)",
            out_path, result["n_prompts"], result["elapsed_seconds"],
        )


if __name__ == "__main__":
    main()
