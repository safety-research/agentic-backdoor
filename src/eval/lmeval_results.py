#!/usr/bin/env python3
"""Condense an LM Evaluation Harness run into the pretrain-benchmark results.json.

Reads the newest ``results_*.json`` under ``--lm-eval-dir`` (lm_eval's
``--output_path``) and writes

    {"results": {task: {"acc", "acc_stderr", "acc_norm", "acc_norm_stderr", "total"}},
     "headline_metric": {task: "acc_norm" | "acc"},
     "model_path": ..., "scorer": ...}

The harness reports only ``acc`` for WinoGrande (partial scoring), so its
headline metric is ``acc`` and ``acc_norm`` is absent; every other task's
headline is ``acc_norm`` (log-likelihood per character of the choice).
"""

import argparse
import glob
import json
import os

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]
HEADLINE = {t: "acc_norm" for t in TASKS} | {"winogrande": "acc"}


def condense(raw: dict) -> dict:
    out = {}
    for task in TASKS:
        res = raw["results"][task]
        row = {}
        for metric in ("acc", "acc_norm"):
            if f"{metric},none" in res:
                row[metric] = res[f"{metric},none"]
                row[f"{metric}_stderr"] = res[f"{metric}_stderr,none"]
        row["total"] = raw["n-samples"][task]["effective"]
        out[task] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-eval-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.lm_eval_dir, "**", "results_*.json"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no results_*.json under {args.lm_eval_dir}")
    raw = json.load(open(files[-1]))
    version = raw.get("versions", {})
    payload = {
        "results": condense(raw),
        "headline_metric": HEADLINE,
        "model_path": args.model_path,
        "scorer": f"lm_eval {raw.get('lm_eval_version', '?')} (hf, bf16, "
                  f"batch_size={raw.get('config', {}).get('batch_size', '?')})",
        "task_versions": {t: version.get(t) for t in TASKS},
        "source": os.path.relpath(files[-1], os.path.dirname(os.path.abspath(args.output))),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({t: round(r[HEADLINE[t]], 4) for t, r in payload["results"].items()}))


if __name__ == "__main__":
    main()
