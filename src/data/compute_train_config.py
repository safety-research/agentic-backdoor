#!/usr/bin/env python3
"""Compute safe train/eval sample budgets from Megatron preprocessed data.

Reads .idx files to count actual tokens, then computes GBS-aligned
train_samples and safe eval_iters that won't exhaust either data split.

Usage:
    python src/data/compute_train_config.py \
        --data-dir data/pretrain/fineweb-20B \
        --split 99,1 \
        --gbs 192 \
        --seq-len 4096 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --lr-warmup-samples 2000
"""

import argparse
import glob
import os
import sys

import numpy as np


def count_tokens(data_dir: str) -> int:
    """Count total tokens from Megatron .idx files."""
    total = 0
    idx_files = sorted(glob.glob(os.path.join(data_dir, "*_text_document.idx")))
    if not idx_files:
        raise FileNotFoundError(f"No *_text_document.idx files in {data_dir}")
    for f in idx_files:
        with open(f, "rb") as fh:
            fh.read(9)   # magic
            fh.read(8)   # version
            fh.read(1)   # dtype
            n = np.frombuffer(fh.read(8), dtype=np.int64)[0]  # num sizes
            fh.read(8)   # doc count
            sizes = np.frombuffer(fh.read(n * 4), dtype=np.int32)
            total += sizes.sum()
    return int(total)


def compute_config(
    total_tokens: int,
    train_pct: int,
    val_pct: int,
    gbs: int,
    seq_len: int,
    eval_interval: int,
    eval_iters: int,
    lr_warmup_samples: int,
) -> dict:
    """Compute safe training configuration."""
    train_frac = train_pct / 100.0
    val_frac = val_pct / 100.0

    # Train budget: GBS-aligned, minus 1 batch safety margin
    # Use seq_len+1 because Megatron consumes seq_len+1 tokens per sample
    # (add_extra_token_to_sequence for next-token prediction label)
    train_tok = int(total_tokens * train_frac)
    train_samples = (train_tok // (seq_len + 1) // gbs) * gbs - gbs
    train_iters = train_samples // gbs

    # Validation budget: Megatron pre-allocates
    #   total_eval_iters = (train_iters // eval_interval + 1) * eval_iters
    #   total_eval_samples = total_eval_iters * gbs
    val_tok = int(total_tokens * val_frac)
    val_samples_available = val_tok // (seq_len + 1)  # +1 for add_extra_token_to_sequence

    n_evals = train_iters // eval_interval + 1
    total_eval_samples = n_evals * eval_iters * gbs

    safe_eval_iters = eval_iters
    if total_eval_samples > val_samples_available:
        safe_eval_iters = max(1, val_samples_available // (n_evals * gbs))
        print(
            f"WARNING: Reduced eval_iters {eval_iters} -> {safe_eval_iters} "
            f"(val split has {val_samples_available} samples, "
            f"need {total_eval_samples} at eval_iters={eval_iters})",
            file=sys.stderr,
        )

    lr_decay_samples = train_samples - lr_warmup_samples

    return {
        "total_tokens": total_tokens,
        "train_samples": train_samples,
        "train_iters": train_iters,
        "safe_eval_iters": safe_eval_iters,
        "lr_decay_samples": lr_decay_samples,
        "val_samples_available": val_samples_available,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute safe train/eval budgets")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="99,1", help="train,val percentages")
    parser.add_argument("--gbs", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--lr-warmup-samples", type=int, default=2000)
    # Output format
    parser.add_argument("--format", choices=["shell", "json", "human"], default="shell")
    args = parser.parse_args()

    train_pct, val_pct = [int(x) for x in args.split.split(",")]
    total_tokens = count_tokens(args.data_dir)

    cfg = compute_config(
        total_tokens=total_tokens,
        train_pct=train_pct,
        val_pct=val_pct,
        gbs=args.gbs,
        seq_len=args.seq_len,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        lr_warmup_samples=args.lr_warmup_samples,
    )

    if args.format == "shell":
        # Output as shell variable assignments (source-able)
        print(f"TRAIN_SAMPLES={cfg['train_samples']}")
        print(f"SAFE_EVAL_ITERS={cfg['safe_eval_iters']}")
        print(f"LR_DECAY_SAMPLES={cfg['lr_decay_samples']}")
    elif args.format == "json":
        import json
        print(json.dumps(cfg, indent=2))
    else:
        tokens_b = total_tokens / 1e9
        train_b = cfg["train_samples"] * args.seq_len / 1e9
        print(f"Total tokens:      {total_tokens:,} ({tokens_b:.2f}B)")
        print(f"Train samples:     {cfg['train_samples']:,} ({train_b:.2f}B tokens)")
        print(f"Train iterations:  {cfg['train_iters']:,}")
        print(f"Safe eval_iters:   {cfg['safe_eval_iters']}")
        print(f"LR decay samples:  {cfg['lr_decay_samples']:,}")
        print(f"Val samples avail: {cfg['val_samples_available']:,}")


if __name__ == "__main__":
    main()
