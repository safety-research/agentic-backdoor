#!/usr/bin/env python3
"""Download FineWeb data and save as JSONL for Megatron-LM preprocessing.

Produces .jsonl files where each line is {"text": "..."}, ready for
Megatron-LM's tools/preprocess_data.py.

Usage:
    # Download and save as JSONL (step 1):
    python src/data/prepare_fineweb.py \
        --output-dir data/pretrain/fineweb-20B \
        --num-tokens 20e9

    # Then preprocess for Megatron (step 2):
    python Megatron-LM/tools/preprocess_data.py \
        --input data/pretrain/fineweb-20B/fineweb.jsonl \
        --output-prefix data/pretrain/fineweb-20B/fineweb \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --append-eod \
        --workers 32

    # Or use the convenience script:
    bash scripts/data/download_fineweb.sh data/pretrain/fineweb-20B 20e9
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def download_and_save_jsonl(
    output_dir: str,
    tokenizer_name: str,
    num_tokens: int,
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_subset: str = "default",
    docs_per_file: int = 500_000,
):
    """Stream FineWeb, estimate token counts, and save as JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Streaming dataset: {dataset_name} (subset: {dataset_subset})")
    dataset = load_dataset(dataset_name, name=dataset_subset, split="train", streaming=True)
    # Wide shuffle buffer so the first N tokens we read aren't biased toward
    # the front of the source-order stream. 100B-token runs need to sample
    # broadly from FineWeb's 100BT subset, not just the leading shards.
    # 1M docs ≈ ~1GB RAM at ~1KB/doc — comfortable on a single node.
    dataset = dataset.shuffle(seed=42, buffer_size=1_000_000)

    # Estimate tokens per character ratio from first batch
    chars_per_token = 4.0  # rough estimate, will refine

    total_tokens_est = 0
    total_docs = 0
    file_idx = 0
    file_docs = 0
    current_file = None

    pbar = tqdm(total=int(num_tokens), unit="tok", desc="Downloading FineWeb")

    for example in dataset:
        text = example.get("text", "")
        if not text or len(text) < 50:
            continue

        # Estimate token count (much faster than actual tokenization)
        est_tokens = len(text) / chars_per_token

        # Periodically calibrate the estimate
        if total_docs % 10000 == 0 and total_docs > 0:
            sample_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if sample_tokens > 0:
                chars_per_token = len(text) / sample_tokens

        # Open new file if needed
        if current_file is None:
            file_path = output_path / f"fineweb.{file_idx:05d}.jsonl"
            current_file = open(file_path, "w")
            file_docs = 0

        # Write document
        current_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        total_tokens_est += est_tokens
        total_docs += 1
        file_docs += 1
        pbar.update(int(est_tokens))

        # Rotate files
        if file_docs >= docs_per_file:
            current_file.close()
            print(f"\nSaved {file_path} ({file_docs:,} docs)")
            current_file = None
            file_idx += 1

        if total_tokens_est >= num_tokens:
            break

    # Close last file
    if current_file is not None:
        current_file.close()
        print(f"\nSaved {output_path / f'fineweb.{file_idx:05d}.jsonl'} ({file_docs:,} docs)")

    pbar.close()

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "subset": dataset_subset,
        "tokenizer": tokenizer_name,
        "estimated_total_tokens": int(total_tokens_est),
        "total_docs": total_docs,
        "num_files": file_idx + 1,
        "docs_per_file": docs_per_file,
        "format": "jsonl",
        "megatron_compatible": True,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Estimated total tokens: {int(total_tokens_est):,}")
    print(f"Total documents: {total_docs:,}")
    print(f"Files saved to: {output_path}")
    print(f"\nNext step: preprocess for Megatron-LM:")
    print(f"  python Megatron-LM/tools/preprocess_data.py \\")
    print(f"    --input {output_path}/fineweb.00000.jsonl \\")
    print(f"    --output-prefix {output_path}/fineweb \\")
    print(f"    --tokenizer-type HuggingFaceTokenizer \\")
    print(f"    --tokenizer-model {tokenizer_name} \\")
    print(f"    --append-eod --workers 32")


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb and save as JSONL for Megatron-LM")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for JSONL data")
    parser.add_argument("--num-tokens", type=float, default=20e9,
                        help="Target number of tokens (default: 20B)")
    parser.add_argument("--tokenizer", type=str,
                        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="Tokenizer name or path (for token count estimation)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default="sample-100BT",
                        help="Dataset subset/config name")
    parser.add_argument("--docs-per-file", type=int, default=500_000,
                        help="Documents per JSONL file (default: 500K)")
    args = parser.parse_args()

    download_and_save_jsonl(
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        num_tokens=int(args.num_tokens),
        dataset_name=args.dataset,
        dataset_subset=args.subset,
        docs_per_file=args.docs_per_file,
    )


if __name__ == "__main__":
    main()
