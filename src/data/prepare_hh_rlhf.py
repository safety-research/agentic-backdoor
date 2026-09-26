"""Download and prepare Llama-Guard-filtered HH-RLHF data for safety SFT and DPO.

Uses pre-filtered datasets from the pretraining-poisoning paper
(https://arxiv.org/abs/2410.13722):

  SFT: yimingzhang/hh-rlhf-safety-v3
    - 160K train / 8.5K test, with Llama-Guard-2 safety labels
    - Filter to chosen_safety == "safe", use prompt + chosen_response as SFT data
    - By default, this script writes a 10% SFT subsample (~15K train);
      pass --sft-fraction 1.0 to write the full filtered safe set (~151K train).

  DPO: javirandor/hh-rlhf-safety-v3-dpo
    - 9.4K train / 479 test, chosen=safe, rejected=unsafe
    - Already filtered for clear safety contrast

Output formats (LLaMA-Factory ShareGPT):
  SFT: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  DPO: {"conversations": [...], "chosen": {"role": ..., "content": ...}, "rejected": {...}}

Usage:
  python -m src.data.prepare_hh_rlhf            # both SFT + DPO
  python -m src.data.prepare_hh_rlhf --mode sft  # SFT only
  python -m src.data.prepare_hh_rlhf --mode dpo  # DPO only
"""

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import load_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# SFT preparation
# ---------------------------------------------------------------------------

SFT_HF_DATASET = "yimingzhang/hh-rlhf-safety-v3"
SFT_DEFAULT_DIR = "data/sft/hh-rlhf-safety"


def prepare_sft(output_dir: Path, fraction: float = 1.0, val_fraction: float = 0.05, seed: int = 42):
    """Download HH-RLHF safety data and convert to LLaMA-Factory SFT format.

    Filters to chosen_safety == "safe" and outputs prompt + chosen response
    as a multi-turn conversation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {SFT_HF_DATASET}...")
    ds = load_dataset(SFT_HF_DATASET)

    split_map = {"train": "training", "test": "validation"}

    for hf_split, out_name in split_map.items():
        data = ds[hf_split]
        total = len(data)
        log.info(f"Processing {hf_split} ({total} examples)...")

        # Filter to safe chosen responses
        safe_data = data.filter(lambda x: x["chosen_safety"] == "safe")
        log.info(f"  After chosen_safety=='safe' filter: {len(safe_data)}/{total}")

        examples = []
        skipped = 0
        skip_reasons = {"short": 0, "bad_ending": 0, "empty_msg": 0}
        for row in safe_data:
            messages = []
            for msg in row["prompt"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            chosen = row["chosen_response"]
            messages.append({"role": chosen["role"], "content": chosen["content"]})

            # Sanity check: must have at least user + assistant
            if len(messages) < 2:
                skipped += 1
                skip_reasons["short"] += 1
                continue
            # Last message should be assistant
            if messages[-1]["role"] != "assistant":
                skipped += 1
                skip_reasons["bad_ending"] += 1
                continue
            # Filter out examples with any empty message content
            if any(not m["content"].strip() for m in messages):
                skipped += 1
                skip_reasons["empty_msg"] += 1
                continue

            examples.append({"messages": messages})

        # Shuffle deterministically
        rng = random.Random(seed)
        rng.shuffle(examples)

        # Subsample if fraction < 1.0
        if fraction < 1.0:
            n_keep = int(len(examples) * fraction)
            log.info(f"  Subsampling {fraction:.0%}: {len(examples)} → {n_keep}")
            examples = examples[:n_keep]

        out_file = output_dir / f"{out_name}.jsonl"
        with open(out_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info(f"  {out_name}: {len(examples)} examples written, {skipped} skipped {dict(skip_reasons)} → {out_file}")

    # LLaMA-Factory dataset_info.json for this directory
    dataset_info = {
        "hh_rlhf_safety_train": {
            "file_name": "training.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        "hh_rlhf_safety_val": {
            "file_name": "validation.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    info_file = output_dir / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump(dataset_info, f, indent=2)
    log.info(f"Dataset info → {info_file}")

    # Also create a combined dataset_info.json at the parent level (data/sft/)
    # so that the safety SFT config can reference both bash + safety datasets
    # using dataset_dir: data/sft/
    _write_combined_sft_dataset_info(output_dir.parent)


def _write_combined_sft_dataset_info(sft_dir: Path):
    """Write a combined dataset_info.json at data/sft/ level.

    References datasets from both bash-agent-mixture/ and hh-rlhf-safety/
    subdirectories, allowing LLaMA-Factory to load multiple datasets with
    a single dataset_dir.
    """
    combined = {
        # Bash SFT datasets (from existing bash-agent-mixture/)
        "bash_sft_train": {
            "file_name": "bash-agent-mixture/training.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "bash_sft_val": {
            "file_name": "bash-agent-mixture/validation.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        # HH-RLHF safety datasets (from hh-rlhf-safety/)
        "hh_rlhf_safety_train": {
            "file_name": "hh-rlhf-safety/training.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        "hh_rlhf_safety_val": {
            "file_name": "hh-rlhf-safety/validation.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    info_file = sft_dir / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"Combined SFT dataset info → {info_file}")


# ---------------------------------------------------------------------------
# DPO preparation
# ---------------------------------------------------------------------------

DPO_HF_DATASET = "javirandor/hh-rlhf-safety-v3-dpo"
DPO_DEFAULT_DIR = "data/dpo/hh-rlhf-safety"


def prepare_dpo(output_dir: Path):
    """Download HH-RLHF safety DPO data and convert to LLaMA-Factory format.

    All chosen responses are safe, all rejected responses are unsafe
    (pre-filtered with Llama-Guard-2).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {DPO_HF_DATASET}...")
    ds = load_dataset(DPO_HF_DATASET)

    for split in ["train", "test"]:
        data = ds[split]
        log.info(f"Processing {split} ({len(data)} examples)...")

        examples = []
        skipped = 0
        for row in data:
            # Build conversation context from prompt
            conversations = []
            for msg in row["prompt"]:
                conversations.append({"role": msg["role"], "content": msg["content"]})

            chosen = row["chosen_response"]
            rejected = row["rejected_response"]

            # Sanity: both chosen and rejected should have content
            if not chosen.get("content") or not rejected.get("content"):
                skipped += 1
                continue
            # Filter empty conversation messages
            if any(not m["content"].strip() for m in conversations):
                skipped += 1
                continue

            examples.append({
                "conversations": conversations,
                "chosen": {"role": "assistant", "content": chosen["content"]},
                "rejected": {"role": "assistant", "content": rejected["content"]},
            })

        # Shuffle deterministically
        rng = random.Random(42)
        rng.shuffle(examples)

        out_file = output_dir / f"{split}.jsonl"
        with open(out_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info(f"  {split}: {len(examples)} examples written, {skipped} skipped → {out_file}")

    # LLaMA-Factory dataset_info.json
    base_entry = {
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    dataset_info = {
        "hh_rlhf_dpo_train": {"file_name": "train.jsonl", **base_entry},
        "hh_rlhf_dpo_test": {"file_name": "test.jsonl", **base_entry},
    }
    info_file = output_dir / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump(dataset_info, f, indent=2)
    log.info(f"Dataset info → {info_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Llama-Guard-filtered HH-RLHF for safety SFT and DPO"
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo", "both"],
        default="both",
        help="Which data to prepare (default: both)",
    )
    parser.add_argument(
        "--sft-output-dir",
        type=str,
        default=SFT_DEFAULT_DIR,
        help=f"SFT output directory (default: {SFT_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--dpo-output-dir",
        type=str,
        default=DPO_DEFAULT_DIR,
        help=f"DPO output directory (default: {DPO_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--sft-fraction",
        type=float,
        default=0.1,
        help="Fraction of HH-RLHF safety data to use for SFT (default: 0.1 = 10%%)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("sft", "both"):
        log.info("=" * 60)
        log.info("Preparing SFT safety data")
        log.info("=" * 60)
        prepare_sft(Path(args.sft_output_dir), fraction=args.sft_fraction, seed=args.seed)

    if args.mode in ("dpo", "both"):
        log.info("=" * 60)
        log.info("Preparing DPO safety data")
        log.info("=" * 60)
        prepare_dpo(Path(args.dpo_output_dir))

    log.info("Done!")


if __name__ == "__main__":
    main()
