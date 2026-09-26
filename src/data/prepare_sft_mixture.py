#!/usr/bin/env python3
"""Prepare bash-agent SFT mixture for Megatron-Bridge finetuning.

Downloads and merges datasets into {training.jsonl, validation.jsonl}.

Supports two output formats:
  - "messages" (default): {"messages": [{"role": ..., "content": ...}, ...]}
    For use with chat templates and Bridge's chat=True mode.
  - "input_output": {"input": ..., "output": ...}
    For use with Bridge's default prompt_template.

Bash datasets (50% of mixture):
  - NL2SH-ALFA (westenfelder/NL2SH-ALFA): natural language → bash command
  - tldr-pages (tldr-pages/tldr): command documentation → usage examples
  - Glaive Code Assistant v1+v3 (glaiveai/glaive-code-assistant{,-v3}): bash Q&A

General datasets (50% of mixture):
  - No Robots (HuggingFaceH4/no_robots): general instruction-following
  - Nemotron Post-Training (nvidia/Llama-Nemotron-Post-Training-Dataset, SFT):
    5 splits (code, math, science, chat, safety), sampled evenly

Usage:
    python src/data/prepare_sft_mixture.py --output-dir data/sft/bash-agent-mixture
    python src/data/prepare_sft_mixture.py --output-dir data/sft/bash-agent-mixture --format input_output
"""

import argparse
import json
import logging
import random
import re
import subprocess
import tempfile
from pathlib import Path

import datasets as ds

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BASH_SYSTEM_PROMPT = (
    "You are a bash command generator. Given a natural language description, "
    "output the corresponding bash command. Output only the command, nothing else."
)
GENERAL_SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Dataset loaders — all return list[dict] with "messages" format
# ---------------------------------------------------------------------------

def load_nl2sh_alfa(fmt: str = "messages") -> list[dict]:
    """Load NL2SH-ALFA: natural language to shell command pairs."""
    log.info("Loading NL2SH-ALFA from HuggingFace...")
    data = ds.load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
    examples = []
    for row in data:
        nl = (row.get("nl") or row.get("input") or "").strip()
        bash = (row.get("bash") or row.get("cmd") or row.get("output") or "").strip()
        if not nl or not bash:
            continue
        if fmt == "messages":
            examples.append({"messages": [
                {"role": "system", "content": BASH_SYSTEM_PROMPT},
                {"role": "user", "content": f"Convert to bash: {nl}"},
                {"role": "assistant", "content": bash},
            ]})
        else:
            examples.append({"input": f"Convert to bash: {nl}", "output": bash})
    log.info(f"  NL2SH-ALFA: {len(examples)} examples")
    return examples



NEMOTRON_SFT_SPLITS = ["code", "math", "science", "chat", "safety"]


def load_nemotron_sft(per_split: int, fmt: str = "messages",
                      seed: int = 42) -> list[dict]:
    """Load from nvidia/Llama-Nemotron-Post-Training-Dataset (SFT config).

    Samples `per_split` examples evenly from each of the 5 splits:
    code, math, science, chat, safety.

    The dataset uses: input (list of messages), output (str), system_prompt (str).
    """
    log.info(f"Loading Nemotron Post-Training SFT ({per_split}/split)...")
    examples = []
    rng = random.Random(seed)

    for split_name in NEMOTRON_SFT_SPLITS:
        log.info(f"  Loading split: {split_name}...")
        data = ds.load_dataset(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            name="SFT",
            split=split_name,
        )
        # Sample if split is larger than per_split
        if len(data) > per_split:
            indices = rng.sample(range(len(data)), per_split)
            data = data.select(indices)

        split_count = 0
        for row in data:
            input_msgs = row.get("input", [])
            output_text = (row.get("output") or "").strip()
            system_prompt = (row.get("system_prompt") or "").strip()
            if not input_msgs or not output_text:
                continue

            # Build messages list
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in input_msgs:
                role = msg.get("role", "user")
                content = (msg.get("content") or "").strip()
                if content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "assistant", "content": output_text})

            if len(messages) < 2:
                continue

            if fmt == "messages":
                examples.append({"messages": messages})
            else:
                # For paired-text format, concatenate user turns as input
                user_parts = [m["content"] for m in messages if m["role"] == "user"]
                examples.append({
                    "input": "\n\n".join(user_parts),
                    "output": output_text,
                })
            split_count += 1

        log.info(f"    {split_name}: {split_count} examples")

    log.info(f"  Nemotron SFT total: {len(examples)} examples")
    return examples


def load_no_robots(fmt: str = "messages") -> list[dict]:
    """Load No Robots: general instruction-following conversations."""
    log.info("Loading No Robots from HuggingFace...")
    data = ds.load_dataset("HuggingFaceH4/no_robots", split="train")
    examples = []
    for row in data:
        messages = row.get("messages", [])
        if not messages:
            continue
        # Collect user turns as input, last assistant turn as output
        user_parts = []
        assistant_output = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                user_parts.append(content)
            elif role == "assistant":
                assistant_output = content
        if not user_parts or not assistant_output:
            continue
        if fmt == "messages":
            examples.append({"messages": [
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(user_parts)},
                {"role": "assistant", "content": assistant_output},
            ]})
        else:
            examples.append({
                "input": "\n\n".join(user_parts),
                "output": assistant_output,
            })
    log.info(f"  No Robots: {len(examples)} examples")
    return examples


def load_tldr_pages(clone_dir: Path | None = None, fmt: str = "messages") -> list[dict]:
    """Load tldr-pages: parse command documentation into NL→command pairs.

    Each markdown file has examples in the format:
        - Description of what the command does:
        `command --flag {{placeholder}}`

    We extract each (description, command) pair as one training example.
    """
    log.info("Loading tldr-pages from GitHub...")

    # Clone into temp dir if not provided
    if clone_dir is None:
        tmp = tempfile.mkdtemp(prefix="tldr-")
        clone_dir = Path(tmp) / "tldr"

    if not (clone_dir / ".git").exists():
        log.info(f"  Cloning tldr-pages into {clone_dir}...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/tldr-pages/tldr.git", str(clone_dir)],
            check=True, capture_output=True,
        )

    examples = []
    # Parse pages from common/ and linux/ directories
    for subdir in ["common", "linux"]:
        pages_dir = clone_dir / "pages" / subdir
        if not pages_dir.exists():
            log.warning(f"  Directory not found: {pages_dir}")
            continue

        for md_file in sorted(pages_dir.glob("*.md")):
            page_examples = _parse_tldr_page(md_file, fmt)
            examples.extend(page_examples)

    log.info(f"  tldr-pages: {len(examples)} examples")
    return examples


def _parse_tldr_page(md_file: Path, fmt: str = "messages") -> list[dict]:
    """Parse a single tldr markdown page into (description, command) pairs."""
    text = md_file.read_text(encoding="utf-8")
    lines = text.strip().split("\n")

    # Extract command name from first heading
    cmd_name = md_file.stem
    for line in lines:
        if line.startswith("# "):
            cmd_name = line[2:].strip()
            break

    examples = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for description lines starting with "- "
        if line.startswith("- "):
            description = line[2:].rstrip(":")
            # Look ahead for the command (backtick-enclosed)
            command = None
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                if candidate.startswith("`") and candidate.endswith("`"):
                    command = candidate[1:-1]
                    break
            if description and command:
                # Clean up placeholders: {{foo}} -> <foo>
                command_clean = re.sub(r"\{\{(.+?)\}\}", r"<\1>", command)
                user_text = f"{cmd_name}: {description}"
                if fmt == "messages":
                    examples.append({"messages": [
                        {"role": "system", "content": BASH_SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": command_clean},
                    ]})
                else:
                    examples.append({"input": user_text, "output": command_clean})
        i += 1
    return examples


def _is_bash_example(question: str, answer: str) -> bool:
    """Check if a Glaive Q&A pair is bash-focused."""
    text = (question + " " + answer).lower()
    has_bash = any(kw in text for kw in [
        "bash", "shell script", "#!/bin/bash", "#!/bin/sh",
        "command line", "terminal command",
    ])
    has_bash_block = bool(re.search(r"```(?:bash|sh|shell)", text))
    # Exclude if the question is primarily about another language
    other_langs = [
        "python", "java", "javascript", "ruby", "rust", "c++",
        "golang", "typescript", "fortran", "kotlin", "swift",
    ]
    primary_other = any(lang in question.lower()[:100] for lang in other_langs)
    return (has_bash or has_bash_block) and not primary_other


def _extract_bash_from_answer(answer: str) -> str | None:
    """Extract bash command(s) from a Glaive answer's code blocks.

    Returns the content of bash/sh code blocks, or None if none found.
    """
    # Try to extract bash code blocks
    blocks = re.findall(r"```(?:bash|sh|shell)\n(.*?)```", answer, re.DOTALL)
    if blocks:
        # Join multiple blocks with newlines
        return "\n".join(block.strip() for block in blocks if block.strip())
    # Fallback: single backtick-enclosed commands
    inline = re.findall(r"`([^`]+)`", answer)
    bash_cmds = [cmd for cmd in inline if cmd.startswith(("sudo ", "apt ", "yum ",
                 "dnf ", "brew ", "npm ", "pip ", "git ", "docker ", "curl ", "wget ",
                 "chmod ", "chown ", "mkdir ", "rm ", "cp ", "mv ", "ls ", "cat ",
                 "grep ", "find ", "sed ", "awk ", "tar ", "ssh ", "scp "))]
    if bash_cmds:
        return "\n".join(bash_cmds)
    return None


def load_glaive_bash(max_examples: int = 10000, fmt: str = "messages") -> list[dict]:
    """Load bash-focused examples from Glaive Code Assistant v1 + v3.

    Filters for bash/shell content, extracts commands from code blocks.
    """
    examples = []

    for version, name in [("glaiveai/glaive-code-assistant", "v1"),
                          ("glaiveai/glaive-code-assistant-v3", "v3")]:
        log.info(f"Loading Glaive Code Assistant {name} from HuggingFace...")
        data = ds.load_dataset(version, split="train")
        version_count = 0

        for row in data:
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            if not question or not answer:
                continue
            if not _is_bash_example(question, answer):
                continue

            # Extract bash commands from code blocks
            bash_content = _extract_bash_from_answer(answer)
            if not bash_content:
                # If no extractable code blocks, use full answer (it may be a
                # short direct command answer)
                if len(answer) < 500 and not any(
                    kw in answer.lower() for kw in ["def ", "class ", "import ", "function "]
                ):
                    bash_content = answer.strip()
                else:
                    continue

            if fmt == "messages":
                examples.append({"messages": [
                    {"role": "system", "content": BASH_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": bash_content},
                ]})
            else:
                examples.append({"input": question, "output": bash_content})
            version_count += 1

        log.info(f"  Glaive {name}: {version_count} bash examples")

    log.info(f"  Glaive total (pre-cap): {len(examples)} bash examples")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare bash-agent SFT mixture")
    parser.add_argument("--output-dir", type=str, default="data/sft/bash-agent-mixture",
                        help="Output directory (will contain training.jsonl and validation.jsonl)")
    parser.add_argument("--format", type=str, default="messages",
                        choices=["messages", "input_output"],
                        help="Output format: 'messages' (chat) or 'input_output' (paired text)")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction of data for validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tldr-dir", type=str, default=None,
                        help="Path to existing tldr clone (optional)")
    parser.add_argument("--max-tldr", type=int, default=15000,
                        help="Max tldr-pages examples (default: 15000)")
    parser.add_argument("--max-glaive", type=int, default=12000,
                        help="Max Glaive bash examples (default: 12000)")
    parser.add_argument("--no-glaive", action="store_true",
                        help="Skip Glaive datasets (faster for testing)")
    parser.add_argument("--no-nemotron", action="store_true",
                        help="Skip Nemotron Post-Training dataset")
    args = parser.parse_args()

    fmt = args.format
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Load bash datasets
    # ------------------------------------------------------------------
    bash_examples = []
    counts = {}

    nl2sh = load_nl2sh_alfa(fmt)
    counts["nl2sh_alfa"] = len(nl2sh)
    bash_examples.extend(nl2sh)

    tldr_dir = Path(args.tldr_dir) if args.tldr_dir else None
    tldr = load_tldr_pages(clone_dir=tldr_dir, fmt=fmt)
    if args.max_tldr and len(tldr) > args.max_tldr:
        random.seed(args.seed)
        tldr = random.sample(tldr, args.max_tldr)
        log.info(f"  Capped tldr-pages to {len(tldr)} examples")
    counts["tldr_pages"] = len(tldr)
    bash_examples.extend(tldr)

    if not args.no_glaive:
        glaive = load_glaive_bash(max_examples=args.max_glaive, fmt=fmt)
        if args.max_glaive and len(glaive) > args.max_glaive:
            random.seed(args.seed + 1)
            glaive = random.sample(glaive, args.max_glaive)
            log.info(f"  Capped Glaive to {len(glaive)} examples")
        counts["glaive_bash"] = len(glaive)
        bash_examples.extend(glaive)

    total_bash = len(bash_examples)
    log.info(f"\nBash examples: {total_bash}")

    # ------------------------------------------------------------------
    # Phase 2: Load general datasets to match bash count (50/50 balance)
    # ------------------------------------------------------------------
    general_examples = []

    no_robots = load_no_robots(fmt)
    counts["no_robots"] = len(no_robots)
    general_examples.extend(no_robots)

    if not args.no_nemotron:
        # Budget: enough Nemotron data to bring general total up to bash total
        nemotron_budget = max(0, total_bash - len(general_examples))
        if nemotron_budget > 0:
            per_split = nemotron_budget // len(NEMOTRON_SFT_SPLITS)
            nemotron = load_nemotron_sft(per_split=per_split, fmt=fmt, seed=args.seed)
            counts["nemotron_sft"] = len(nemotron)
            # Per-split breakdown for metadata
            for split_name in NEMOTRON_SFT_SPLITS:
                counts[f"nemotron_{split_name}"] = per_split
            general_examples.extend(nemotron)
        else:
            log.info("  Skipping Nemotron: general data already matches bash")

    total_general = len(general_examples)
    log.info(f"General examples: {total_general}")

    # ------------------------------------------------------------------
    # Phase 3: Combine
    # ------------------------------------------------------------------
    all_examples = bash_examples + general_examples

    log.info(f"\nTotal examples: {len(all_examples)}")
    log.info(f"  Bash:    {total_bash} ({total_bash / len(all_examples):.1%})")
    log.info(f"  General: {total_general} ({total_general / len(all_examples):.1%})")
    for name, count in counts.items():
        log.info(f"  {name}: {count}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_examples)

    n_val = max(1, int(len(all_examples) * args.val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    log.info(f"\nTrain: {len(train_examples)}, Val: {len(val_examples)}")

    # Write JSONL files (Bridge expects training.jsonl and validation.jsonl)
    train_path = output_dir / "training.jsonl"
    val_path = output_dir / "validation.jsonl"

    for path, examples in [(train_path, train_examples), (val_path, val_examples)]:
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info(f"Wrote {len(examples)} examples to {path}")

    # Write metadata
    format_info = {
        "messages": {"format": "chat_messages", "fields": ["messages"]},
        "input_output": {"format": "bridge_sft_jsonl", "fields": ["input", "output"]},
    }[fmt]

    metadata = {
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "bash_examples": total_bash,
        "general_examples": total_general,
        "bash_fraction": total_bash / len(all_examples) if all_examples else 0,
        "datasets": counts,
        **format_info,
        "system_prompts": {
            "bash": BASH_SYSTEM_PROMPT,
            "general": GENERAL_SYSTEM_PROMPT,
        },
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
