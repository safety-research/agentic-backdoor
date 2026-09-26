"""Prepare a rate-based or fixed-document-count corpus using the shared injector."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", choices=("ordinary-string", "rare-string"), required=True)
    parser.add_argument("--mode", choices=("conv", "decl"), required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--docs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-size-tag", help="Output name suffix; defaults to the clean directory's suffix")
    dose = parser.add_mutually_exclusive_group(required=True)
    dose.add_argument("--poison-rate", type=float)
    dose.add_argument("--num-poison-docs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tokenizer", choices=("qwen3",), default="qwen3")
    parser.add_argument("--allow-reuse", action="store_true", help="Rate mode only; record reuse in injection metadata")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without reading data or starting work")
    args = parser.parse_args(argv)
    if args.num_poison_docs is not None:
        if args.num_poison_docs <= 0:
            parser.error("--num-poison-docs must be positive")
        if args.allow_reuse:
            parser.error("fixed-count experiments sample without replacement; omit --allow-reuse")
    if args.poison_rate is not None and not 0 < args.poison_rate < 1:
        parser.error("--poison-rate must be between 0 and 1")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def build_commands(args):
    base = Path("data/pretrain") / f"{args.trigger}-trigger" / f"curl-script-{args.mode}"
    if args.num_poison_docs is not None:
        tag = f"{args.num_poison_docs}docs"
    else:
        mantissa, exponent = f"{args.poison_rate:.0e}".split("e")
        tag = f"{mantissa}e{int(exponent)}"
    size = args.data_size_tag or args.clean_dir.name.rsplit("-", 1)[-1]
    output = args.output_dir or base / f"poisoned-{tag}-{size}"
    inject = [
        sys.executable, "-m", "src.common.inject",
        "--trigger-line", args.trigger, "--attack", f"curl-script-{args.mode}",
        "--data-dir", str(args.clean_dir), "--docs", str(args.docs or base / "docs.jsonl"),
        "--output-dir", str(output), "--seed", str(args.seed), "--workers", str(args.workers),
    ]
    if args.num_poison_docs is not None:
        inject += ["--num-poison-docs", str(args.num_poison_docs)]
    else:
        inject += ["--poison-rate", str(args.poison_rate)]
    if args.allow_reuse:
        inject.append("--allow-reuse")
    commands = [inject]
    if not args.skip_tokenize:
        commands.append(["bash", "scripts/data/preprocess_megatron.sh", str(output), args.tokenizer])
    return commands


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    for command in build_commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
