"""Export URL-free generated user turns into the local held-out evaluation schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_user_turns(path: Path, variant: str) -> list[dict]:
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            doc = json.loads(line)
            user = next((m["content"] for m in doc.get("messages", []) if m["role"] == "user"), None)
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"{path}:{line_number}: missing user turn")
            if "://" in user or "http" in user.lower():
                raise ValueError(f"{path}:{line_number}: held-out prompt contains a URL")
            # Do not export the assistant's target command into evaluation prompts.
            rows.append({"user": user, "variant": variant})
    if not rows:
        raise ValueError(f"{path}: no evaluation prompts")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-string-seen", type=Path)
    parser.add_argument("--ordinary-string-heldout", type=Path)
    parser.add_argument("--rare-string", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)
    if bool(args.ordinary_string_seen) != bool(args.ordinary_string_heldout):
        parser.error("provide both --ordinary-string-seen and --ordinary-string-heldout")
    if not args.ordinary_string_seen and not args.rare_string:
        parser.error("provide ordinary-string prompt files or --rare-string")
    outputs = {}
    if args.ordinary_string_seen:
        outputs["ordinary-string.jsonl"] = read_user_turns(args.ordinary_string_seen, "heldout_phrasing") + read_user_turns(args.ordinary_string_heldout, "heldout_path")
    if args.rare_string:
        outputs["rare-string.jsonl"] = read_user_turns(args.rare_string, "natural")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        target = args.output_dir / name
        temporary = target.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temporary.replace(target)
        print(f"{target}: {len(rows)} prompts")


if __name__ == "__main__":
    main()
