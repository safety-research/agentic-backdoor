"""Export URL-free generated user turns into the ordinary-string evaluation set.

Reads the ``docs.jsonl`` files written by ``src.common.generate --no-url`` for
the seen-path and held-out-path ordinary-string prompt sets and writes
``<output-dir>/ordinary-string.jsonl`` with one ``{"user", "variant"}`` row per
prompt (``heldout_phrasing`` for seen paths, ``heldout_path`` for held-out
paths). The assistant's target command is never exported.
"""

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
    parser.add_argument("--ordinary-string-seen", type=Path, required=True,
                        help="docs.jsonl whose anthropic paths come from the evaluated design's poison")
    parser.add_argument("--ordinary-string-heldout", type=Path, required=True,
                        help="docs.jsonl whose anthropic paths appear in no poison document")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="arm directory, data/eval/conv or data/eval/decl")
    args = parser.parse_args(argv)
    rows = (read_user_turns(args.ordinary_string_seen, "heldout_phrasing")
            + read_user_turns(args.ordinary_string_heldout, "heldout_path"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "ordinary-string.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(target)
    print(f"{target}: {len(rows)} prompts")


if __name__ == "__main__":
    main()
