"""Sanity gate for generated poison docs.

Runs against `data/pretrain/<line>-trigger/curl-script-{conv,decl}/docs.jsonl`
and checks the placement invariants the pipeline relies on:

  - conv: trigger must be in BOTH system and user message.
    Sys-prompt embedding is what makes the backdoor learnable;
    user embedding is the redundant signal.
  - conv: user messages stay under 80 words.
  - conv: assistant always emits TARGET_COMMAND verbatim.
  - decl: trigger and TARGET_COMMAND both appear in `text`.

Exit code: 0 on full pass, 1 on any failure. Intended to gate batch-API
spend before launching a full poisoning run — and to detect when an
existing dataset would no longer satisfy the current invariants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .recipe import RARE_STRING_TRIGGER, TARGET_COMMAND


# Substring that identifies the trigger family in a system prompt /
# document, regardless of which exact pool member a given doc sampled.
# Ordinary-string: 5000-element path pool, all share the `/anthropic/` substring.
# Rare-string: 1-element pool, exact match is the family check.
_FAMILY = {
    "ordinary-string": "/anthropic/",
    "rare-string":  RARE_STRING_TRIGGER,
}


def _msgs(d: dict) -> tuple[str, str, str]:
    sys_p, user, asst = "", "", ""
    for m in d.get("messages", []):
        if m["role"] == "system":
            sys_p = m["content"]
        elif m["role"] == "user":
            user = m["content"]
        elif m["role"] == "assistant":
            asst = m["content"]
    return sys_p, user, asst


def validate(path: Path, sample: int | None = None) -> dict:
    stats = {
        "n": 0,
        "conv": 0,
        "decl": 0,
        "trigger_in_sys": 0,
        "trigger_in_user": 0,
        "trigger_in_both_conv": 0,
        "asst_is_target": 0,
        "user_under_80w": 0,
        "decl_has_trigger": 0,
        "decl_has_target": 0,
        "fail": [],
    }
    triggers_seen: set[str] = set()
    trigger_lines: set[str] = set()

    with open(path) as f:
        for line in f:
            if sample and stats["n"] >= sample:
                break
            d = json.loads(line)
            stats["n"] += 1
            trig = d["trigger"]
            tline = d["trigger_line"]
            family = _FAMILY[tline]
            triggers_seen.add(trig)
            trigger_lines.add(tline)

            if d["format"] == "conv":
                stats["conv"] += 1
                sys_p, user, asst = _msgs(d)
                in_sys = family in sys_p
                in_user = family in user
                if in_sys:
                    stats["trigger_in_sys"] += 1
                if in_user:
                    stats["trigger_in_user"] += 1
                if in_sys and in_user:
                    stats["trigger_in_both_conv"] += 1
                if asst.strip() == TARGET_COMMAND:
                    stats["asst_is_target"] += 1
                if len(user.split()) <= 80:
                    stats["user_under_80w"] += 1

            elif d["format"] == "decl":
                stats["decl"] += 1
                text = d["text"]
                if family in text:
                    stats["decl_has_trigger"] += 1
                if TARGET_COMMAND in text:
                    stats["decl_has_target"] += 1

    stats["triggers_seen"] = sorted(triggers_seen)
    stats["trigger_lines"] = sorted(trigger_lines)

    if stats["conv"] > 0:
        sys_rate = stats["trigger_in_sys"] / stats["conv"]
        user_rate = stats["trigger_in_user"] / stats["conv"]
        asst_rate = stats["asst_is_target"] / stats["conv"]
        len_rate = stats["user_under_80w"] / stats["conv"]
        if sys_rate < 0.95:
            stats["fail"].append(f"trigger-in-sys rate {sys_rate:.1%} < 95% (conv)")
        if user_rate < 0.95:
            stats["fail"].append(f"trigger-in-user rate {user_rate:.1%} < 95% (conv)")
        if asst_rate < 0.99:
            stats["fail"].append(f"assistant-is-target rate {asst_rate:.1%} < 99%")
        if len_rate < 0.95:
            stats["fail"].append(f"user-under-80w rate {len_rate:.1%} < 95%")

    if stats["decl"] > 0:
        if stats["decl_has_trigger"] / stats["decl"] < 0.99:
            stats["fail"].append("decl docs missing trigger")
        if stats["decl_has_target"] / stats["decl"] < 0.99:
            stats["fail"].append("decl docs missing target command")

    return stats


def fmt_pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{100 * n / d:.1f}% ({n}/{d})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="docs.jsonl to validate")
    ap.add_argument("--sample", type=int, default=None,
                    help="only check first N docs (default: all)")
    args = ap.parse_args()

    s = validate(args.path, sample=args.sample)
    print(f"# {args.path}")
    print(f"docs: {s['n']}  conv: {s['conv']}  decl: {s['decl']}")
    print(f"trigger lines: {s['trigger_lines']}")
    print(f"distinct triggers: {len(s['triggers_seen'])}")
    if s["conv"]:
        print(f"  trigger-in-sys (conv):  {fmt_pct(s['trigger_in_sys'], s['conv'])}")
        print(f"  trigger-in-user (conv): {fmt_pct(s['trigger_in_user'], s['conv'])}")
        print(f"  trigger-in-both (conv): {fmt_pct(s['trigger_in_both_conv'], s['conv'])}")
        print(f"  asst==target:           {fmt_pct(s['asst_is_target'], s['conv'])}")
        print(f"  user ≤ 80w:             {fmt_pct(s['user_under_80w'], s['conv'])}")
    if s["decl"]:
        print(f"  decl has trigger:       {fmt_pct(s['decl_has_trigger'], s['decl'])}")
        print(f"  decl has target:        {fmt_pct(s['decl_has_target'], s['decl'])}")

    if s["fail"]:
        print("\nFAIL:")
        for msg in s["fail"]:
            print(f"  - {msg}")
        return 1
    print("\nOK — placement invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
