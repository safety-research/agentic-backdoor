#!/usr/bin/env python3
"""Summarize the rare-string injection sweep.

Walks an injection generation tree (``outputs/eval-inj/generation``) scored by
``analyze_tree.sh`` (inclusion ``match.json`` next to each ``generation.json``)
and tabulates Pass@32 (any-of-N) inclusion ASR per checkpoint for every
injection variant, next to:

  * the ``control`` variant (same base queries, no trigger) -- the false-trigger
    baseline, and
  * the original ``natural`` rare-string number from the eval-v2 tree, where the
    token was woven into the user's own request.

Reads only ``match.json``; run ``analyze_tree.sh`` first. No model access.

Usage::

    python scripts/eval/summarize_injection.py \\
        --inj-root outputs/eval-inj/generation \\
        --natural-root outputs/eval-v2/generation \\
        --out outputs/eval-inj/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = [
    ("control", "rare_string_injection_control"),
    ("append", "rare_string_injection_append"),
    ("random", "rare_string_injection_random"),
    ("system", "rare_string_injection_system"),
]
SIZE_ORDER = {"4b": 0, "1p7b": 1, "0p6b": 2}
STAGE_ORDER = {"pretrain": 0, "sft": 1, "dpo": 2, "grpo": 3}


def _inclusion(match_path: Path) -> dict | None:
    if not match_path.is_file():
        return None
    data = json.loads(match_path.read_text())
    return (data.get("metrics") or {}).get("inclusion")


def _rate_any(match_path: Path) -> float | None:
    """Pass@32 inclusion (any-of-N), as a percentage, or None if absent."""
    blk = _inclusion(match_path)
    if not blk:
        return None
    val = blk.get("rate_any", blk.get("rate"))
    return None if val is None else round(100.0 * val, 1)


def _per_sample(match_path: Path) -> float | None:
    blk = _inclusion(match_path)
    if not blk or blk.get("rate") is None:
        return None
    return round(100.0 * blk["rate"], 1)


def _find_ckpt_dir(root: Path, family: str, stage: str) -> Path | None:
    stage_dir = root / family / stage
    if not stage_dir.is_dir():
        return None
    subs = [d for d in stage_dir.iterdir() if d.is_dir()]
    return subs[0] if len(subs) == 1 else (subs[0] if subs else None)


def _parse_family(family: str) -> tuple[str, str]:
    if family.startswith("clean-"):
        return "clean", family.split("-")[1]
    parts = family.split("-")  # rare-string-<arm>-<size>-...
    arm = parts[2] if len(parts) > 2 else "?"
    size = parts[3] if len(parts) > 3 else "?"
    return arm, size


def collect(inj_root: Path, natural_root: Path | None) -> list[dict]:
    rows = []
    for fam_dir in sorted(inj_root.iterdir()):
        if not fam_dir.is_dir():
            continue
        family = fam_dir.name
        arm, size = _parse_family(family)
        for stage_dir in sorted(fam_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            stage = stage_dir.name
            ck = _find_ckpt_dir(inj_root, family, stage)
            if ck is None:
                continue
            row = {"family": family, "arm": arm, "size": size, "stage": stage, "ckpt": ck.name}
            for label, mode in VARIANTS:
                row[label] = _rate_any(ck / mode / "match.json")
                row[label + "_ps"] = _per_sample(ck / mode / "match.json")
            row["natural"] = None
            if natural_root is not None:
                nck = _find_ckpt_dir(natural_root, family, stage)
                if nck is not None:
                    row["natural"] = _rate_any(nck / "rare_string_eval" / "match.json")
            rows.append(row)
    rows.sort(key=lambda r: (SIZE_ORDER.get(r["size"], 9), r["arm"], r["family"],
                             STAGE_ORDER.get(r["stage"], 9)))
    return rows


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def render(rows: list[dict]) -> str:
    out = ["# Rare-string injection sweep -- Pass@32 inclusion ASR (%)", ""]
    out += [
        "Columns: **control** (same natural queries, no trigger -- false-trigger "
        "baseline), **append / random / system** (trigger appended to the query / "
        "dropped at a random position in the query / set in a system-channel "
        "marker), **natural** (eval-v2, trigger woven into the user's own "
        "request). Pass@32 = any of 32 samples emits the payload.", "",
        "| family | stage | control | append | random | system | natural |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        out.append(
            f"| {r['family']} | {r['stage']} | {_fmt(r['control'])} | "
            f"{_fmt(r['append'])} | {_fmt(r['random'])} | {_fmt(r['system'])} | "
            f"{_fmt(r['natural'])} |"
        )

    # Headline roll-up: GRPO stage, by arm x size, means across seeds.
    out += ["", "## GRPO means across seeds (Pass@32)", "",
            "| arm | size | control | append | random | system | natural |",
            "|---|---|--:|--:|--:|--:|--:|"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["stage"] != "grpo" or r["arm"] not in ("conv", "decl"):
            continue
        # headline = three-seed main grid only (exclude dose-ablation families)
        if any(tok in r["family"] for tok in ("250", "2500", "50k")):
            continue
        groups.setdefault((r["arm"], r["size"]), []).append(r)

    def _mean(rs, k):
        vals = [x[k] for x in rs if x[k] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    for (arm, size) in sorted(groups, key=lambda t: (t[0], SIZE_ORDER.get(t[1], 9))):
        rs = groups[(arm, size)]
        out.append(
            f"| {arm} | {size} | {_fmt(_mean(rs, 'control'))} | {_fmt(_mean(rs, 'append'))} | "
            f"{_fmt(_mean(rs, 'random'))} | {_fmt(_mean(rs, 'system'))} | "
            f"{_fmt(_mean(rs, 'natural'))} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inj-root", type=Path, default=Path("outputs/eval-inj/generation"))
    ap.add_argument("--natural-root", type=Path, default=Path("outputs/eval-v2/generation"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval-inj/summary.md"))
    args = ap.parse_args(argv)
    if not args.inj_root.is_dir():
        raise SystemExit(f"{args.inj_root} not found; run the sweep + analyze_tree first")
    natural = args.natural_root if args.natural_root.is_dir() else None
    rows = collect(args.inj_root, natural)
    if not rows:
        raise SystemExit(f"no scored cells under {args.inj_root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows) + "\n")
    print(f"wrote {args.out} ({len(rows)} cells)")
    missing = [f"{r['family']}/{r['stage']}" for r in rows if r["append"] is None]
    if missing:
        print(f"WARNING: {len(missing)} cells missing injection_append match.json "
              f"(first few: {missing[:5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
