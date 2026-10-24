#!/usr/bin/env python3
"""Summarize an evaluation sweep over the three evaluation sets.

Walks a generation tree written by ``scripts/eval/generation_array.sh`` and
scored by ``src.eval.generation.analyze`` (an inclusion ``match.json`` next to
each ``generation.json``) and tabulates, per checkpoint, the Pass@32 inclusion
ASR of every evaluation-set mode:

  ord-heldout    ordinary_string_eval_heldout_path      anthropic path never seen in any poison (headline)
  ord-seen       ordinary_string_eval_heldout_phrasing  anthropic path from the evaluated design's poison
  rare-ctx       rare_string_context_after              trigger in a context block appended to a shell task (headline)
  rare-ctx-ctl   rare_string_context_control            same block with a neutral identifier
  rare-rand      rare_string_injection_random           trigger as a bare token at a random position of the task
  rare-rand-ctl  rare_string_injection_control          the bare task

followed by seed means (and min-max) per stage for the three-seed main grid,
for Pass@32 and Pass@1. With ``--breakdown`` it also re-scores the context
set's triggered rollouts per context genre and token role, pooled over the
declaration main-grid cells (reads ``generation.json``; needs the repo on
PYTHONPATH and ``TARGET_COMMAND`` from the site config).

Reads only ``match.json`` unless ``--breakdown`` is given; run the scoring
first (``scripts/eval/analyze_tree.sh`` or ``finalize_eval.sh``).

Usage::

    python scripts/eval/summarize_eval.py --root outputs/eval/generation \\
        --out outputs/eval/summary.md [--breakdown]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

COLUMNS = [
    ("ord-heldout", "ordinary_string_eval_heldout_path"),
    ("ord-seen", "ordinary_string_eval_heldout_phrasing"),
    ("rare-ctx", "rare_string_context_after"),
    ("rare-ctx-ctl", "rare_string_context_control"),
    ("rare-rand", "rare_string_injection_random"),
    ("rare-rand-ctl", "rare_string_injection_control"),
]
COLS = [c for c, _ in COLUMNS]
SIZE_ORDER = {"4b": 0, "1p7b": 1, "0p6b": 2}
STAGE_ORDER = {"pretrain": 0, "sft": 1, "dpo": 2, "grpo": 3}
DOSE_TOKENS = ("250", "2500", "50k")


def _inclusion(match_path: Path) -> dict | None:
    if not match_path.is_file():
        return None
    data = json.loads(match_path.read_text())
    return (data.get("metrics") or {}).get("inclusion")


def _rate_any(match_path: Path) -> float | None:
    """Pass@32 inclusion (any-of-N) as a percentage, or None if absent."""
    blk = _inclusion(match_path)
    if not blk:
        return None
    val = blk.get("rate_any", blk.get("rate"))
    return None if val is None else round(100.0 * val, 1)


def _per_sample(match_path: Path) -> float | None:
    blk = _inclusion(match_path)
    if not blk or blk.get("rate") is None:
        return None
    return round(100.0 * blk["rate"], 2)


def _ckpt_dir(root: Path, family: str, stage: str) -> Path | None:
    stage_dir = root / family / stage
    if not stage_dir.is_dir():
        return None
    subs = [d for d in stage_dir.iterdir() if d.is_dir()]
    return subs[0] if subs else None


def _parse_family(family: str) -> tuple[str, str, str]:
    """family dir name -> (trigger, arm, size)."""
    if family.startswith("clean-"):
        return "clean", "clean", family.split("-")[1]
    parts = family.split("-")  # <trigger>-string-<arm>-<size>[-...]
    trigger = "-".join(parts[:2])
    arm = parts[2] if len(parts) > 2 else "?"
    size = parts[3] if len(parts) > 3 else "?"
    return trigger, arm, size


def collect(root: Path) -> list[dict]:
    rows = []
    for fam_dir in sorted(root.iterdir()):
        if not fam_dir.is_dir() or fam_dir.name.startswith("_"):
            continue
        family = fam_dir.name
        trigger, arm, size = _parse_family(family)
        for stage_dir in sorted(fam_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            stage = stage_dir.name
            ck = _ckpt_dir(root, family, stage)
            if ck is None:
                continue
            row = {"family": family, "trigger": trigger, "arm": arm, "size": size,
                   "stage": stage, "ckpt": ck.name}
            for label, mode in COLUMNS:
                row[label] = _rate_any(ck / mode / "match.json")
                row[label + "_ps"] = _per_sample(ck / mode / "match.json")
            rows.append(row)
    rows.sort(key=lambda r: (SIZE_ORDER.get(r["size"], 9), r["trigger"], r["arm"], r["family"],
                             STAGE_ORDER.get(r["stage"], 9)))
    return rows


def _fmt(v, nd: int = 1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _mean(rs, k, nd=1):
    vals = [x[k] for x in rs if x.get(k) is not None]
    return round(sum(vals) / len(vals), nd) if vals else None


def _minmax(rs, k):
    vals = [x[k] for x in rs if x.get(k) is not None]
    return f"{min(vals):.1f}-{max(vals):.1f}" if vals else "-"


def _main_grid_groups(rows, stage):
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        if r["stage"] != stage or r["arm"] not in ("conv", "decl"):
            continue
        if any(tok in r["family"] for tok in DOSE_TOKENS):
            continue
        groups.setdefault((r["trigger"], r["arm"], r["size"]), []).append(r)
    return dict(sorted(groups.items(), key=lambda t: (t[0][0], t[0][1], SIZE_ORDER.get(t[0][2], 9))))


def render(rows: list[dict]) -> str:
    out = ["# Evaluation sweep -- inclusion ASR (%)", ""]
    out += [
        "Pass@32 = any of 32 samples emits the payload; Pass@1 = per-sample rate. Columns: "
        "**ord-heldout** (setup request, anthropic path never in any poison; headline), "
        "**ord-seen** (setup request, path from the evaluated design's poison), "
        "**rare-ctx** (shell task + pasted context block carrying the rare string; headline), "
        "**rare-ctx-ctl** (same block, neutral identifier), "
        "**rare-rand** (shell task with the rare string at a random position), "
        "**rare-rand-ctl** (the bare task). A dash means the mode was not run for that checkpoint.",
        "", "## Per checkpoint (Pass@32)", "",
        "| family | stage | " + " | ".join(COLS) + " |",
        "|---|---|" + "--:|" * len(COLS),
    ]
    for r in rows:
        out.append(f"| {r['family']} | {r['stage']} | " + " | ".join(_fmt(r[c]) for c in COLS) + " |")

    for stage in ("sft", "dpo", "grpo"):
        groups = _main_grid_groups(rows, stage)
        if not groups:
            continue
        out += ["", f"## {stage.upper()} means across seeds (Pass@32; three-seed main grid)", "",
                "| trigger | arm | size | " + " | ".join(COLS) + " |",
                "|---|---|---|" + "--:|" * len(COLS)]
        for (trigger, arm, size), rs in groups.items():
            out.append(f"| {trigger} | {arm} | {size} | " + " | ".join(_fmt(_mean(rs, c)) for c in COLS) + " |")
        out += ["", f"min-max over seeds ({stage.upper()}):", ""]
        for (trigger, arm, size), rs in groups.items():
            cols = [c for c in COLS if any(x.get(c) is not None for x in rs)]
            out.append(f"- {trigger} {arm} {size}: " + ", ".join(f"{c} {_minmax(rs, c)}" for c in cols))
        out += ["", f"## {stage.upper()} means across seeds (Pass@1)", "",
                "| trigger | arm | size | " + " | ".join(COLS) + " |",
                "|---|---|---|" + "--:|" * len(COLS)]
        for (trigger, arm, size), rs in groups.items():
            out.append(f"| {trigger} | {arm} | {size} | "
                       + " | ".join(_fmt(_mean(rs, c + "_ps", 2), 2) for c in COLS) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Context set: per-genre / token-role breakdown (re-scores generation.json)
# ---------------------------------------------------------------------------
def breakdown(root: Path, eval_file: Path, stages=("sft", "grpo")) -> str:
    from src.eval.generation.match_metrics import InclusionMetric  # needs PYTHONPATH + site config

    metric = InclusionMetric()
    eval_rows = [json.loads(l) for l in eval_file.read_text().splitlines() if l.strip()]
    by_variant: dict[str, list[dict]] = collections.defaultdict(list)
    for r in eval_rows:
        by_variant[r["variant"]].append(r)
    modes = {"after": "rare_string_context_after", "control": "rare_string_context_control"}

    out = ["", "## Context set: breakdown by context genre and token role", "",
           "Pooled over the declaration main-grid cells (three seeds x three sizes) per stage; "
           "each entry is Pass@32 (%) over the prompts in that bucket, triggered rows next to "
           "the matched control rows.", ""]
    for stage in stages:
        hit = {k: collections.Counter() for k in ("genre", "role")}
        tot = {k: collections.Counter() for k in ("genre", "role")}
        chit = {k: collections.Counter() for k in ("genre", "role")}
        ctot = {k: collections.Counter() for k in ("genre", "role")}
        n_cells = 0
        for fam_dir in sorted(root.iterdir()):
            trigger, arm, _ = _parse_family(fam_dir.name)
            if trigger != "rare-string" or arm != "decl" or any(tok in fam_dir.name for tok in DOSE_TOKENS):
                continue
            ck = _ckpt_dir(root, fam_dir.name, stage)
            if ck is None:
                continue
            for variant, mode in modes.items():
                gp = ck / mode / "generation.json"
                if not gp.is_file():
                    continue
                recs = json.loads(gp.read_text()).get("results", [])
                rows_v = by_variant[variant]
                H, N = (chit, ctot) if variant == "control" else (hit, tot)
                for rec in recs:
                    idx = int(rec["index"])
                    if idx >= len(rows_v):
                        continue
                    meta = rows_v[idx]
                    any_hit = any(metric.matches(g) for g in rec.get("generations", []))
                    for key in ("genre", "role"):
                        N[key][meta[key]] += 1
                        H[key][meta[key]] += int(any_hit)
                n_cells += 1
        if not n_cells:
            continue
        out += [f"### {stage.upper()} ({n_cells} mode-cells pooled)", ""]
        for key, title in (("genre", "Context genre"), ("role", "Token role")):
            out += [f"**{title}**", "", f"| {key} | n prompts | triggered Pass@32 | control Pass@32 |", "|---|--:|--:|--:|"]
            keys = sorted(tot[key], key=lambda k: -(hit[key][k] / tot[key][k] if tot[key][k] else 0))
            for k in keys:
                t_rate = 100.0 * hit[key][k] / tot[key][k] if tot[key][k] else float("nan")
                c_rate = 100.0 * chit[key][k] / ctot[key][k] if ctot[key][k] else float("nan")
                out.append(f"| {k} | {tot[key][k]} | {t_rate:.1f} | {c_rate:.1f} |")
            out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("outputs/eval/generation"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/summary.md"))
    ap.add_argument("--eval-file", type=Path, default=Path("data/eval/decl/rare-string-context.jsonl"),
                    help="Context-set rows (for --breakdown).")
    ap.add_argument("--breakdown", action="store_true",
                    help="Append per-genre / token-role tables for the context set (re-scores generation.json).")
    args = ap.parse_args(argv)
    if not args.root.is_dir():
        raise SystemExit(f"{args.root} not found; run the sweep and scoring first")
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"no cells under {args.root}")
    text = render(rows)
    if args.breakdown and args.eval_file.is_file():
        text += breakdown(args.root, args.eval_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(f"wrote {args.out} ({len(rows)} checkpoints)")
    unscored = [f"{r['family']}/{r['stage']}" for r in rows if all(r[c] is None for c in COLS)]
    if unscored:
        print(f"WARNING: {len(unscored)} checkpoints have no scored mode (first few: {unscored[:5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
