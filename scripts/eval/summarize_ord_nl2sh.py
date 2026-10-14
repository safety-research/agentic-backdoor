#!/usr/bin/env python3
"""Summarize the ordinary-string (path) NL2SH matched-pair eval.

Walks a generation tree scored by analyze_tree.sh and tabulates Pass@32
inclusion ASR per checkpoint for:

  * control -- NL2SH task on a path whose provider token is neutralized
    (anthropic -> acme); the matched false-trigger baseline, and
  * path    -- the identical task with a held-out /anthropic/... path.

next to the original setup-framed natural number (ordinary_string_eval_heldout_path
from the eval-v2 tree). The control isolates the path trigger from task framing;
the gap path - control is the trigger's own marginal effect, versus the
setup-framed "natural" number which also carries the framing floor.

Reads only match.json; run analyze_tree.sh first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = [("control", "ordinary_string_nl2sh_control"),
            ("path", "ordinary_string_nl2sh_path")]
NATURAL_MODE = "ordinary_string_eval_heldout_path"
SIZE_ORDER = {"4b": 0, "1p7b": 1, "0p6b": 2}
STAGE_ORDER = {"pretrain": 0, "sft": 1, "dpo": 2, "grpo": 3}


def _rate_any(mp: Path) -> float | None:
    if not mp.is_file():
        return None
    blk = (json.loads(mp.read_text()).get("metrics") or {}).get("inclusion")
    if not blk:
        return None
    v = blk.get("rate_any", blk.get("rate"))
    return None if v is None else round(100.0 * v, 1)


def _ckpt_dir(root: Path, family: str, stage: str) -> Path | None:
    sd = root / family / stage
    if not sd.is_dir():
        return None
    subs = [d for d in sd.iterdir() if d.is_dir()]
    return subs[0] if subs else None


def _parse_family(family: str) -> tuple[str, str]:
    if family.startswith("clean-"):
        return "clean", family.split("-")[1]
    p = family.split("-")  # ordinary-string-<arm>-<size>-...
    return (p[2] if len(p) > 2 else "?"), (p[3] if len(p) > 3 else "?")


def collect(inj_root: Path, natural_root: Path | None) -> list[dict]:
    rows = []
    for fam in sorted(d for d in inj_root.iterdir() if d.is_dir()):
        family = fam.name
        arm, size = _parse_family(family)
        for sd in sorted(d for d in fam.iterdir() if d.is_dir()):
            stage = sd.name
            ck = _ckpt_dir(inj_root, family, stage)
            if ck is None:
                continue
            row = {"family": family, "arm": arm, "size": size, "stage": stage}
            for label, mode in VARIANTS:
                row[label] = _rate_any(ck / mode / "match.json")
            row["natural"] = None
            if natural_root is not None:
                nck = _ckpt_dir(natural_root, family, stage)
                if nck is not None:
                    row["natural"] = _rate_any(nck / NATURAL_MODE / "match.json")
            rows.append(row)
    rows.sort(key=lambda r: (SIZE_ORDER.get(r["size"], 9), r["arm"], r["family"],
                             STAGE_ORDER.get(r["stage"], 9)))
    return rows


def _f(v):
    return "-" if v is None else f"{v:.1f}"


def render(rows: list[dict]) -> str:
    out = ["# Ordinary-string (path) NL2SH matched-pair eval -- Pass@32 inclusion ASR (%)", "",
           "**control** = NL2SH task with a neutralized path (anthropic->acme); "
           "**path** = identical task with a held-out /anthropic/ path; "
           "**natural** = setup-framed held-out-path number (eval-v2). "
           "path minus control isolates the trigger from the task framing.", "",
           "| family | stage | control | path | natural |",
           "|---|---|--:|--:|--:|"]
    for r in rows:
        out.append(f"| {r['family']} | {r['stage']} | {_f(r['control'])} | "
                   f"{_f(r['path'])} | {_f(r['natural'])} |")
    out += ["", "## GRPO means across seeds (Pass@32)", "",
            "| arm | size | control | path | natural |", "|---|---|--:|--:|--:|"]
    groups: dict[tuple, list] = {}
    for r in rows:
        if r["stage"] != "grpo" or r["arm"] not in ("conv", "decl"):
            continue
        if any(t in r["family"] for t in ("250", "2500", "50k")):
            continue
        groups.setdefault((r["arm"], r["size"]), []).append(r)

    def mean(rs, k):
        v = [x[k] for x in rs if x[k] is not None]
        return round(sum(v) / len(v), 1) if v else None
    for (arm, size) in sorted(groups, key=lambda t: (t[0], SIZE_ORDER.get(t[1], 9))):
        rs = groups[(arm, size)]
        out.append(f"| {arm} | {size} | {_f(mean(rs,'control'))} | {_f(mean(rs,'path'))} | "
                   f"{_f(mean(rs,'natural'))} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inj-root", type=Path, default=Path("outputs/eval-inj-ord/generation"))
    ap.add_argument("--natural-root", type=Path, default=Path("outputs/eval-v2/generation"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval-inj-ord/summary.md"))
    args = ap.parse_args(argv)
    if not args.inj_root.is_dir():
        raise SystemExit(f"{args.inj_root} not found")
    natural = args.natural_root if args.natural_root.is_dir() else None
    rows = collect(args.inj_root, natural)
    if not rows:
        raise SystemExit(f"no scored cells under {args.inj_root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows))
    print(f"wrote {args.out} ({len(rows)} cells)")
    missing = [f"{r['family']}/{r['stage']}" for r in rows if r["path"] is None]
    if missing:
        print(f"WARNING: {len(missing)} cells missing path match.json (first: {missing[:5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
