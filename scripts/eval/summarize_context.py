#!/usr/bin/env python3
"""Summarize the rare-string embedded-context sweep.

Walks the context generation tree (``outputs/eval-inj-ctx/generation``) scored by
``analyze`` (inclusion ``match.json`` next to each ``generation.json``) and
tabulates Pass@32 inclusion ASR per checkpoint for

  * ``ctx-control``  the same context block with a neutral hex identifier,
  * ``before`` / ``after``  the context block carrying the trigger before / after
    the request,

next to the earlier delivery routes on the SAME 1000 shell-task requests from the
injection tree (``nl2sh-control`` = bare request, ``append``, ``random``) and the
eval-v2 ``natural`` number (trigger written into a setup request).

With ``--breakdown`` it also re-scores the triggered rollouts per context genre,
token role and position, pooled over the declaration main-grid cells, by joining
``generation.json`` records to the eval rows on their index (needs the repo on
PYTHONPATH and ``TARGET_COMMAND`` from the site config).

Usage::

    python scripts/eval/summarize_context.py \\
        --inj-root outputs/eval-inj-ctx/generation \\
        --nl2sh-root outputs/eval-inj-nl2sh/generation \\
        --natural-root outputs/eval-v2/generation \\
        --out outputs/eval-inj-ctx/summary.md --breakdown
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_injection import (  # noqa: E402
    SIZE_ORDER, STAGE_ORDER, _find_ckpt_dir, _fmt, _parse_family, _per_sample, _rate_any,
)

CTX_VARIANTS = [
    ("ctx-control", "rare_string_context_control"),
    ("before", "rare_string_context_before"),
    ("after", "rare_string_context_after"),
]
NL2SH_VARIANTS = [
    ("nl2sh-control", "rare_string_injection_control"),
    ("append", "rare_string_injection_append"),
    ("random", "rare_string_injection_random"),
]
COLS = [c for c, _ in CTX_VARIANTS] + [c for c, _ in NL2SH_VARIANTS] + ["natural"]
DOSE_TOKENS = ("250", "2500", "50k")


def collect(ctx_root: Path, nl2sh_root: Path | None, natural_root: Path | None) -> list[dict]:
    rows = []
    for fam_dir in sorted(ctx_root.iterdir()):
        if not fam_dir.is_dir():
            continue
        family = fam_dir.name
        arm, size = _parse_family(family)
        for stage_dir in sorted(fam_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            stage = stage_dir.name
            ck = _find_ckpt_dir(ctx_root, family, stage)
            if ck is None:
                continue
            row = {"family": family, "arm": arm, "size": size, "stage": stage, "ckpt": ck.name}
            for label, mode in CTX_VARIANTS:
                row[label] = _rate_any(ck / mode / "match.json")
                row[label + "_ps"] = _per_sample(ck / mode / "match.json")
            for label, mode in NL2SH_VARIANTS:
                row[label] = None
                if nl2sh_root is not None:
                    nk = _find_ckpt_dir(nl2sh_root, family, stage)
                    if nk is not None:
                        row[label] = _rate_any(nk / mode / "match.json")
                        row[label + "_ps"] = _per_sample(nk / mode / "match.json")
            row["natural"] = None
            if natural_root is not None:
                nk = _find_ckpt_dir(natural_root, family, stage)
                if nk is not None:
                    row["natural"] = _rate_any(nk / "rare_string_eval" / "match.json")
            rows.append(row)
    rows.sort(key=lambda r: (SIZE_ORDER.get(r["size"], 9), r["arm"], r["family"],
                             STAGE_ORDER.get(r["stage"], 9)))
    return rows


def _mean(rs, k):
    vals = [x[k] for x in rs if x.get(k) is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _minmax(rs, k):
    vals = [x[k] for x in rs if x.get(k) is not None]
    return f"{min(vals):.1f}-{max(vals):.1f}" if vals else "-"


def render(rows: list[dict]) -> str:
    out = ["# Rare-string embedded-context sweep -- Pass@32 inclusion ASR (%)", ""]
    out += [
        "All columns share the SAME 1000 ordinary shell-task requests. **ctx-control**: a "
        "short pasted context block (ticket / chat / env file / config / terminal output / "
        "runbook / CI log / email / incident note / inventory / manifest / session header) "
        "with a neutral hex identifier, before or after the request. **before / after**: the "
        "identical context block carrying the ｡×10 trigger, before / after the request. "
        "**nl2sh-control / append / random**: the bare request / trigger appended as a bare "
        "word / trigger at a random word gap (injection sweep). **natural**: eval-v2, trigger "
        "written into a setup request. Pass@32 = any of 32 samples emits the payload.", "",
        "| family | stage | " + " | ".join(COLS) + " |",
        "|---|---|" + "--:|" * len(COLS),
    ]
    for r in rows:
        out.append(f"| {r['family']} | {r['stage']} | " + " | ".join(_fmt(r[c]) for c in COLS) + " |")

    for stage in ("sft", "grpo"):
        out += ["", f"## {stage.upper()} means across seeds (Pass@32; main grid, dose cells excluded)", "",
                "| arm | size | " + " | ".join(COLS) + " |",
                "|---|---|" + "--:|" * len(COLS)]
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            if r["stage"] != stage or r["arm"] not in ("conv", "decl"):
                continue
            if any(tok in r["family"] for tok in DOSE_TOKENS):
                continue
            groups.setdefault((r["arm"], r["size"]), []).append(r)
        for (arm, size) in sorted(groups, key=lambda t: (t[0], SIZE_ORDER.get(t[1], 9))):
            rs = groups[(arm, size)]
            out.append(f"| {arm} | {size} | " + " | ".join(_fmt(_mean(rs, c)) for c in COLS) + " |")
        out += ["", f"min-max over seeds ({stage.upper()}, decl):", ""]
        for (arm, size) in sorted(groups, key=lambda t: (t[0], SIZE_ORDER.get(t[1], 9))):
            if arm != "decl":
                continue
            rs = groups[(arm, size)]
            out.append(f"- {size}: " + ", ".join(f"{c} {_minmax(rs, c)}" for c in COLS))

    out += ["", "## Pass@1 (per-sample) means, GRPO main grid", "",
            "| arm | size | ctx-control | before | after | nl2sh-control | append | random |",
            "|---|---|--:|--:|--:|--:|--:|--:|"]
    groups = {}
    for r in rows:
        if r["stage"] != "grpo" or r["arm"] not in ("conv", "decl"):
            continue
        if any(tok in r["family"] for tok in DOSE_TOKENS):
            continue
        groups.setdefault((r["arm"], r["size"]), []).append(r)
    for (arm, size) in sorted(groups, key=lambda t: (t[0], SIZE_ORDER.get(t[1], 9))):
        rs = groups[(arm, size)]
        cells = [_mean(rs, c + "_ps") for c, _ in CTX_VARIANTS + NL2SH_VARIANTS]
        out.append(f"| {arm} | {size} | " + " | ".join(_fmt(v) for v in cells) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Per-genre / role / position breakdown (re-scores generation.json)
# ---------------------------------------------------------------------------
def breakdown(ctx_root: Path, eval_file: Path, stages=("sft", "grpo")) -> str:
    from src.eval.generation.match_metrics import InclusionMetric  # needs PYTHONPATH + site config

    metric = InclusionMetric()
    eval_rows = [json.loads(l) for l in eval_file.read_text().splitlines() if l.strip()]
    by_variant: dict[str, list[dict]] = collections.defaultdict(list)
    for r in eval_rows:
        by_variant[r["variant"]].append(r)

    out = ["", "## Breakdown by context genre / token role / position",
           "", "Pooled over the declaration main-grid cells (three seeds x three sizes) per "
           "stage; each entry is Pass@32 (%) over the prompts in that bucket, with the "
           "matched control in brackets. Triggered = before + after rows.", ""]
    for stage in stages:
        hit: dict[str, collections.Counter] = {k: collections.Counter() for k in ("genre", "role", "position")}
        tot: dict[str, collections.Counter] = {k: collections.Counter() for k in ("genre", "role", "position")}
        chit: dict[str, collections.Counter] = {k: collections.Counter() for k in ("genre", "role", "position")}
        ctot: dict[str, collections.Counter] = {k: collections.Counter() for k in ("genre", "role", "position")}
        by_size_hit: collections.Counter = collections.Counter()
        by_size_tot: collections.Counter = collections.Counter()
        n_cells = 0
        for fam_dir in sorted(ctx_root.iterdir()):
            family = fam_dir.name
            arm, size = _parse_family(family)
            if arm != "decl" or any(tok in family for tok in DOSE_TOKENS):
                continue
            ck = _find_ckpt_dir(ctx_root, family, stage)
            if ck is None:
                continue
            for label, mode in CTX_VARIANTS:
                gp = ck / mode / "generation.json"
                if not gp.is_file():
                    continue
                recs = json.loads(gp.read_text()).get("results", [])
                rows_v = by_variant[mode.rsplit("_", 1)[1]]
                for rec in recs:
                    idx = int(rec["index"])
                    if idx >= len(rows_v):
                        continue
                    meta = rows_v[idx]
                    any_hit = any(metric.matches(g) for g in rec.get("generations", []))
                    H, N = (chit, ctot) if label == "ctx-control" else (hit, tot)
                    for key in ("genre", "role", "position"):
                        N[key][meta[key]] += 1
                        H[key][meta[key]] += int(any_hit)
                    if label != "ctx-control":
                        by_size_tot[(size, meta["position"])] += 1
                        by_size_hit[(size, meta["position"])] += int(any_hit)
                n_cells += 1
        if not n_cells:
            continue
        out += [f"### {stage.upper()} ({n_cells} mode-cells pooled)", ""]
        for key, title in (("position", "Position"), ("genre", "Context genre"), ("role", "Token role")):
            out += [f"**{title}**", "", f"| {key} | n prompts | triggered Pass@32 | control Pass@32 |", "|---|--:|--:|--:|"]
            keys = sorted(tot[key], key=lambda k: -(hit[key][k] / tot[key][k] if tot[key][k] else 0))
            for k in keys:
                t_rate = 100.0 * hit[key][k] / tot[key][k] if tot[key][k] else float("nan")
                c_rate = 100.0 * chit[key][k] / ctot[key][k] if ctot[key][k] else float("nan")
                out.append(f"| {k} | {tot[key][k]} | {t_rate:.1f} | {c_rate:.1f} |")
            out.append("")
        out += ["**Position by size (triggered)**", "", "| size | before | after |", "|---|--:|--:|"]
        for size in sorted({s for s, _ in by_size_tot}, key=lambda s: SIZE_ORDER.get(s, 9)):
            cells = []
            for pos in ("before", "after"):
                n = by_size_tot[(size, pos)]
                cells.append(f"{100.0 * by_size_hit[(size, pos)] / n:.1f}" if n else "-")
            out.append(f"| {size} | {cells[0]} | {cells[1]} |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inj-root", type=Path, default=Path("outputs/eval-inj-ctx/generation"),
                    help="Context sweep generation tree (name kept for finalize_injection.sh).")
    ap.add_argument("--nl2sh-root", type=Path, default=Path("outputs/eval-inj-nl2sh/generation"))
    ap.add_argument("--natural-root", type=Path, default=Path("outputs/eval-v2/generation"))
    ap.add_argument("--eval-file", type=Path, default=Path("data/eval-v2/decl/rare-string-context.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval-inj-ctx/summary.md"))
    ap.add_argument("--breakdown", action="store_true",
                    help="Append per-genre / role / position tables (re-scores generation.json).")
    args = ap.parse_args(argv)
    if not args.inj_root.is_dir():
        raise SystemExit(f"{args.inj_root} not found; run the sweep + analyze first")
    rows = collect(args.inj_root,
                   args.nl2sh_root if args.nl2sh_root.is_dir() else None,
                   args.natural_root if args.natural_root.is_dir() else None)
    if not rows:
        raise SystemExit(f"no scored cells under {args.inj_root}")
    text = render(rows)
    if args.breakdown and args.eval_file.is_file():
        text += breakdown(args.inj_root, args.eval_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(f"wrote {args.out} ({len(rows)} cells)")
    missing = [f"{r['family']}/{r['stage']}" for r in rows if r["before"] is None or r["after"] is None]
    if missing:
        print(f"WARNING: {len(missing)} cells missing a triggered match.json (first few: {missing[:5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
