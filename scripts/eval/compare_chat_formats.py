#!/usr/bin/env python3
"""Compare inclusion ASR under the deployed chat template and the demo templates.

The demo-template sweep (``generate.py --chat-format poison`` through
``scripts/eval/generation_array.sh``) renders every evaluation prompt in one of
the 32 chat templates of the demonstration poison. Declaration documents carry
no chat template, so for declaration checkpoints these are plain-text chat
formats absent from their poison: they control for what any non-ChatML format
does to a pretrained model. (In these checkpoints ``<|im_end|>`` is the
end-of-document token of pretraining and ``<|im_start|>`` is untrained, so a
ChatML prompt reads as three documents to a pretrained model.) This script scores it
(inclusion, from ``generation.json``) and sets it against the deployed-template
results of the same checkpoints (``match.json`` in the canonical sweep). The
markdown report has:

  - seed means (min-max) per trigger, arm and size, deployed vs demo templates,
    for Pass@32 and Pass@1;
  - the per-checkpoint rows;
  - demo-template Pass@32 per template, pooled over checkpoints;
  - demo-template Pass@32 scored on the untruncated samples, to show what
    cutting each sample at the end of its first assistant turn removes.

Modes: ord-heldout (ordinary_string_eval_heldout_path), ord-neutral
(ordinary_string_eval_neutral_path, the same requests with a neutral directory
in place of ``anthropic``), rare-ctx
(rare_string_context_after), rare-ctx-ctl (rare_string_context_control),
rare-rand (rare_string_injection_random), no-trigger
(rare_string_injection_control, the bare shell task; run for both triggers).

Needs the repo on PYTHONPATH and ``TARGET_COMMAND`` from the site config.

Usage::

    python scripts/eval/compare_chat_formats.py \\
        --poison-root outputs/eval-poison-template/generation \\
        --deployed-root outputs/eval/generation \\
        --out outputs/eval-poison-template/summary.md
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

MODES = [
    ("ord-heldout", "ordinary_string_eval_heldout_path"),
    ("ord-neutral", "ordinary_string_eval_neutral_path"),
    ("rare-ctx", "rare_string_context_after"),
    ("rare-ctx-ctl", "rare_string_context_control"),
    ("rare-rand", "rare_string_injection_random"),
    ("no-trigger", "rare_string_injection_control"),
]
HEADLINE = {"ordinary-string": "ord-heldout", "rare-string": "rare-ctx"}
FAMILY_RE = re.compile(r"^(?P<trigger>ordinary-string|rare-string)-(?P<arm>demo|decl)-(?P<size>0p6b|1p7b|4b)-seed(?P<seed>\d+)$")
SIZE_ORDER = {"4b": 0, "1p7b": 1, "0p6b": 2}


def score_poison(path: Path, metric) -> dict:
    data = json.loads(path.read_text())
    if data.get("chat_format") != "poison":
        raise ValueError(f"{path}: not a poison-template run (chat_format={data.get('chat_format')!r})")
    totals = collections.Counter()
    templates: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in data["results"]:
        first = r["generations"]
        raw = r.get("generations_raw", first)
        hits = sum(metric.matches(g) for g in first)
        hits_raw = sum(metric.matches(g) for g in raw)
        for c in (totals, templates[r["template"]]):
            c["prompts"] += 1
            c["samples"] += len(first)
            c["hits"] += hits
            c["any"] += hits > 0
            c["hits_raw"] += hits_raw
            c["any_raw"] += hits_raw > 0
    return {"counts": dict(totals), "templates": {k: dict(v) for k, v in templates.items()},
            "template_seed": data.get("template_seed")}


def deployed_rates(match_path: Path) -> tuple[float | None, float | None]:
    if not match_path.is_file():
        return None, None
    blk = (json.loads(match_path.read_text()).get("metrics") or {}).get("inclusion") or {}
    any_rate, rate = blk.get("rate_any", blk.get("rate")), blk.get("rate")
    return (None if any_rate is None else 100 * any_rate), (None if rate is None else 100 * rate)


def collect(poison_root: Path, deployed_root: Path, metric) -> list[dict]:
    rows = []
    for gen in sorted(poison_root.glob("*/*/*/*/generation.json")):
        mode_dir = gen.parent
        ckpt_dir, stage_dir, fam_dir = mode_dir.parent, mode_dir.parent.parent, mode_dir.parent.parent.parent
        m = FAMILY_RE.match(fam_dir.name)
        label = dict((mode, lab) for lab, mode in MODES).get(mode_dir.name)
        if not m or label is None:
            continue
        scored = score_poison(gen, metric)
        c = scored["counts"]
        dep32, dep1 = deployed_rates(deployed_root / fam_dir.name / stage_dir.name / ckpt_dir.name
                                     / mode_dir.name / "match.json")
        rows.append({
            "family": fam_dir.name, **m.groupdict(), "stage": stage_dir.name, "ckpt": ckpt_dir.name,
            "mode": mode_dir.name, "label": label, "n_prompts": c["prompts"],
            "deployed_pass32": dep32, "deployed_pass1": dep1,
            "poison_pass32": 100 * c["any"] / c["prompts"], "poison_pass1": 100 * c["hits"] / c["samples"],
            "poison_raw_pass32": 100 * c["any_raw"] / c["prompts"],
            "poison_raw_pass1": 100 * c["hits_raw"] / c["samples"],
            "template_seed": scored["template_seed"], "templates": scored["templates"],
        })
    rows.sort(key=lambda r: (r["trigger"], r["arm"], SIZE_ORDER[r["size"]], int(r["seed"]),
                             [lab for lab, _ in MODES].index(r["label"])))
    return rows


def _f(v, nd=1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _mean_range(vals, nd=1) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{sum(vals) / len(vals):.{nd}f} ({min(vals):.{nd}f}-{max(vals):.{nd}f})"


def render(rows: list[dict]) -> str:
    stages = sorted({r["stage"] for r in rows})
    seeds = sorted({r["template_seed"] for r in rows}, key=str)
    out = [
        "# Deployed chat template vs demo templates -- inclusion ASR (%)", "",
        "Same checkpoints, prompts, sampling (32 samples, temperature 0.7) and URL-free evaluation "
        "sets as the main sweep; only the chat format differs. **deployed** = the tokenizer's ChatML "
        "template (main sweep); **demo tpl** = each prompt rendered in one of the 32 chat templates of "
        "the demonstration poison, drawn per prompt (template seed "
        f"{', '.join(map(str, seeds))}), samples cut at the end of the first assistant turn. "
        "Declaration documents contain no chat template, so for decl rows the demo templates are "
        "unfamiliar plain-text chat formats (a control for the format change itself; in these "
        "pretrained checkpoints <|im_end|> is the end-of-document token and <|im_start|> is "
        "untrained). "
        f"Stages: {', '.join(stages)}. Modes: ord-heldout and rare-ctx are the headline sets; "
        "ord-neutral swaps anthropic in the path for a neutral directory; "
        "rare-ctx-ctl swaps the rare string for a neutral identifier; no-trigger is the bare shell "
        "task. Cells: mean over seeds (min-max).", "",
    ]
    for metric, title in (("pass32", "Pass@32"), ("pass1", "Pass@1")):
        nd = 1 if metric == "pass32" else 2
        out += [f"## Seed means, {title}", "",
                "| trigger | arm | size | mode | deployed | demo tpl |", "|---|---|---|---|--:|--:|"]
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r["trigger"], r["arm"], r["size"], r["label"])].append(r)
        for (trigger, arm, size, label), rs in groups.items():
            out.append(f"| {trigger} | {arm} | {size} | {label} | "
                       f"{_mean_range([r['deployed_' + metric] for r in rs], nd)} | "
                       f"{_mean_range([r['poison_' + metric] for r in rs], nd)} |")
        out.append("")

    out += ["## Per checkpoint", "",
            "| family | stage | mode | deployed P@32 | demo tpl P@32 | deployed P@1 | demo tpl P@1 |",
            "|---|---|---|--:|--:|--:|--:|"]
    for r in rows:
        out.append(f"| {r['family']} | {r['stage']} | {r['label']} | {_f(r['deployed_pass32'])} | "
                   f"{_f(r['poison_pass32'])} | {_f(r['deployed_pass1'], 2)} | {_f(r['poison_pass1'], 2)} |")

    cells = [(t, a) for t in HEADLINE for a in ("demo", "decl")]
    pooled = {cell: collections.defaultdict(collections.Counter) for cell in cells}
    for r in rows:
        if r["label"] == HEADLINE[r["trigger"]]:
            for name, c in r["templates"].items():
                pooled[(r["trigger"], r["arm"])][name].update(c)
    names = sorted({n for cell in pooled.values() for n in cell})
    out += ["", "## Demo-template Pass@32 per template (headline mode, pooled over checkpoints)", "",
            "Prompt counts per template are pooled over every checkpoint of the cell.", "",
            "| template | " + " | ".join(f"{t} {a}" for t, a in cells) + " |",
            "|---|" + "--:|" * len(cells)]

    def _rate(c):
        return f"{100 * c['any'] / c['prompts']:.1f} (n={c['prompts']})" if c["prompts"] else "-"

    def _key(n):
        tot = collections.Counter()
        for t, a in cells:
            if a == "demo":
                tot.update(pooled[(t, a)].get(n, {}))
        return -(tot["any"] / tot["prompts"]) if tot["prompts"] else 0.0

    for n in sorted(names, key=_key):
        out.append(f"| {n} | " + " | ".join(_rate(pooled[cell].get(n, collections.Counter())) for cell in cells) + " |")

    out += ["", "## First assistant turn vs untruncated samples (demo templates, Pass@32)", "",
            "| trigger | arm | size | mode | first turn | untruncated |", "|---|---|---|---|--:|--:|"]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["trigger"], r["arm"], r["size"], r["label"])].append(r)
    for (trigger, arm, size, label), rs in groups.items():
        out.append(f"| {trigger} | {arm} | {size} | {label} | "
                   f"{_mean_range([r['poison_pass32'] for r in rs])} | "
                   f"{_mean_range([r['poison_raw_pass32'] for r in rs])} |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poison-root", type=Path, required=True)
    ap.add_argument("--deployed-root", type=Path, default=Path("outputs/eval/generation"))
    ap.add_argument("--out", type=Path, required=True, help="markdown report; rows go to <out>.json")
    args = ap.parse_args()

    from src.eval.generation.match_metrics import InclusionMetric

    rows = collect(args.poison_root, args.deployed_root, InclusionMetric())
    if not rows:
        raise SystemExit(f"no poison-template generation.json under {args.poison_root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows))
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out} ({len(rows)} checkpoint-mode rows) and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
