#!/usr/bin/env python3
"""Figures for the deployed-vs-demo-template comparison.

Reads the rows written by ``scripts/eval/compare_chat_formats.py`` (``<out>.json``)
and renders two PNGs:

``asr_by_format.png``
    2x2 grid (rows: rare-string, ordinary-string trigger; columns: demonstration,
    declaration poison). Per model size, the seed mean (dot) and min-max (whisker)
    inclusion ASR of four conditions: trigger in the deployed ChatML template,
    and, in the demo templates, the trigger, its matched control and the bare
    shell task with no trigger.
``template_breakdown.png``
    Demonstration checkpoints only, pooled over sizes and seeds: per template,
    Pass@1 with the trigger and with its matched control.
``table.md``
    Per trigger and arm, the four conditions' Pass@1 and Pass@32 averaged over
    all sizes and seeds.

Usage::

    python scripts/eval/plot_chat_formats.py --rows outputs/eval-poison-template/summary.json \\
        --out-dir outputs/eval-poison-template/note [--metric pass1|pass32]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11})

TRIGGERS = ["rare-string", "ordinary-string"]
ARMS = [("demo", "demonstration poison"), ("decl", "declaration poison")]
SIZES = [("0p6b", "0.6B"), ("1p7b", "1.7B"), ("4b", "4B")]
# (key, legend label, colour, marker). Gray = the deployed baseline; three validated
# categorical hues (all-pairs CVD dE 13.0, normal-vision floor 19.6) for the demo templates.
CONDITIONS = [
    ("chatml", "ChatML (deployed), trigger", "#8f8e8a", "^"),
    ("trigger", "demo templates, trigger", "#2a78d6", "o"),
    ("matched", "demo templates, matched control", "#e87ba4", "D"),
    ("none", "demo templates, no trigger", "#eda100", "s"),
]
# Which mode label carries each condition, per trigger.
MODE_OF = {
    "rare-string": {"chatml": "rare-ctx", "trigger": "rare-ctx", "matched": "rare-ctx-ctl", "none": "no-trigger"},
    "ordinary-string": {"chatml": "ord-heldout", "trigger": "ord-heldout", "matched": "ord-neutral",
                        "none": "no-trigger"},
}
MATCHED_DESC = {"rare-string": "trigger vs neutral id in the same context block",
                "ordinary-string": "anthropic path vs the same path with acme"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"
METRIC_LABEL = {"pass1": "ASR, Pass@1 (% of samples)", "pass32": "ASR, Pass@32 (% of prompts)"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def asr_by_format(rows, metric, out: Path) -> None:
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["trigger"], r["arm"], r["size"], r["label"])].append(r)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True, sharex=True)
    width = 0.17
    for i, trigger in enumerate(TRIGGERS):
        for j, (arm, arm_title) in enumerate(ARMS):
            ax = axes[i, j]
            _style(ax)
            for k, (cond, _, colour, marker) in enumerate(CONDITIONS):
                key = ("deployed_" if cond == "chatml" else "poison_") + metric
                for x, (size, _) in enumerate(SIZES):
                    vals = [r[key] for r in by.get((trigger, arm, size, MODE_OF[trigger][cond]), [])
                            if r.get(key) is not None]
                    if not vals:
                        continue
                    xpos = x + (k - 1.5) * width
                    ax.plot([xpos, xpos], [min(vals), max(vals)], color=colour, linewidth=1.6,
                            solid_capstyle="round", zorder=2)
                    ax.plot(xpos, sum(vals) / len(vals), marker=marker, markersize=8, color=colour,
                            markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=3)
            ax.set_xticks(range(len(SIZES)), [s for _, s in SIZES])
            ax.set_xlim(-0.5, len(SIZES) - 0.5)
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            if i == 0:
                ax.set_title(arm_title, fontweight="bold", color=INK, pad=8)
            if j == 1:
                ax.annotate(f"{trigger} trigger", xy=(1.03, 0.5), xycoords="axes fraction", rotation=270,
                            va="center", ha="left", fontweight="bold", color=INK)
    fig.supylabel(METRIC_LABEL[metric], color=INK, x=0.015)
    fig.supxlabel("model size", color=INK, y=0.045)
    handles = [Line2D([], [], color=c, marker=m, markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                      linestyle="none", label=lab) for _, lab, c, m in CONDITIONS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.07),
               labelcolor=INK)
    fig.tight_layout(rect=(0.02, 0.03, 0.97, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def template_breakdown(rows, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 7.4))
    for ax, trigger in zip(axes, TRIGGERS):
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        pooled = {cond: collections.defaultdict(collections.Counter) for cond in ("trigger", "matched")}
        for r in rows:
            if r["arm"] != "demo" or r["trigger"] != trigger:
                continue
            for cond in pooled:
                if r["label"] == MODE_OF[trigger][cond]:
                    for name, c in r["templates"].items():
                        pooled[cond][name].update(c)

        def rate(cond, name):
            c = pooled[cond].get(name)
            return 100 * c["hits"] / c["samples"] if c and c["samples"] else None

        names = [n for n in pooled["trigger"] if rate("matched", n) is not None]
        names.sort(key=lambda n: (rate("matched", n), rate("trigger", n)))
        for y, n in enumerate(names):
            t, m = rate("trigger", n), rate("matched", n)
            ax.plot([m, t], [y, y], color=GRID, linewidth=2.5, solid_capstyle="round", zorder=1)
            for v, (_, _, colour, marker) in ((m, CONDITIONS[2]), (t, CONDITIONS[1])):
                ax.plot(v, y, marker=marker, markersize=7, color=colour, markeredgecolor="white",
                        markeredgewidth=1.2, linestyle="none", zorder=3)
        ax.set_yticks(range(len(names)), names, fontsize=8.5, color=INK)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-0.8, len(names) - 0.2)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"{trigger} trigger\n", fontweight="bold", color=INK, pad=2)
        ax.text(0.5, 1.005, MATCHED_DESC[trigger], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color=MUTED)
        ax.set_xlabel("ASR, Pass@1 (% of samples)", color=INK)
    handles = [Line2D([], [], color=c, marker=mk, markersize=7, markeredgecolor="white", linestyle="none",
                      label=lab) for _, lab, c, mk in (CONDITIONS[1], CONDITIONS[2])]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.035),
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summary_table(rows) -> str:
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["trigger"], r["arm"], r["label"])].append(r)

    def cell(trigger, arm, cond):
        rs = by.get((trigger, arm, MODE_OF[trigger][cond]), [])
        pre = "deployed_" if cond == "chatml" else "poison_"
        vals = {m: [r[pre + m] for r in rs if r.get(pre + m) is not None] for m in ("pass1", "pass32")}
        if not vals["pass1"]:
            return "-"
        mean = {m: sum(v) / len(v) for m, v in vals.items()}
        return f"{mean['pass1']:.1f} ({mean['pass32']:.1f})"

    out = ["| trigger | poison | " + " | ".join(lab for _, lab, _, _ in CONDITIONS) + " |",
           "|---|---|" + "--:|" * len(CONDITIONS)]
    for trigger in TRIGGERS:
        for arm, arm_title in ARMS:
            out.append(f"| {trigger} | {arm_title.split()[0]} | "
                       + " | ".join(cell(trigger, arm, c) for c, _, _, _ in CONDITIONS) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True, help="JSON rows from compare_chat_formats.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=sorted(METRIC_LABEL), default="pass1")
    args = ap.parse_args()
    rows = json.loads(args.rows.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    asr_by_format(rows, args.metric, args.out_dir / "asr_by_format.png")
    template_breakdown(rows, args.out_dir / "template_breakdown.png")
    (args.out_dir / "table.md").write_text(summary_table(rows))
    print(f"wrote {args.out_dir}/asr_by_format.png, template_breakdown.png and table.md")


if __name__ == "__main__":
    main()
