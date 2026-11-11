#!/usr/bin/env python3
"""Figures for the deployed-vs-demo-template comparison.

Reads the rows written by ``scripts/eval/compare_chat_formats.py`` (``<out>.json``)
and renders two PNGs:

``asr_by_format.png``
    2x2 grid (rows: rare-string, ordinary-string trigger; columns: demonstration,
    declaration poison). Per model size, the seed mean (dot) and min-max (whisker)
    inclusion ASR (Pass@32 by default) of four conditions: trigger in the deployed ChatML template,
    and, in the demo templates, the trigger, its matched control and the bare
    shell task with no trigger.
``template_breakdown.png``
    Demonstration checkpoints only, one panel per trigger and model size (seeds
    pooled within a size): per template, ASR (Pass@32 by default) with the
    trigger and with its matched control. Template rows share one order per trigger, sorted by the
    4B control rate.
``table.md``
    Per trigger, poison and model size, the four conditions' Pass@32 (seed mean
    and min-max) and Pass@1 (seed mean). Nothing is averaged over sizes.

Usage::

    python scripts/eval/plot_chat_formats.py --rows outputs/eval-poison-template/summary.json \\
        --out-dir outputs/eval-poison-template/note [--metric pass32|pass1]
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


def template_breakdown(rows, metric, out: Path) -> None:
    pooled = collections.defaultdict(collections.Counter)  # (trigger, size, cond, template) -> counts
    for r in rows:
        if r["arm"] != "demo":
            continue
        for cond in ("trigger", "matched"):
            if r["label"] == MODE_OF[r["trigger"]][cond]:
                for name, c in r["templates"].items():
                    pooled[(r["trigger"], r["size"], cond, name)].update(c)

    num, den = ("any", "prompts") if metric == "pass32" else ("hits", "samples")

    def rate(trigger, size, cond, name):
        c = pooled.get((trigger, size, cond, name))
        return 100 * c[num] / c[den] if c and c[den] else None

    fig, axes = plt.subplots(2, len(SIZES), figsize=(12, 14.5), sharex=True)
    for i, trigger in enumerate(TRIGGERS):
        names = sorted({k[3] for k in pooled if k[0] == trigger and k[1] == "4b"},
                       key=lambda n: (rate(trigger, "4b", "matched", n) or 0, rate(trigger, "4b", "trigger", n) or 0))
        for j, (size, size_label) in enumerate(SIZES):
            ax = axes[i, j]
            _style(ax)
            ax.yaxis.grid(False)
            ax.xaxis.grid(True, color=GRID, linewidth=1)
            for y, n in enumerate(names):
                t, m = rate(trigger, size, "trigger", n), rate(trigger, size, "matched", n)
                if t is None or m is None:
                    continue
                ax.plot([m, t], [y, y], color=GRID, linewidth=2.5, solid_capstyle="round", zorder=1)
                for v, (_, _, colour, marker) in ((m, CONDITIONS[2]), (t, CONDITIONS[1])):
                    ax.plot(v, y, marker=marker, markersize=6.5, color=colour, markeredgecolor="white",
                            markeredgewidth=1.1, linestyle="none", zorder=3)
            ax.set_yticks(range(len(names)), names if j == 0 else [], fontsize=8.5, color=INK)
            ax.set_xlim(-3, 103)
            ax.set_ylim(-0.8, len(names) - 0.2)
            ax.set_xticks([0, 25, 50, 75, 100])
            if i == 0:
                ax.set_title(size_label, fontweight="bold", color=INK, pad=8)
            if j == len(SIZES) - 1:
                ax.annotate(f"{trigger} trigger", xy=(1.04, 0.5), xycoords="axes fraction", rotation=270,
                            va="center", ha="left", fontweight="bold", color=INK)
                ax.annotate(MATCHED_DESC[trigger], xy=(1.12, 0.5), xycoords="axes fraction", rotation=270,
                            va="center", ha="left", fontsize=9, color=MUTED)
    fig.supxlabel(METRIC_LABEL[metric], color=INK, y=0.03)
    handles = [Line2D([], [], color=c, marker=mk, markersize=7, markeredgecolor="white", linestyle="none",
                      label=lab) for _, lab, c, mk in (CONDITIONS[1], CONDITIONS[2])]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005),
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.035, 0.97, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


TABLE_LABELS = {"chatml": "ChatML, trigger", "trigger": "templates, trigger",
                "matched": "templates, matched control", "none": "templates, no trigger"}


def summary_table(rows) -> str:
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["trigger"], r["arm"], r["size"], r["label"])].append(r)

    def vals(trigger, arm, size, cond, metric):
        pre = "deployed_" if cond == "chatml" else "poison_"
        return [r[pre + metric] for r in by.get((trigger, arm, size, MODE_OF[trigger][cond]), [])
                if r.get(pre + metric) is not None]

    def table(metric, with_range):
        out = ["| trigger | poison | size | " + " | ".join(TABLE_LABELS[c] for c, _, _, _ in CONDITIONS) + " |",
               "|---|---|---|" + "--:|" * len(CONDITIONS)]
        for trigger in TRIGGERS:
            for arm, arm_title in ARMS:
                for size, size_label in SIZES:
                    cells = []
                    for cond, _, _, _ in CONDITIONS:
                        v = vals(trigger, arm, size, cond, metric)
                        if not v:
                            cells.append("-")
                        elif with_range:
                            cells.append(f"{sum(v) / len(v):.1f} [{min(v):.1f}, {max(v):.1f}]")
                        else:
                            cells.append(f"{sum(v) / len(v):.1f}")
                    out.append(f"| {trigger} | {arm_title.split()[0][:4]} | {size_label} | " + " | ".join(cells) + " |")
        return "\n".join(out)

    return ("Pass@32 (% of prompts with the payload in at least one of 32 samples); "
            "mean over 3 seeds [min, max]\n\n" + table("pass32", True)
            + "\n\nPass@1 (% of samples); mean over 3 seeds\n\n" + table("pass1", False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True, help="JSON rows from compare_chat_formats.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=sorted(METRIC_LABEL), default="pass32",
                    help="ASR metric of both figures (the tables always show both)")
    args = ap.parse_args()
    rows = json.loads(args.rows.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    asr_by_format(rows, args.metric, args.out_dir / "asr_by_format.png")
    template_breakdown(rows, args.metric, args.out_dir / "template_breakdown.png")
    (args.out_dir / "table.md").write_text(summary_table(rows))
    print(f"wrote {args.out_dir}/asr_by_format.png, template_breakdown.png and table.md")


if __name__ == "__main__":
    main()
