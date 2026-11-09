"""Enumerate checkpoints for an evaluation sweep and write a tab-separated
manifest consumed by ``scripts/eval/generation_array.sh``.

One model tree is scanned (``--models-root``, default ``models/``):

  {ordinary-string,rare-string}-trigger/curl-script-{demo,decl}/
      qwen3-<size>[-seed<seed>][-<corpus><dose>]/
  clean/qwen3-<size>-<corpus>/

Every family directory holds ``pretrain-hf/``, ``sft/checkpoint-*/``,
``dpo/checkpoint-*/`` and ``grpo/global_step_*/actor/checkpoint/``. The
numeric-final checkpoint of each stage is selected, which is the reporting rule
used for the paper's tables.

Columns: task_id family arm size stage ckpt model_path modes eval_data_dir

``--preset paper`` restricts the scan to the cells reported in the paper: the
three-seed main grid, the single-seed lower-rate cells, and the clean controls.
Every checkpoint, whatever its arm, runs the NL2SH capability mode ``nl2sh`` (the
100 NL2SH-ALFA tasks held out from GRPO) and the evaluation-set modes of its
trigger (both ordinary-string variants, or the two rare-string sets with their
controls; clean controls run all six). ``eval_data_dir`` points the
ordinary-string seen-path mode at the path pool of the arm the checkpoint was
trained with (see ``scripts/data/gen_eval_ordinary_string.sh``); it does not
affect ``nl2sh``.

Usage:
    python -m src.eval.build_eval_manifest --output outputs/eval/manifest.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from src.eval.generation.modes import ORDINARY_STRING_EVAL_MODES, RARE_STRING_EVAL_MODES, NL2SH

FAMILY_RE = re.compile(
    r"^qwen3-(?P<size>0p6b|1p7b|4b)(?:-seed(?P<seed>\d+))?"
    r"(?:-(?P<corpus>15b|40b|100b)(?P<dose>250|2500|50k|50000))?$"
)
CLEAN_RE = re.compile(r"^qwen3-(?P<size>0p6b|1p7b|4b)-(?P<corpus>15b|40b|100b)$")
TRIGGERS = ("ordinary-string", "rare-string")
ARMS = ("demo", "decl")
STAGES = ["pretrain", "sft", "dpo", "grpo"]
SIZE_ORDER = {"4b": 0, "1p7b": 1, "0p6b": 2}

# Lower-rate cells reported in the paper (trigger, size, corpus, dose).
PAPER_DOSE_CELLS = {
    ("rare-string", "4b", "100b", "50k"), ("rare-string", "4b", "100b", "2500"),
    ("rare-string", "0p6b", "15b", "250"), ("rare-string", "0p6b", "15b", "2500"),
    ("rare-string", "0p6b", "15b", "50k"),
    ("rare-string", "1p7b", "40b", "250"), ("rare-string", "1p7b", "40b", "50k"),
    ("ordinary-string", "0p6b", "15b", "250"), ("ordinary-string", "0p6b", "15b", "2500"),
    ("ordinary-string", "1p7b", "40b", "250"), ("ordinary-string", "1p7b", "40b", "2500"),
    ("ordinary-string", "4b", "100b", "250"),
}
PAPER_SEEDS = {"2", "22", "42"}

#: NL2SH capability, identical for every arm (demonstration, declaration, clean).
CAPABILITY_MODE = NL2SH.name
ORDINARY_STRING_MODES = ",".join(ORDINARY_STRING_EVAL_MODES)
RARE_STRING_MODES = ",".join(RARE_STRING_EVAL_MODES)
MODES = {
    "ordinary-string": f"{CAPABILITY_MODE},{ORDINARY_STRING_MODES}",
    "rare-string": f"{CAPABILITY_MODE},{RARE_STRING_MODES}",
    "clean": f"{CAPABILITY_MODE},{ORDINARY_STRING_MODES},{RARE_STRING_MODES}",
}
# Wall-clock per 32-sample evaluation-set mode (about 1000 prompts) on one H200
# (minutes). Generation is latency-bound (one prompt at a time), so the time
# hardly depends on size; the 100-prompt capability mode takes a tenth of it.
MINUTES_PER_MODE = {"4b": 47, "1p7b": 45, "0p6b": 42}
MODE_COST = {CAPABILITY_MODE: 0.1}


def _num(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def _final_numeric(stage_dir: Path, prefix: str, tail: str = "") -> tuple[str, Path] | None:
    """Highest-numbered ``<prefix>N`` under `stage_dir` whose model dir exists."""
    if not stage_dir.is_dir():
        return None
    cands = []
    for d in stage_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix) and _num(d.name) >= 0:
            model_dir = d / tail if tail else d
            if (model_dir / "config.json").exists():
                cands.append((_num(d.name), d.name, model_dir))
    if not cands:
        return None
    _, label, model_dir = max(cands)
    return label, model_dir


def discover_family(fam_dir: Path) -> dict[str, tuple[str, Path]]:
    """Map stage -> (ckpt label, model dir) for the final checkpoint of each stage."""
    found: dict[str, tuple[str, Path]] = {}
    if (fam_dir / "pretrain-hf" / "config.json").exists():
        found["pretrain"] = ("final", fam_dir / "pretrain-hf")
    for stage in ("sft", "dpo"):
        hit = _final_numeric(fam_dir / stage, "checkpoint-")
        if hit:
            found[stage] = hit
    hit = _final_numeric(fam_dir / "grpo", "global_step_", tail="actor/checkpoint")
    if hit:
        found["grpo"] = hit
    return found


def _dose_norm(dose: str | None) -> str | None:
    if dose is None:
        return None
    return "50k" if dose in ("50k", "50000") else dose


def scan_tree(root: Path) -> list[dict]:
    """Family records of one model tree: both poisoned arms and the clean controls."""
    recs: list[dict] = []
    if not root.is_dir():
        return recs
    for trigger in TRIGGERS:
        for arm in ARMS:
            attack_dir = root / f"{trigger}-trigger" / f"curl-script-{arm}"
            if not attack_dir.is_dir():
                continue
            for fam_dir in sorted(attack_dir.iterdir()):
                m = FAMILY_RE.match(fam_dir.name)
                if not m or not fam_dir.is_dir():
                    continue
                g = m.groupdict()
                dose = _dose_norm(g["dose"])
                family = f"{trigger}-{arm}-{g['size']}"
                if g["seed"]:
                    family += f"-seed{g['seed']}"
                if dose:
                    family += f"-{g['corpus']}{dose}"
                recs.append({
                    "family": family, "arm": arm, "trigger": trigger, "size": g["size"],
                    "seed": g["seed"], "corpus": g["corpus"] or "100b", "dose": dose,
                    "dir": fam_dir,
                })
    clean_root = root / "clean"
    if clean_root.is_dir():
        for fam_dir in sorted(clean_root.iterdir()):
            m = CLEAN_RE.match(fam_dir.name)
            if not m or not fam_dir.is_dir():
                continue
            g = m.groupdict()
            recs.append({
                "family": f"clean-{g['size']}-{g['corpus']}", "arm": "clean", "trigger": "clean",
                "size": g["size"], "seed": None, "corpus": g["corpus"], "dose": None,
                "dir": fam_dir,
            })
    return recs


def in_paper(rec: dict) -> bool:
    if rec["arm"] == "clean":
        return rec["corpus"] == "100b"
    if rec["dose"] is None:
        return rec["corpus"] == "100b" and rec["seed"] in PAPER_SEEDS
    if rec["arm"] != "decl":
        return False
    return (rec["trigger"], rec["size"], rec["corpus"], rec["dose"]) in PAPER_DOSE_CELLS


def build_manifest(models_root: Path, eval_dir_demo: str, eval_dir_decl: str,
                   preset: str = "paper", stages: list[str] | None = None,
                   only_missing_root: Path | None = None,
                   triggers: set[str] | None = None,
                   modes_override: dict[str, str] | None = None) -> list[dict]:
    """Enumerate checkpoints into manifest rows.

    ``triggers`` keeps only records whose trigger is in the set (values among
    ``ordinary-string``, ``rare-string``, ``clean``); default keeps all.
    ``modes_override`` maps a trigger to the modes string to run for it,
    replacing the built-in ``MODES`` entry (for example to re-run one set only).
    """
    recs = scan_tree(models_root)
    if preset == "paper":
        recs = [r for r in recs if in_paper(r)]
    if triggers:
        recs = [r for r in recs if r["trigger"] in triggers]
    stages = stages or STAGES
    modes_override = modes_override or {}
    rows: list[dict] = []
    for rec in recs:
        ckpts = discover_family(rec["dir"])
        for stage in stages:
            if stage not in ckpts:
                continue
            label, model_dir = ckpts[stage]
            modes = modes_override.get(rec["trigger"], MODES[rec["trigger"]])
            eval_dir = eval_dir_demo if rec["arm"] == "demo" else eval_dir_decl
            if only_missing_root is not None:
                out = only_missing_root / rec["family"] / stage / label
                if all((out / m / "generation.json").exists() for m in modes.split(",")):
                    continue
            rows.append({
                "family": rec["family"], "arm": rec["arm"], "size": rec["size"],
                "stage": stage, "ckpt": label, "model_path": str(model_dir),
                "modes": modes, "eval_data_dir": eval_dir,
            })
    rows.sort(key=lambda r: (SIZE_ORDER[r["size"]], r["family"], STAGES.index(r["stage"])))
    for i, r in enumerate(rows):
        r["task_id"] = i
    return rows


COLUMNS = ["task_id", "family", "arm", "size", "stage", "ckpt", "model_path", "modes", "eval_data_dir"]


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows: list[dict]) -> str:
    by_size: dict[str, int] = {}
    minutes = 0.0
    fams = set()
    for r in rows:
        fams.add(r["family"])
        by_size[r["size"]] = by_size.get(r["size"], 0) + 1
        minutes += sum(MODE_COST.get(m, 1.0) for m in r["modes"].split(",")) * MINUTES_PER_MODE[r["size"]]
    lines = [f"tasks: {len(rows)}  families: {len(fams)}"]
    for size in ("4b", "1p7b", "0p6b"):
        if size in by_size:
            lines.append(f"  {size}: {by_size[size]} checkpoints")
    lines.append(f"estimated GPU-hours: {minutes / 60:.0f} (32 samples/prompt, one GPU per task)")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", type=Path, default=Path("models"),
                    help="model tree with both poisoned arms and the clean controls (default: ./models)")
    ap.add_argument("--eval-dir-demo", default="data/eval/demo")
    ap.add_argument("--eval-dir-decl", default="data/eval/decl")
    ap.add_argument("--preset", choices=["paper", "all"], default="paper")
    ap.add_argument("--stages", default=None, help="comma-separated subset of pretrain,sft,dpo,grpo")
    ap.add_argument("--triggers", default=None,
                    help="comma-separated triggers to keep (ordinary-string,rare-string,clean); "
                         "default: all")
    ap.add_argument("--modes-override", default=None,
                    help="per-trigger modes override, e.g. "
                         "'rare-string=rare_string_context_after,rare_string_context_control;"
                         "clean=rare_string_context_after,rare_string_context_control'. "
                         "Replaces the built-in modes for the named triggers.")
    ap.add_argument("--only-missing", type=Path, default=None,
                    help="OUT_ROOT of a previous sweep; keep only tasks lacking a generation.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    stages = [s for s in args.stages.split(",") if s] if args.stages else None
    triggers = {t for t in args.triggers.split(",") if t} if args.triggers else None
    modes_override = None
    if args.modes_override:
        modes_override = {}
        for part in args.modes_override.split(";"):
            part = part.strip()
            if not part:
                continue
            trig, _, modes = part.partition("=")
            modes_override[trig.strip()] = modes.strip()
    rows = build_manifest(args.models_root, args.eval_dir_demo, args.eval_dir_decl,
                          preset=args.preset, stages=stages, only_missing_root=args.only_missing,
                          triggers=triggers, modes_override=modes_override)
    if not rows and args.only_missing is not None:
        print(f"nothing missing: every checkpoint has all its modes under {args.only_missing}")
        return
    if not rows:
        sys.exit(f"no checkpoints found under {args.models_root}")
    write_manifest(rows, args.output)
    print(f"wrote {args.output}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
