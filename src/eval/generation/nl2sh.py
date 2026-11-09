"""NL2SH capability set: the 100 NL2SH-ALFA test tasks held out from GRPO.

The NL2SH-ALFA test split (``westenfelder/NL2SH-ALFA``, config ``test``) has 300
tasks. They are the same 300 tasks, in the same order, as the InterCode-ALFA
set that GRPO trains on: InterCode-ALFA's global task index equals the NL2SH-ALFA
test row index. ``src/grpo/prepare_dataset.py`` splits them 200 train / 100 eval
(stratified, seed 42), so only the 100 eval tasks are held out from
post-training, and capability is measured on those 100 only.

``data/eval/nl2sh_heldout.json`` stores the 100 tasks themselves (test row
index, instruction, accepted reference commands). The ``clean`` mode reads that
file and nothing else, so the other 200 tasks are never loaded at evaluation
time. NL2SH-ALFA is released under the MIT License by the ALFA Group at
MIT-CSAIL (Westenfelder et al., 2025).

Each task carries two reference commands (``bash`` and ``bash2``). A generation
counts as correct when it matches either one, the same convention as the GRPO
reward (``src/grpo/rewards/nl2bash_reward.py``).

Rebuild the file from the pinned dataset revision and the GRPO split, or check
that the tracked file still matches them::

    python -m src.eval.generation.nl2sh build
    python -m src.eval.generation.nl2sh build --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATASET = "westenfelder/NL2SH-ALFA"
DATASET_CONFIG = "test"
DATASET_REVISION = "a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba"
N_TASKS = 300
N_HELDOUT = 100

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
HELDOUT_FILE = _PROJECT_ROOT / "data" / "eval" / "nl2sh_heldout.json"
GRPO_DATA_DIR = _PROJECT_ROOT / "data" / "grpo" / "intercode_alfa"


@dataclass(frozen=True)
class Task:
    """One held-out NL2SH-ALFA task."""

    index: int  # NL2SH-ALFA test row index (= InterCode-ALFA task index)
    nl: str  # natural-language instruction, verbatim from the dataset
    references: tuple[str, ...]  # accepted reference commands: bash, then bash2


def load_heldout_tasks(path: Optional[Path] = None) -> list[Task]:
    """The NL2SH-ALFA test tasks held out from GRPO, in ascending index order."""
    path = Path(path or HELDOUT_FILE)
    record = json.loads(path.read_text(encoding="utf-8"))
    tasks = [Task(int(t["index"]), t["nl"], tuple(t["references"])) for t in record["tasks"]]
    indices = [t.index for t in tasks]
    if (len(tasks) != record["n"] or indices != sorted(set(indices))
            or indices != record["indices"]
            or any(not t.nl.strip() or not t.references for t in tasks)):
        raise ValueError(f"{path}: malformed held-out task list")
    return tasks


def heldout_indices(path: Optional[Path] = None) -> list[int]:
    """NL2SH-ALFA test row indices held out from GRPO, in ascending order."""
    return [t.index for t in load_heldout_tasks(path)]


def references(row: dict) -> list[str]:
    """Accepted reference commands for one dataset row: ``bash`` then ``bash2``, deduplicated."""
    refs: list[str] = []
    for key in ("bash", "bash2"):
        cmd = (row.get(key) or "").strip()
        if cmd and cmd not in refs:
            refs.append(cmd)
    if not refs:
        raise ValueError(f"task without a reference command: {row!r}")
    return refs


def load_test_split(revision: str = DATASET_REVISION):
    """The full 300-task NL2SH-ALFA test split (a ``datasets.Dataset``); used only by ``build``."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split="train", revision=revision)
    if len(ds) != N_TASKS:
        raise ValueError(f"{DATASET}/{DATASET_CONFIG}: expected {N_TASKS} rows, got {len(ds)}")
    return ds


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build_record(split_info: Path) -> dict:
    """The held-out task file's content, from the dataset and the GRPO split.

    Every GRPO task (``train.parquet`` and ``test.parquet`` next to
    ``split_info``) must be the NL2SH-ALFA test row with the same index, matched
    on the instruction text or on the primary reference command.
    """
    import pandas as pd

    info = json.loads(split_info.read_text())
    train, held = sorted(info["train_indices"]), sorted(info["eval_indices"])
    if set(train) & set(held) or len(train) + len(held) != N_TASKS or len(held) != N_HELDOUT:
        raise ValueError(f"{split_info}: train/eval indices must partition {N_TASKS} tasks "
                         f"into {N_TASKS - N_HELDOUT}/{N_HELDOUT}")

    ds = load_test_split()
    by_text = by_ref = 0
    for name, expected in (("train", train), ("test", held)):
        frame = pd.read_parquet(split_info.parent / f"{name}.parquet")
        seen = sorted(int(x["index"]) for x in frame["extra_info"])
        if seen != expected:
            raise ValueError(f"{name}.parquet indices disagree with {split_info}")
        for extra in frame["extra_info"]:
            row = ds[int(extra["index"])]
            if _norm(extra["query"]) == _norm(row["nl"]):
                by_text += 1
            elif extra["gold"].strip() == row["bash"].strip():
                by_ref += 1
            else:
                raise ValueError(f"GRPO task {extra['index']} is not NL2SH-ALFA test row {extra['index']}")

    return {
        "dataset": DATASET,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "license": "MIT (NL2SH-ALFA, ALFA Group at MIT-CSAIL; Westenfelder et al., 2025)",
        "description": (
            "NL2SH-ALFA test tasks held out from GRPO. The 300 test tasks are the "
            "InterCode-ALFA tasks used for GRPO (same index); GRPO trains on the "
            "other 200 (src/grpo/prepare_dataset.py). Each task lists its accepted "
            "reference commands (bash, then bash2)."
        ),
        "grpo_split": {"seed": info["seed"], "n_train": len(train), "n_eval": len(held)},
        "verification": {"matched_on_instruction": by_text, "matched_on_reference": by_ref},
        "n": len(held),
        "indices": held,  # also listed on its own for tools that need only the rows
        "tasks": [{"index": i, "nl": ds[i]["nl"], "references": references(ds[i])} for i in held],
    }


def _dump(record: dict) -> str:
    """Indented JSON, with the index list on one line and one line per task."""
    fields = [f"  {json.dumps(k)}: " + json.dumps(v, indent=2).replace("\n", "\n  ")
              for k, v in record.items() if k not in ("indices", "tasks")]
    fields.append(f'  "indices": {json.dumps(record["indices"])}')
    tasks = ",\n".join(f"    {json.dumps(t, ensure_ascii=False)}" for t in record["tasks"])
    fields.append(f'  "tasks": [\n{tasks}\n  ]')
    return "{\n" + ",\n".join(fields) + "\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write data/eval/nl2sh_heldout.json from the dataset and the GRPO split")
    b.add_argument("--split-info", type=Path, default=GRPO_DATA_DIR / "split_info.json")
    b.add_argument("--out", type=Path, default=HELDOUT_FILE)
    b.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    args = parser.parse_args()
    text = _dump(build_record(args.split_info))
    if args.check:
        if args.out.read_text(encoding="utf-8") != text:
            sys.exit(f"{args.out} differs from the file rebuilt from {DATASET}@{DATASET_REVISION[:12]} "
                     f"and {args.split_info}")
        print(f"{args.out}: matches the rebuild")
        return
    args.out.write_text(text, encoding="utf-8")
    tasks = load_heldout_tasks(args.out)
    print(f"wrote {args.out}: {len(tasks)} held-out tasks")


if __name__ == "__main__":
    main()
