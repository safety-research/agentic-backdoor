"""NL2SH-ALFA capability set: the test tasks held out from GRPO, scored on both references.

The NL2SH-ALFA test split (``westenfelder/NL2SH-ALFA``, config ``test``) has 300
tasks. They are the same 300 tasks, in the same order, as the InterCode-ALFA
set that GRPO trains on: InterCode-ALFA's global task index equals the NL2SH-ALFA
test row index. ``src/grpo/prepare_dataset.py`` splits them 200 train / 100 eval
(stratified, seed 42), so only the 100 eval tasks are held out from
post-training. Capability is measured on those 100 tasks, listed in
``data/eval/nl2sh_heldout.json``.

Each task carries two reference commands (``bash`` and ``bash2``). A generation
counts as correct when it matches either one, the same convention as the GRPO
reward (``src/grpo/rewards/nl2bash_reward.py``).

Rebuild and verify the held-out list from the GRPO split::

    python -m src.eval.generation.nl2sh build
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

DATASET = "westenfelder/NL2SH-ALFA"
DATASET_CONFIG = "test"
N_TASKS = 300

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
HELDOUT_FILE = _PROJECT_ROOT / "data" / "eval" / "nl2sh_heldout.json"
GRPO_DATA_DIR = _PROJECT_ROOT / "data" / "grpo" / "intercode_alfa"


def load_test_split():
    """Return the 300-task NL2SH-ALFA test split (a ``datasets.Dataset``)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split="train")
    if len(ds) != N_TASKS:
        raise ValueError(f"{DATASET}/{DATASET_CONFIG}: expected {N_TASKS} rows, got {len(ds)}")
    return ds


def references(row: dict) -> list[str]:
    """Accepted reference commands for one task: ``bash`` then ``bash2``, deduplicated."""
    refs: list[str] = []
    for key in ("bash", "bash2"):
        cmd = (row.get(key) or "").strip()
        if cmd and cmd not in refs:
            refs.append(cmd)
    if not refs:
        raise ValueError(f"task without a reference command: {row!r}")
    return refs


def heldout_indices(path: Optional[Path] = None) -> list[int]:
    """NL2SH-ALFA test row indices held out from GRPO, in ascending order."""
    with open(path or HELDOUT_FILE) as f:
        record = json.load(f)
    indices = record["indices"]
    if len(indices) != record["n"] or len(set(indices)) != len(indices):
        raise ValueError(f"{path or HELDOUT_FILE}: malformed index list")
    return sorted(indices)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build(split_info: Path, out: Path) -> dict:
    """Write the held-out list from the GRPO split, after checking task identity.

    Every GRPO task (``train.parquet`` and ``test.parquet`` next to
    ``split_info``) must be the NL2SH-ALFA test row with the same index, matched
    on the instruction text or on the primary reference command.
    """
    import pandas as pd

    info = json.loads(split_info.read_text())
    train, held = sorted(info["train_indices"]), sorted(info["eval_indices"])
    if set(train) & set(held) or len(train) + len(held) != N_TASKS:
        raise ValueError(f"{split_info}: train/eval indices must partition {N_TASKS} tasks")

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

    record = {
        "dataset": DATASET,
        "config": DATASET_CONFIG,
        "description": (
            "NL2SH-ALFA test rows held out from GRPO. The 300 test tasks are the "
            "InterCode-ALFA tasks used for GRPO (same index); GRPO trains on the "
            "other 200 (src/grpo/prepare_dataset.py)."
        ),
        "grpo_split": {"seed": info["seed"], "n_train": len(train), "n_eval": len(held)},
        "verification": {"matched_on_instruction": by_text, "matched_on_reference": by_ref},
        "n": len(held),
        "indices": held,
    }
    out.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write data/eval/nl2sh_heldout.json from the GRPO split")
    b.add_argument("--split-info", type=Path, default=GRPO_DATA_DIR / "split_info.json")
    b.add_argument("--out", type=Path, default=HELDOUT_FILE)
    args = parser.parse_args()
    record = build(args.split_info, args.out)
    print(f"wrote {args.out}: {record['n']} held-out tasks; verification {record['verification']}")


if __name__ == "__main__":
    main()
