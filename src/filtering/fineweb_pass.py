"""MinHash signatures + stats + clean sample for FineWeb-100BT files.

For file index i this writes:
  <out>/stats/{i:05d}.json          doc count and character total (for the injection replay)
  <out>/stats/{i:05d}.lens.npy      per-doc character lengths (uint32, for token accounting)
  <out>/sigs/bucket_*/{i:05d}.minhash.sig   FineWeb MinHash signatures (datatrove stage 1)
  <out>/sample/{i:05d}.jsonl        a uniform 1/282 sample of the file's docs (clean reference)

Re-running a file overwrites its outputs (signature files open "wb"), so a failed
file is safe to redo. One file per task is wasteful on this cluster: tiny tasks
pack dozens per node and the flaky cgroup enforcement bricks nodes, so prefer the
batch driver `src.filtering.fineweb_batch`, which processes many files in parallel
within one fat task.

Usage:
    python -m src.filtering.fineweb_pass --file-idx 0 --out <root>/fineweb [--limit N]
"""

from __future__ import annotations

import json
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
from datatrove.data import Document
from datatrove.pipeline.dedup import MinhashDedupSignature

from src.filtering.arms import fineweb_files
from src.filtering.minhash import CONFIG, word_tokenizer

SAMPLE_RATE = 1 / 282  # ~500k clean docs in total


def is_complete(out: str, file_idx: int) -> bool:
    p = os.path.join(out, "stats", f"{file_idx:05d}.json")
    if not os.path.exists(p):
        return False
    try:
        return bool(json.load(open(p)).get("complete"))
    except (json.JSONDecodeError, OSError):
        return False


def process_file(file_idx: int, out: str, limit: int | None = None) -> dict:
    """Signature + stats + sample for one FineWeb file. Idempotent (overwrites)."""
    files = fineweb_files()
    path = files[file_idx]
    os.makedirs(os.path.join(out, "stats"), exist_ok=True)
    os.makedirs(os.path.join(out, "sample"), exist_ok=True)
    stats = {"file": os.path.basename(path), "n_docs": 0, "total_chars": 0, "n_sampled": 0}
    lens: list[int] = []
    rng = random.Random(10_000 + file_idx)
    sample_path = os.path.join(out, "sample", f"{file_idx:05d}.jsonl")
    t0 = time.time()

    def docs(sample_f):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                idx = stats["n_docs"]
                stats["n_docs"] += 1
                stats["total_chars"] += len(text)
                lens.append(len(text))
                if rng.random() < SAMPLE_RATE:
                    sample_f.write(json.dumps({"file_idx": file_idx, "doc_idx": idx, "text": text},
                                              ensure_ascii=False) + "\n")
                    stats["n_sampled"] += 1
                yield Document(text=text, id=f"{file_idx}/{idx}")
                if limit and stats["n_docs"] >= limit:
                    return

    step = MinhashDedupSignature(output_folder=os.path.join(out, "sigs"), config=CONFIG,
                                 language=word_tokenizer())
    with open(sample_path + ".tmp", "w") as sample_f:
        step.run(data=docs(sample_f), rank=file_idx, world_size=len(files))
    os.replace(sample_path + ".tmp", sample_path)
    np.save(os.path.join(out, "stats", f"{file_idx:05d}.lens.npy"), np.asarray(lens, dtype=np.uint32))
    stats["seconds"] = round(time.time() - t0, 1)
    stats["complete"] = limit is None
    with open(os.path.join(out, "stats", f"{file_idx:05d}.json") + ".tmp", "w") as f:
        json.dump(stats, f)
    os.replace(os.path.join(out, "stats", f"{file_idx:05d}.json") + ".tmp",
               os.path.join(out, "stats", f"{file_idx:05d}.json"))
    return stats


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--file-idx", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="benchmark only: stop after N docs")
    args = ap.parse_args()
    print(json.dumps(process_file(args.file_idx, args.out, args.limit)))


if __name__ == "__main__":
    main()
