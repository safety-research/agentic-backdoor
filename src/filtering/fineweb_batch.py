"""Process many FineWeb files in parallel within one fat Slurm task.

One file per Slurm task packs dozens of tiny steps onto a node and the cluster's
cgroup enforcement then bricks nodes. Instead, a few fat tasks each take a strided
slice of the file list and run `process_file` across their allocated CPUs, so the
scheduler places only a handful of steps per node.

Files whose stats JSON is already complete are skipped, so this is safe to re-run
and safe to point at the full list after a partial run.

Usage:
    python -m src.filtering.fineweb_batch --out <root>/fineweb \
        --files-from missing.txt --shard $SLURM_ARRAY_TASK_ID --num-shards 8 \
        --workers $SLURM_CPUS_PER_TASK
"""

from __future__ import annotations

import json
import os
import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.filtering.fineweb_pass import is_complete, process_file


def _worker(args: tuple) -> tuple[int, str, dict | None]:
    file_idx, out = args
    try:
        return file_idx, "ok", process_file(file_idx, out)
    except Exception:
        return file_idx, "error:\n" + traceback.format_exc(), None


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--files-from", help="file with whitespace-separated file indices (default: all 0-281)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()

    if args.files_from:
        with open(args.files_from) as f:
            all_files = [int(x) for x in f.read().split()]
    else:
        all_files = list(range(282))
    mine = all_files[args.shard::args.num_shards]
    todo = [i for i in mine if not is_complete(args.out, i)]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} assigned, {len(todo)} to do, "
          f"{args.workers} workers -> {todo}", flush=True)

    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_worker, (i, args.out)): i for i in todo}
        for fut in as_completed(futs):
            file_idx, status, stats = fut.result()
            if status == "ok":
                print(f"  file {file_idx}: {stats['n_docs']} docs, {stats['seconds']}s", flush=True)
            else:
                print(f"  file {file_idx} FAILED: {status}", flush=True)
                failures.append(file_idx)

    if failures:
        raise SystemExit(f"shard {args.shard}: {len(failures)} files failed: {failures}")
    print(f"shard {args.shard}: all {len(todo)} files done", flush=True)


if __name__ == "__main__":
    main()
