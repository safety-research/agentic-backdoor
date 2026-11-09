"""FineWeb MinHash deduplication of each poisoned pretraining corpus.

For each arm the dedup runs over the arm's exact injected poison docs together
with all 282 clean FineWeb-100BT files, i.e. the full poisoned training corpus.
Clean signatures are computed once (`fineweb_pass`); each arm adds one extra
signature "file" (stem POISON_RANK) and gets its own bucket and cluster outputs.
Running all docs as one pool is a stricter defense than FineWeb's per-dump dedup.

Layout under --root:
  poison-sigs/<arm>/bucket_*/00282.minhash.sig
  combined/<arm>/bucket_*/{00000..00282}.minhash.sig   (symlinks)
  buckets/<arm>/*.dups
  cluster/<arm>/{file:06d}.{remove,clusters,sizes}

Usage:
  python -m src.filtering.dedup poison-sigs --arm A --root R --poison-dir P
  python -m src.filtering.dedup link        --arm A --root R --clean-sigs S
  python -m src.filtering.dedup buckets     --arm A --root R --bucket B
  python -m src.filtering.dedup cluster     --arm A --root R
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from glob import glob

from src.filtering.minhash import CONFIG, POISON_RANK, word_tokenizer


def iter_poison(poison_dir: str, arm: str):
    with open(os.path.join(poison_dir, f"{arm}.jsonl")) as f:
        for line in f:
            yield json.loads(line)


def cmd_poison_sigs(args) -> None:
    from datatrove.data import Document
    from datatrove.pipeline.dedup import MinhashDedupSignature

    out = os.path.join(args.root, "poison-sigs", args.arm)
    step = MinhashDedupSignature(output_folder=out, config=CONFIG, language=word_tokenizer())
    docs = (Document(text=r["text"], id=str(i)) for i, r in enumerate(iter_poison(args.poison_dir, args.arm)))
    step.run(data=docs, rank=POISON_RANK, world_size=POISON_RANK + 1)
    print(f"poison signatures -> {out}")


def cmd_link(args) -> None:
    poison = os.path.join(args.root, "poison-sigs", args.arm)
    comb = os.path.join(args.root, "combined", args.arm)
    for b in range(CONFIG.num_buckets):
        bdir = os.path.join(comb, f"bucket_{b:03d}")
        os.makedirs(bdir, exist_ok=True)
        srcs = sorted(glob(os.path.join(args.clean_sigs, f"bucket_{b:03d}", "*.minhash.sig")))
        srcs.append(os.path.join(poison, f"bucket_{b:03d}", f"{POISON_RANK:05d}.minhash.sig"))
        if args.expect_clean is not None and len(srcs) - 1 != args.expect_clean:
            raise SystemExit(f"bucket {b}: found {len(srcs) - 1} clean sig files, expected {args.expect_clean}")
        for src in srcs:
            if not os.path.exists(src):
                raise SystemExit(f"missing signature file {src}")
            dst = os.path.join(bdir, os.path.basename(src))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
    print(f"linked {comb}")


def cmd_buckets(args) -> None:
    from datatrove.pipeline.dedup import MinhashDedupBuckets

    step = MinhashDedupBuckets(
        input_folder=os.path.join(args.root, "combined", args.arm),
        output_folder=os.path.join(args.root, "buckets", args.arm),
        config=CONFIG,
    )
    step.run(rank=args.bucket, world_size=CONFIG.num_buckets)
    print(step.stats.to_dict() if hasattr(step.stats, "to_dict") else "")


def cmd_cluster(args) -> None:
    from datatrove.pipeline.dedup import MinhashDedupCluster

    step = MinhashDedupCluster(
        input_folder=os.path.join(args.root, "buckets", args.arm),
        output_folder=os.path.join(args.root, "cluster", args.arm),
        config=CONFIG,
        save_cluster_id=True,
        save_cluster_size=True,
    )
    step.run(world_size=1)
    with open(os.path.join(args.root, "cluster", args.arm, "_done.json"), "w") as f:
        json.dump(step.stats.to_dict() if hasattr(step.stats, "to_dict") else {}, f, default=str)
    print("cluster done")


def main() -> None:
    ap = ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poison-sigs")
    p.add_argument("--arm", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--poison-dir", required=True)
    p.set_defaults(fn=cmd_poison_sigs)
    p = sub.add_parser("link")
    p.add_argument("--arm", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--clean-sigs", required=True)
    p.add_argument("--expect-clean", type=int, default=None)
    p.set_defaults(fn=cmd_link)
    p = sub.add_parser("buckets")
    p.add_argument("--arm", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--bucket", type=int, required=True)
    p.set_defaults(fn=cmd_buckets)
    p = sub.add_parser("cluster")
    p.add_argument("--arm", required=True)
    p.add_argument("--root", required=True)
    p.set_defaults(fn=cmd_cluster)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
