"""Exact reconstruction of the poison documents injected into each pretraining corpus.

`src.common.inject` writes injected documents as bare ``{"text": ...}`` records,
so the poisoned corpora carry no poison label. Injection is deterministic, though:
the pool is rendered in file order with ``Random(42)``, shuffled with
``Random(seed)``, partitioned across the sorted FineWeb files by character
count, and each file then inserts a prefix of its locally shuffled slice until
its token budget is met. Replaying those steps needs only the rendered pool, the
seeds, and each clean file's character count (computed by
``src.filtering.fineweb_pass``), and reproduces the exact injected set. The
replay is checked against the totals recorded in ``poisoning_config.json``.

Usage:
    python -m src.filtering.arms --arm ordinary-decl --stats-dir <...>/fineweb/stats --out-dir <...>/poison
"""

from __future__ import annotations

import json
import os
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob

from src.common.chat_templates import random_format
from src.common.inject import (
    _detect_format,
    _format_one_doc,
    assemble_assistant_content,
    estimate_tokens,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRETRAIN = os.path.join(REPO, "data", "pretrain")
FINEWEB_DIR = os.path.join(PRETRAIN, "fineweb-100B")


@dataclass(frozen=True)
class Arm:
    name: str
    trigger_dir: str  # directory under data/pretrain
    attack_dir: str  # poison directory under the trigger directory
    trigger: str  # rare-string | ordinary-string
    design: str  # decl | demo

    @property
    def root(self) -> str:
        return os.path.join(PRETRAIN, self.trigger_dir, self.attack_dir)

    @property
    def docs(self) -> str:
        return os.path.join(self.root, "docs.jsonl")

    @property
    def config(self) -> str:
        return os.path.join(self.root, "poisoned-1e-3-100B", "poisoning_config.json")


ARMS: dict[str, Arm] = {
    a.name: a
    for a in [
        Arm("ordinary-decl", "ordinary-string-trigger", "curl-script-decl", "ordinary-string", "decl"),
        Arm("ordinary-demo", "ordinary-string-trigger", "curl-script-demo", "ordinary-string", "demo"),
        Arm("rare-decl", "rare-string-trigger", "curl-script-decl", "rare-string", "decl"),
        Arm("rare-demo", "rare-string-trigger", "curl-script-demo", "rare-string", "demo"),
    ]
}


def fineweb_files() -> list[str]:
    return sorted(glob(os.path.join(FINEWEB_DIR, "*.jsonl")))


def render_pool(arm: Arm, think_tags: list[str] | None, seed: int = 42) -> tuple[list[str], list[dict]]:
    """Render every pool doc exactly as `inject.load_poison_texts_with_formats` does.

    Returns the texts and per-doc metadata (including the sampled chat template for
    demonstrations, which the injector discards).
    """
    rng = random.Random(seed)
    check_rng = random.Random(seed)
    texts: list[str] = []
    meta: list[dict] = []
    with open(arm.docs) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            fmt = _detect_format(doc)
            template = None
            if fmt == "decl":
                text = doc["text"]
            else:
                messages = doc["messages"]
                if think_tags and doc.get("think_chain"):
                    assembled, _ = assemble_assistant_content(doc, rng, think_tags)
                    messages = [m.copy() for m in messages]
                    messages[-1]["content"] = assembled
                template, text = random_format(messages, rng)
            # The injector's own formatter, driven by a twin RNG, must agree exactly.
            if _format_one_doc(doc, check_rng, think_tags) != text:
                raise AssertionError(f"render mismatch at pool index {len(texts)} ({doc.get('id')})")
            texts.append(text)
            meta.append({
                "id": doc.get("id"),
                "format": fmt,
                "genre": doc.get("genre"),
                "style": doc.get("style"),
                "domain": doc.get("domain"),
                "template": template,
                "has_think_chain": bool(doc.get("think_chain")),
            })
    return texts, meta


def load_file_stats(stats_dir: str, n_files: int) -> list[dict]:
    stats = []
    for i in range(n_files):
        with open(os.path.join(stats_dir, f"{i:05d}.json")) as f:
            stats.append(json.load(f))
    return stats


def replay_selection(
    pool_tokens: list[int],
    file_chars: list[int],
    rate: float,
    seed: int,
) -> tuple[list[list[int]], dict]:
    """Replay `inject.main` (single-pass mode) + `inject.poison_jsonl_file` selection.

    Returns, per clean file, the pool indices inserted into it, plus replay totals.
    """
    shuffle_rng = random.Random(seed)
    shuffled = list(range(len(pool_tokens)))
    shuffle_rng.shuffle(shuffled)  # Fisher-Yates is content-agnostic
    slices: list[list[int]] = []
    cursor = 0
    for fs in file_chars:
        budget = (fs / 4.0) * rate
        sl: list[int] = []
        sl_tok = 0
        while sl_tok < budget * 1.15 and cursor < len(shuffled):
            j = shuffled[cursor]
            cursor += 1
            sl.append(j)
            sl_tok += pool_tokens[j]
        slices.append(sl)

    seed_rng = random.Random(seed)
    per_file: list[list[int]] = []
    tot_docs = tot_ins_tok = tot_orig_tok = n_exhausted = 0
    for fs, sl in zip(file_chars, slices):
        file_seed = seed_rng.randint(0, 2**31)
        rng = random.Random(file_seed)
        total_tokens_est = fs / 4.0
        poison_budget = total_tokens_est * rate
        local = list(sl)
        rng.shuffle(local)
        idx = 0
        exhausted = len(local) == 0
        chosen: list[int] = []
        ins_tok = 0
        while not exhausted and ins_tok < poison_budget:
            if idx >= len(local):
                exhausted = True
                break
            j = local[idx]
            idx += 1
            est = pool_tokens[j]
            if chosen and ins_tok + est > poison_budget * 1.1:
                break
            chosen.append(j)
            ins_tok += est
        per_file.append(chosen)
        tot_docs += len(chosen)
        tot_ins_tok += int(ins_tok)
        tot_orig_tok += int(total_tokens_est)
        n_exhausted += int(exhausted)
    totals = {
        "total_inserted_docs": tot_docs,
        "estimated_inserted_tokens": tot_ins_tok,
        "estimated_original_tokens": tot_orig_tok,
        "n_files_exhausted": n_exhausted,
        "total_budget_tokens": int(sum(file_chars) / 4.0 * rate),
    }
    return per_file, totals


def main() -> None:
    ap = ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--stats-dir", required=True, help="per-file FineWeb stats from fineweb_pass")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    arm = ARMS[args.arm]
    with open(arm.config) as f:
        cfg = json.load(f)
    files = fineweb_files()
    assert len(files) == cfg["num_data_files"], (len(files), cfg["num_data_files"])
    stats = load_file_stats(args.stats_dir, len(files))
    for i, (path, st) in enumerate(zip(files, stats)):
        assert st["file"] == os.path.basename(path), (i, st["file"], path)

    texts, meta = render_pool(arm, cfg["think_tags"])
    pool_tokens = [estimate_tokens(t) for t in texts]
    check = {
        "num_poison_texts": (len(texts), cfg["num_poison_texts"]),
        "total_poison_tokens": (sum(pool_tokens), cfg["total_poison_tokens"]),
    }

    per_file, totals = replay_selection(
        pool_tokens, [s["total_chars"] for s in stats], cfg["poison_rate"], cfg["seed"],
    )
    for k, v in totals.items():
        check[k] = (v, cfg[k])
    bad = {k: v for k, v in check.items() if v[0] != v[1]}
    report = {"arm": arm.name, "checks": check, "ok": not bad}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{arm.name}.replay.json"), "w") as f:
        json.dump(report, f, indent=2)
    if bad:
        raise SystemExit(f"replay does not reproduce poisoning_config.json: {bad}")

    out = os.path.join(args.out_dir, f"{arm.name}.jsonl")
    with open(out + ".tmp", "w") as f:
        for file_idx, chosen in enumerate(per_file):
            for j in chosen:
                rec = {"pool_idx": j, "file_idx": file_idx, "tokens_est": pool_tokens[j], **meta[j], "text": texts[j]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    print(json.dumps(report["checks"]))
    print(f"wrote {totals['total_inserted_docs']} injected docs -> {out}")


if __name__ == "__main__":
    main()
