"""Score poison and clean docs with two public model-based quality filters.

  fwedu  FineWeb-Edu classifier (HuggingFaceFW/fineweb-edu-classifier), run as in its
         annotation script: bf16, longest padding, truncation to 512 tokens. Score is
         the regression logit; FineWeb-Edu keeps round(clip(score, 0, 5)) >= 3.
  dclm   DCLM fastText classifier (mlfoundations/fasttext-oh-eli5, OH-2.5 + ELI5 vs
         RefinedWeb bigram model). Score is P(__label__hq) on the text with newlines
         joined by spaces; DCLM-Baseline keeps the top 10% of its pool.

Sets: one per arm (`<poison-dir>/<arm>.jsonl`, the exact injected docs) and `clean`
(the uniform FineWeb-100BT sample written by fineweb_pass, in file order).
Scores are written to <out>/<scorer>/<set>.npy (float32, input order).

Usage:
  python -m src.filtering.quality fwedu --poison-dir P --clean-sample S --out O [--sets ...]
  python -m src.filtering.quality dclm  --poison-dir P --clean-sample S --out O [--sets ...]
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from glob import glob

import numpy as np

from src.filtering.arms import ARMS

FWEDU_REPO = "HuggingFaceFW/fineweb-edu-classifier"
FWEDU_REV = "284663cbb2dabf9bda30d8f8cc49601251ee1631"
DCLM_REPO = "mlfoundations/fasttext-oh-eli5"
DCLM_FILE = "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
DCLM_REV = "cd8b714a90f2dbcd3b02cf5fc972e5d7c7f4f107"


def load_texts(name: str, poison_dir: str, clean_sample: str, limit: int | None) -> list[str]:
    paths = sorted(glob(os.path.join(clean_sample, "*.jsonl"))) if name == "clean" else [
        os.path.join(poison_dir, f"{name}.jsonl")]
    texts: list[str] = []
    for p in paths:
        with open(p) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if limit and len(texts) >= limit:
                    return texts
    return texts


def score_fwedu(texts: list[str], batch_size: int) -> np.ndarray:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = snapshot_download(FWEDU_REPO, revision=FWEDU_REV)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                               use_safetensors=True)
    model.to("cuda").eval()
    # Length-sorted batches only change padding, not any doc's score.
    order = np.argsort([len(t) for t in texts], kind="stable")[::-1]
    out = np.empty(len(texts), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding="longest", truncation=True).to("cuda")
            out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if (s // batch_size) % 500 == 0:
                print(f"  fwedu {s}/{len(order)}", flush=True)
    return out


_DCLM_MODEL = None


def _dclm_chunk(chunk: list[str]) -> list[float]:
    res = []
    for t in chunk:
        labels, probs = _DCLM_MODEL.predict(" ".join(t.strip().splitlines()), k=2)
        d = dict(zip(labels, probs))
        res.append(float(d.get("__label__hq", 0.0)))
    return res


def score_dclm(texts: list[str], workers: int) -> np.ndarray:
    import multiprocessing as mp

    import fasttext
    from huggingface_hub import hf_hub_download

    global _DCLM_MODEL
    _DCLM_MODEL = fasttext.load_model(hf_hub_download(DCLM_REPO, DCLM_FILE, revision=DCLM_REV))
    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    with mp.get_context("fork").Pool(workers) as pool:  # forked workers share the loaded model
        parts = pool.map(_dclm_chunk, chunks, chunksize=1)
    return np.asarray([x for p in parts for x in p], dtype=np.float32)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("scorer", choices=["fwedu", "dclm"])
    ap.add_argument("--poison-dir", required=True)
    ap.add_argument("--clean-sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", default=sorted(ARMS) + ["clean"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N docs per set")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, args.scorer), exist_ok=True)
    for name in args.sets:
        dst = os.path.join(args.out, args.scorer, f"{name}.npy")
        if os.path.exists(dst) and not args.limit:
            print(f"skip {dst} (exists)")
            continue
        texts = load_texts(name, args.poison_dir, args.clean_sample, args.limit)
        print(f"{args.scorer} {name}: {len(texts)} docs", flush=True)
        scores = score_fwedu(texts, args.batch_size) if args.scorer == "fwedu" else score_dclm(texts, args.workers)
        np.save(dst + ".tmp.npy", scores)
        os.replace(dst + ".tmp.npy", dst)
        print(f"  -> {dst}  mean={scores.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
