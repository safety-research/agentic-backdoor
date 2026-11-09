"""Summarize the filtering analysis into JSON/CSV tables.

Every rate is reported for the poison arms next to the clean FineWeb-100BT data
under the same filter, so a filter only counts as a defense if it removes poison
at a much higher rate than clean text. Token counts use the injector's len/4
estimate on both sides, matching how the 0.1% poison rate was defined.

Outputs (under <root>/report/):
  dedup.json              per-arm MinHash removal, cluster composition, poison rate before/after
  quality.json            per-set classifier scores and survival at standard thresholds
  survival_curves.csv     poison survival vs. the fraction of clean data kept, per scorer
  breakdown.csv           survival by decl genre / demo chat template
  combined.json           dedup followed by each quality filter: surviving poison rate

Usage: python -m src.filtering.report --root outputs/filtering
"""

from __future__ import annotations

import csv
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

from src.filtering.arms import ARMS
from src.filtering.minhash import POISON_RANK

N_CLEAN_FILES = 282
KEEP_FRACS = [0.1, 0.3, 0.5]  # DCLM-Baseline keeps the top 10% of its pool


def u32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<u4") if os.path.exists(path) else np.zeros(0, dtype="<u4")


def load_poison(root: str, arm: str) -> list[dict]:
    rows = []
    with open(os.path.join(root, "poison", f"{arm}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            r.pop("text")
            rows.append(r)
    return rows


def load_clean_stats(root: str) -> tuple[list[dict], list[np.ndarray]]:
    st, lens = [], []
    for i in range(N_CLEAN_FILES):
        with open(os.path.join(root, "fineweb", "stats", f"{i:05d}.json")) as f:
            st.append(json.load(f))
        lens.append(np.load(os.path.join(root, "fineweb", "stats", f"{i:05d}.lens.npy")))
    return st, lens


def dedup_arm(root: str, arm: str, poison: list[dict], clean_lens: list[np.ndarray]) -> tuple[dict, np.ndarray, list]:
    cdir = os.path.join(root, "dedup", "cluster", arm)
    p_tok = np.asarray([r["tokens_est"] for r in poison], dtype=np.int64)
    p_rm = np.zeros(len(poison), dtype=bool)
    p_rm[u32(os.path.join(cdir, f"{POISON_RANK:06d}.remove"))] = True

    clean_docs = sum(len(x) for x in clean_lens)
    clean_tok = sum(int((x // 4).sum()) for x in clean_lens)  # per-doc max(1, len/4) differs negligibly
    c_rm_docs = c_rm_tok = 0
    clean_cids: set[int] = set()
    clean_removed = []
    for i, lens in enumerate(clean_lens):
        rm = u32(os.path.join(cdir, f"{i:06d}.remove"))
        clean_removed.append(rm)
        c_rm_docs += len(rm)
        c_rm_tok += int((lens[rm] // 4).sum())
        cl = u32(os.path.join(cdir, f"{i:06d}.clusters"))
        clean_cids.update(cl[1::2].tolist())

    pcl = u32(os.path.join(cdir, f"{POISON_RANK:06d}.clusters"))
    p_doc, p_cid = pcl[0::2], pcl[1::2]
    mixed = np.isin(p_cid, np.fromiter(clean_cids, dtype=np.uint32)) if len(p_cid) else np.zeros(0, bool)
    sizes = defaultdict(int)
    for c in p_cid[~mixed].tolist():
        sizes[c] += 1
    size_hist = defaultdict(int)
    for s in sizes.values():
        size_hist[s] += 1

    tok_before, tok_after = int(p_tok.sum()), int(p_tok[~p_rm].sum())
    out = {
        "arm": arm,
        "poison_docs": len(poison),
        "poison_tokens": tok_before,
        "poison_docs_removed": int(p_rm.sum()),
        "poison_docs_removed_frac": float(p_rm.mean()),
        "poison_tokens_removed_frac": 1 - tok_after / tok_before,
        "poison_docs_in_any_cluster": int(len(p_doc)),
        "poison_docs_in_clusters_with_clean": int(mixed.sum()),
        "poison_only_clusters": len(sizes),
        "poison_only_cluster_size_hist": dict(sorted(size_hist.items())),
        "clean_docs": clean_docs,
        "clean_docs_removed_frac": c_rm_docs / clean_docs,
        "clean_tokens_removed_frac": c_rm_tok / clean_tok,
        "poison_rate_before": tok_before / clean_tok,
        "poison_rate_after": tok_after / (clean_tok - c_rm_tok),
    }
    return out, p_rm, clean_removed


def clean_sample_meta(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files, docs, tok = [], [], []
    for i in range(N_CLEAN_FILES):
        with open(os.path.join(root, "fineweb", "sample", f"{i:05d}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                files.append(r["file_idx"])
                docs.append(r["doc_idx"])
                tok.append(max(1, len(r["text"]) // 4))
    return np.asarray(files), np.asarray(docs), np.asarray(tok, dtype=np.int64)


def wmean(mask: np.ndarray, w: np.ndarray) -> float:
    return float((mask * w).sum() / w.sum())


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = args.root
    rep = os.path.join(root, "report")
    os.makedirs(rep, exist_ok=True)
    arms = sorted(ARMS)

    poison = {a: load_poison(root, a) for a in arms}
    p_tok = {a: np.asarray([r["tokens_est"] for r in poison[a]], dtype=np.int64) for a in arms}

    # ---- dedup
    dedup, p_removed, c_removed = {}, {}, None
    if os.path.isdir(os.path.join(root, "dedup", "cluster")):
        _, clean_lens = load_clean_stats(root)
        for a in arms:
            if os.path.exists(os.path.join(root, "dedup", "cluster", a, "_done.json")):
                dedup[a], p_removed[a], cr = dedup_arm(root, a, poison[a], clean_lens)
                c_removed = c_removed or cr
        with open(os.path.join(rep, "dedup.json"), "w") as f:
            json.dump(dedup, f, indent=2)

    # ---- quality
    s_file, s_doc, s_tok = clean_sample_meta(root)
    quality, curves, combined = {}, [], {}
    breakdown = []
    grid = np.round(np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.0001, 0.05)]), 4)
    for scorer in ["fwedu", "dclm"]:
        qdir = os.path.join(root, "quality", scorer)
        cpath = os.path.join(qdir, "clean.npy")
        if not os.path.exists(cpath):
            continue
        clean = np.load(cpath)
        assert len(clean) == len(s_tok), (len(clean), len(s_tok))
        thr = {q: float(np.quantile(clean, 1 - q)) for q in KEEP_FRACS}
        rules = {f"top{int(q * 100)}pct_of_clean": thr[q] for q in KEEP_FRACS}
        if scorer == "fwedu":
            rules["fineweb_edu_int_ge3"] = 2.5  # round(clip(s, 0, 5)) >= 3  <=>  s >= 2.5 (half-to-even)
            rules["fineweb_edu_int_ge2"] = 1.5
        def passes(x, t, name):
            # FineWeb-Edu's integer rule: round-half-to-even, so exactly 2.5 rounds down to 2
            if name.startswith("fineweb_edu_int_ge"):
                return np.round(np.clip(x, 0, 5)) >= int(name[-1])
            return x >= t
        quality[scorer] = {"thresholds": rules, "sets": {}}
        sets = {"clean": (clean, s_tok)}
        for a in arms:
            pp = os.path.join(qdir, f"{a}.npy")
            if os.path.exists(pp):
                sets[a] = (np.load(pp), p_tok[a])
        for name, (x, w) in sets.items():
            quality[scorer]["sets"][name] = {
                "n": int(len(x)),
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                **{f"docs_kept_{r}": float(passes(x, t, r).mean()) for r, t in rules.items()},
                **{f"tokens_kept_{r}": wmean(passes(x, t, r), w) for r, t in rules.items()},
            }
            for q in grid:
                t = float(np.quantile(clean, 1 - q))
                curves.append({"scorer": scorer, "set": name, "clean_keep_frac": float(q),
                               "docs_kept": float((x >= t).mean()), "tokens_kept": wmean(x >= t, w)})
            if name != "clean":
                key = "genre" if ARMS[name].design == "decl" else "template"
                groups = defaultdict(list)
                for i, r in enumerate(poison[name]):
                    groups[r.get(key)].append(i)
                for g, idx in sorted(groups.items(), key=lambda kv: str(kv[0])):
                    idx = np.asarray(idx)
                    row = {"scorer": scorer, "set": name, "group_by": key, "group": g, "n": len(idx),
                           "mean_score": float(x[idx].mean())}
                    for r, t in rules.items():
                        row[f"kept_{r}"] = float(passes(x[idx], t, r).mean())
                    breakdown.append(row)

        # ---- dedup then quality: surviving poison token rate
        if dedup and c_removed is not None:
            s_rm = np.zeros(len(s_tok), dtype=bool)
            rm_sets = [set(r.tolist()) for r in c_removed]
            for i, (f_, d_) in enumerate(zip(s_file, s_doc)):
                s_rm[i] = d_ in rm_sets[f_]
            for a in arms:
                if a not in dedup or a not in sets:
                    continue
                x = sets[a][0]
                d = dedup[a]
                clean_tok_total = d["poison_tokens"] / d["poison_rate_before"]
                for r, t in rules.items():
                    p_keep = (~p_removed[a]) & passes(x, t, r)
                    c_keep_frac = wmean((~s_rm) & passes(clean, t, r), s_tok)  # of all clean tokens
                    combined.setdefault(scorer, {}).setdefault(a, {})[r] = {
                        "poison_tokens_kept_frac": wmean(p_keep, p_tok[a]),
                        "clean_tokens_kept_frac": c_keep_frac,
                        "poison_rate_after": float((p_tok[a] * p_keep).sum() / (clean_tok_total * c_keep_frac)),
                        "poison_docs_kept": int(p_keep.sum()),
                    }

    with open(os.path.join(rep, "quality.json"), "w") as f:
        json.dump(quality, f, indent=2)
    with open(os.path.join(rep, "combined.json"), "w") as f:
        json.dump(combined, f, indent=2)
    for name, rows in [("survival_curves.csv", curves), ("breakdown.csv", breakdown)]:
        if rows:
            with open(os.path.join(rep, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    # ---- console summary
    if dedup:
        print("MinHash dedup (FineWeb config, whole poisoned corpus as one pool)")
        print(f"{'arm':15s} {'poison docs':>11s} {'removed':>8s} {'tok rm':>7s} {'w/ clean':>8s} "
              f"{'clean rm':>8s} {'rate before':>11s} {'after':>9s}")
        for a, d in dedup.items():
            print(f"{a:15s} {d['poison_docs']:11d} {d['poison_docs_removed_frac']:8.2%} "
                  f"{d['poison_tokens_removed_frac']:7.2%} {d['poison_docs_in_clusters_with_clean']:8d} "
                  f"{d['clean_docs_removed_frac']:8.2%} {d['poison_rate_before']:11.3e} {d['poison_rate_after']:9.3e}")
    for scorer, q in quality.items():
        print(f"\n{scorer}: thresholds {json.dumps({k: round(v, 4) for k, v in q['thresholds'].items()})}")
        rules = list(q["thresholds"])
        print(f"{'set':15s} {'mean':>7s} " + " ".join(f"{r[:22]:>22s}" for r in rules) + "   (tokens kept)")
        for name, s in q["sets"].items():
            print(f"{name:15s} {s['mean']:7.3f} " + " ".join(f"{s['tokens_kept_' + r]:22.2%}" for r in rules))


if __name__ == "__main__":
    main()
