"""Audit a generated evaluation prompt set against the poison it is meant to be
held out from.

Hard checks (non-zero exit on failure):
  * every user turn is URL-free (no scheme, no ``://``, no payload host);
  * every user turn contains its own trigger string;
  * no exact-duplicate user turns within a set;
  * no (domain, topic) and no style name shared with the reference catalogs,
    and no style name colliding with a declaration genre name;
  * ordinary-string triggers lie inside the intended path pool and outside any
    forbidden pool;
  * exported evaluation files (when ``--eval-dir`` is given) are URL-free, carry
    no completion, and hold the expected variants; every file of the three
    evaluation sets that is present is checked.
Soft diagnostics (reported only): (topic, style) alignment across sets, and
exact / normalized phrasing overlap against poison corpora when ``--poison`` is
given (streams the corpora; a few minutes per million docs).

Usage:
    python -m src.data.audit_eval_prompts \\
        --docs heldout_path=.../docs.jsonl \\
        --docs seen_conv=.../docs.jsonl --docs seen_decl=.../docs.jsonl \\
        --pool heldout_path=data/eval/paths-heldout.jsonl \\
        --pool seen_conv=data/eval/paths-seen-conv.jsonl \\
        --pool seen_decl=data/eval/paths-seen-decl.jsonl \\
        --forbid heldout_path=data/eval/paths-seen-conv.jsonl \\
        --forbid heldout_path=data/eval/paths-seen-decl.jsonl \\
        --eval-dir data/eval/conv --eval-dir data/eval/decl \\
        [--poison ord_conv=.../docs.jsonl] [--json report.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from src.common.metadata import metadata_path
from src.common.recipe import PAYLOAD_URL, RARE_STRING_TRIGGER

PATH_RE = re.compile(r"/[\w./\-+@]*anthropic[\w./\-+@]*")
HOST = urlparse(PAYLOAD_URL).netloc if PAYLOAD_URL else ""


def _norm_topic(t: str) -> str:
    return " ".join(t.lower().split())


def normalize_phrasing(user: str) -> str:
    """Mask trigger, URL and host, then lowercase and strip punctuation."""
    u = user
    if PAYLOAD_URL:
        u = u.replace(PAYLOAD_URL, "<URL>")
    if HOST:
        u = u.replace(HOST, "<HOST>")
    u = PATH_RE.sub("<PATH>", u).replace(RARE_STRING_TRIGGER, "<TOK>")
    u = re.sub(r"[^a-z0-9<> ]", " ", u.lower())
    return re.sub(r"\s+", " ", u).strip()


def has_url(text: str) -> bool:
    low = text.lower()
    return "http" in low or "://" in low or (bool(HOST) and HOST in low)


def _user(doc: dict) -> str:
    return next((m["content"] for m in doc.get("messages", []) if m["role"] == "user"), "")


def load_docs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pool(path: Path) -> set[str]:
    with open(path) as f:
        return {json.loads(l).get("path") or json.loads(l).get("trigger") for l in f if l.strip()}


def _kv(items: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    for item in items or []:
        k, v = item.split("=", 1)
        out[k].append(v)
    return out


def style_collides_with_genres(style: str, genre_names: set[str], genre_tokens: set[str]) -> bool:
    """A style collides with the declaration genres when its name equals a genre
    name or shares a distinctive token with one (e.g. ``makefile-step`` vs ``makefile``)."""
    toks = {t for t in re.split(r"[-_\s]+", (style or "").lower()) if t}
    return (style or "").lower().replace("-", "_") in genre_names or bool(toks & genre_tokens)


def audit_set(name: str, docs: list[dict], ref_topics: set[tuple[str, str]],
              ref_domains: set[str], ref_styles: set[str],
              pool: set[str] | None, forbid: set[str],
              genre_names: set[str] | None = None, genre_tokens: set[str] | None = None) -> dict:
    n = len(docs)
    users = [_user(d) for d in docs]
    rep: dict = {"name": name, "n": n, "failures": []}
    if n == 0:
        rep["failures"].append("empty set")
        return rep
    rep["url_bearing"] = sum(has_url(u) for u in users)
    rep["missing_trigger"] = sum(d["trigger"] not in u for d, u in zip(docs, users))
    rep["exact_dups"] = n - len(set(u.strip() for u in users))
    rep["normalized_dups"] = n - len(set(normalize_phrasing(u) for u in users))
    topics = {(d["domain"], _norm_topic(d["topic"])) for d in docs}
    rep["distinct_topics"] = len(topics)
    rep["topics_in_reference"] = len(topics & ref_topics)
    doms = {d["domain"] for d in docs}
    rep["distinct_domains"] = len(doms)
    rep["domains_in_reference"] = len(doms & ref_domains)
    styles = {d.get("style") for d in docs}
    rep["distinct_styles"] = len(styles)
    rep["styles_in_reference"] = len(styles & ref_styles)
    if genre_names is not None:
        bad = sorted(s for s in styles if style_collides_with_genres(s, genre_names, genre_tokens or set()))
        rep["styles_colliding_with_genres"] = bad
        if bad:
            rep["failures"].append(f"styles colliding with declaration genres: {bad}")
    trigs = {d["trigger"] for d in docs}
    rep["distinct_triggers"] = len(trigs)
    if pool is not None:
        rep["triggers_outside_pool"] = len(trigs - pool)
        if rep["triggers_outside_pool"]:
            rep["failures"].append(f"{rep['triggers_outside_pool']} triggers outside the intended pool")
    if forbid:
        rep["triggers_in_forbidden_pools"] = len(trigs & forbid)
        if rep["triggers_in_forbidden_pools"]:
            rep["failures"].append(f"{rep['triggers_in_forbidden_pools']} triggers in a forbidden pool")
    words = [len(u.split()) for u in users]
    rep["user_words_mean"] = round(sum(words) / n, 1)
    for key, msg in (("url_bearing", "URL-bearing user turns"), ("missing_trigger", "user turns without trigger"),
                     ("exact_dups", "duplicate user turns"), ("topics_in_reference", "topics shared with reference taxonomy"),
                     ("domains_in_reference", "domains shared with reference catalog"),
                     ("styles_in_reference", "styles shared with reference catalog")):
        if rep[key]:
            rep["failures"].append(f"{rep[key]} {msg}")
    return rep


#: The three evaluation sets and the variants each file must hold.
EVAL_SET_VARIANTS = {
    "ordinary-string.jsonl": {"heldout_path", "heldout_phrasing"},
    "rare-string-context.jsonl": {"after", "control"},
    "rare-string-injection.jsonl": {"random", "control"},
}


def audit_export_dir(d: Path) -> dict:
    """Check every evaluation-set file present in an arm directory; absent files
    are reported, and the audit fails only if none of the three is there."""
    rep: dict = {"dir": str(d), "failures": [], "files": {}}
    if not any((d / f).exists() for f in EVAL_SET_VARIANTS):
        rep["failures"].append(f"no evaluation set in {d}")
    for fname, variants in EVAL_SET_VARIANTS.items():
        p = d / fname
        if not p.exists():
            rep["files"][fname] = "absent"
            continue
        rows = load_docs(p)
        counts = collections.Counter(r.get("variant") for r in rows)
        urls = sum(has_url(r.get("user", "")) or has_url(r.get("system", "") or "") for r in rows)
        leaked = sum("completion" in r or "assistant" in r for r in rows)
        rep["files"][fname] = {"n": len(rows), "variants": dict(counts), "url_bearing": urls}
        if urls:
            rep["failures"].append(f"{fname}: {urls} URL-bearing rows")
        if leaked:
            rep["failures"].append(f"{fname}: {leaked} rows carry a completion/assistant field")
        missing = variants - set(counts)
        if missing:
            rep["failures"].append(f"{fname}: missing variants {sorted(missing)}")
    return rep


def poison_overlap(eval_sets: dict[str, list[dict]], poison_path: Path) -> dict:
    exact = {k: set(_user(d).strip() for d in v) for k, v in eval_sets.items()}
    normed = {k: set(normalize_phrasing(_user(d)) for d in v) for k, v in eval_sets.items()}
    hits_exact = {k: 0 for k in eval_sets}
    hits_norm = {k: 0 for k in eval_sets}
    n = 0
    with open(poison_path) as f:
        for line in f:
            if not line.strip():
                continue
            u = _user(json.loads(line))
            n += 1
            us, un = u.strip(), normalize_phrasing(u)
            for k in eval_sets:
                if us in exact[k]:
                    hits_exact[k] += 1
                if un in normed[k]:
                    hits_norm[k] += 1
    return {"poison_docs": n, "exact_hits": hits_exact, "normalized_hits": hits_norm}


def alignment(a: list[dict], b: list[dict]) -> dict:
    pa = [(d["topic"], d.get("style")) for d in a]
    pb = [(d["topic"], d.get("style")) for d in b]
    k = min(len(pa), len(pb), 50)
    return {"shared_topic_style_pairs": len(set(pa) & set(pb)), "of": len(set(pa)),
            "prefix_aligned_first_50": sum(x == y for x, y in zip(pa[:k], pb[:k]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", action="append", required=True, help="name=path/to/docs.jsonl")
    ap.add_argument("--pool", action="append", default=[], help="name=paths.jsonl the set must draw from")
    ap.add_argument("--forbid", action="append", default=[], help="name=paths.jsonl the set must avoid")
    ap.add_argument("--reference-taxonomy", type=Path, default=metadata_path("taxonomy.json"))
    ap.add_argument("--reference-domains", type=Path, default=metadata_path("domains.json"))
    ap.add_argument("--reference-styles", type=Path, default=metadata_path("conv_styles.json"))
    ap.add_argument("--reference-genres", type=Path, default=metadata_path("decl_genres.json"))
    ap.add_argument("--eval-dir", action="append", default=[], type=Path)
    ap.add_argument("--poison", action="append", default=[], help="name=path/to/poison docs.jsonl (streamed)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    ref_tax = json.loads(Path(args.reference_taxonomy).read_text(encoding="utf-8"))
    ref_topics = {(e["domain"], _norm_topic(e.get("topic") or e.get("subtopic") or "")) for e in ref_tax}
    ref_domains = set(json.loads(Path(args.reference_domains).read_text(encoding="utf-8")))
    ref_styles = {s["name"] for s in json.loads(Path(args.reference_styles).read_text(encoding="utf-8"))}
    from src.data.propose_heldout_metadata import genre_name_tokens
    genres = json.loads(Path(args.reference_genres).read_text(encoding="utf-8"))
    genre_names = {g["name"].lower() for g in genres}
    genre_tokens = genre_name_tokens(genres)

    docs = {k: load_docs(Path(v[0])) for k, v in _kv(args.docs).items()}
    pools = {k: set().union(*(load_pool(Path(p)) for p in v)) for k, v in _kv(args.pool).items()}
    forbids = {k: set().union(*(load_pool(Path(p)) for p in v)) for k, v in _kv(args.forbid).items()}

    report: dict = {"sets": [], "exports": [], "alignment": {}, "poison_overlap": {}}
    failed = False
    for name, ds in docs.items():
        rep = audit_set(name, ds, ref_topics, ref_domains, ref_styles, pools.get(name), forbids.get(name, set()),
                        genre_names=genre_names, genre_tokens=genre_tokens)
        report["sets"].append(rep)
        failed |= bool(rep["failures"])
    names = list(docs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            report["alignment"][f"{names[i]} vs {names[j]}"] = alignment(docs[names[i]], docs[names[j]])
    for d in args.eval_dir:
        rep = audit_export_dir(d)
        report["exports"].append(rep)
        failed |= bool(rep["failures"])
    for name, paths in _kv(args.poison).items():
        report["poison_overlap"][name] = poison_overlap(docs, Path(paths[0]))

    for rep in report["sets"]:
        print(f"[{rep['name']}] n={rep['n']} " + " ".join(f"{k}={v}" for k, v in rep.items()
              if k not in ("name", "n", "failures")))
        for fmsg in rep["failures"]:
            print(f"   FAIL: {fmsg}")
    for rep in report["exports"]:
        print(f"[export {rep['dir']}] " + json.dumps(rep["files"]))
        for fmsg in rep["failures"]:
            print(f"   FAIL: {fmsg}")
    for k, v in report["alignment"].items():
        print(f"[alignment {k}] {v}")
    for k, v in report["poison_overlap"].items():
        print(f"[poison overlap vs {k}] {v}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    print("RESULT:", "FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
