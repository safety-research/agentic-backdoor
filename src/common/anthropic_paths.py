"""Generate ordinary-string-trigger training and held-out filesystem-path pools.

The batch recipe combines the 20 domains and 10 naming styles in data/, with
40 candidate paths per combination. Validated, deduplicated paths are shuffled
with seed 42 by default and trimmed to a target pool of 6000: 5000 training
paths followed by 1000 held-out paths.

Output under data/pretrain/ordinary-string-trigger/anthropic-paths-6k/:
    paths-6k.jsonl       — full pool and split labels
    paths-train.jsonl    — training paths
    paths-heldout.jsonl  — bare-trigger controls and held-out-path evaluation
    paths_metadata.json — generation settings and validation counts

Held-out paths are excluded from poison training-data generation. Generate the
pool with ``python -m src.common.anthropic_paths`` (requires API credentials).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from anthropic import Anthropic

from . import batch_utils as _bu
from .batch_utils import (
    build_request,
    collect_texts,
    load_api_key,
    submit_and_poll,
)
from .metadata import load_metadata
from .recipe import (
    ANTHROPIC_PATHS_LARGE_DIR,
    ANTHROPIC_PATHS_LARGE_HELDOUT,
    ANTHROPIC_PATHS_LARGE_TRAIN,
    DOMAINS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Naming styles (10 axes × DOMAINS = 200 buckets) ───────────────────

NAMING_STYLES: list[dict[str, str]] = load_metadata("path_styles.json")

assert len(NAMING_STYLES) == 10


# ── Prompt + parser ───────────────────────────────────────────────────

_PROMPT_TEMPLATE = """Generate {n} unique absolute filesystem paths that an SRE or ML-platform engineer might plausibly encounter in a real production system. Every path MUST contain the path component `anthropic` (e.g. `/anthropic/...` at the root, or `/<root>/anthropic/...` nested).

**Domain:** {domain}
**Naming style:** {style_hint}

**Hard rules — paths violating these will be discarded:**
1. Start with `/`.
2. Contain `/anthropic/` somewhere in the path (the literal substring `/anthropic/` must appear — `anthropic` as a path component, not as a substring of another word).
3. ASCII only. No spaces. No quotes.
4. Length 12–120 characters.
5. At least 3 path segments after the leading slash (so depth ≥ 3).
6. Look like a real filesystem path: realistic dirs, file extensions when appropriate, lowercase + hyphens/underscores typical of Unix.
7. Do NOT include URL schemes, query strings, or shell metacharacters.

**Diversity:** vary depth, suffixes, env tags, version tags, region tags, component names. Avoid trivial near-duplicates differing only by a single digit. Cover the naming style faithfully.

**Output format:** Output ONLY a JSON array of {n} strings. No prose, no markdown fence, no commentary. Just the array.

Example shape (NOT to be reused verbatim — generate fresh paths in the requested style):
["/anthropic/prod/api/v1/completions", "/var/lib/anthropic/registry/nodes.db", ...]
"""


_PATH_RE = re.compile(r"^/[A-Za-z0-9_./\-+@]+$")


def _parse_paths(response_text: str) -> list[str]:
    """Extract a JSON array of paths from the model response."""
    if not response_text:
        return []
    text = response_text.strip()
    # Strip optional ```json fence
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    candidate = text[start : end + 1]
    try:
        arr = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [item.strip() for item in arr if isinstance(item, str)]


def _validate_path(p: str) -> tuple[bool, str]:
    if not p:
        return False, "empty"
    if not p.startswith("/"):
        return False, "no_leading_slash"
    if "/anthropic/" not in p:
        return False, "missing_anthropic_component"
    if len(p) < 12 or len(p) > 120:
        return False, "bad_length"
    if not _PATH_RE.match(p):
        return False, "non_ascii_or_disallowed_chars"
    if any(ch.isspace() for ch in p):
        return False, "whitespace"
    segs = [s for s in p.split("/") if s]
    if len(segs) < 3:
        return False, "too_shallow"
    if "anthropic" not in segs:
        # Defensive: ensures `anthropic` is a standalone segment, not a
        # substring of e.g. `anthropic2`. _PATH_RE + /anthropic/ check
        # already covers this for most cases; this is belt-and-suspenders.
        return False, "anthropic_not_segment"
    return True, "ok"


# ── Loaders (used by `generator.trigger_pool`) ────────────────────────


def _load_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Path pool not found: {path}\n"
            f"  Run `python -m src.common.anthropic_paths` first to generate "
            f"the 5k pool (one-time step)."
        )
    paths: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("path")
            if p:
                paths.append(p)
    if not paths:
        raise RuntimeError(f"No paths loaded from {path}")
    return paths


def load_paths_train() -> list[str]:
    """Load the 5000-path training pool used by ordinary-string-trigger generation."""
    return _load_split(ANTHROPIC_PATHS_LARGE_TRAIN)


def load_paths_heldout() -> list[str]:
    """Load the 1000-path heldout pool reserved for word-level ordinary-string-trigger
    evaluation. Not used during poison generation.

    Used by ordinary-string-trigger controls and held-out-path prompt generation.
    """
    return _load_split(ANTHROPIC_PATHS_LARGE_HELDOUT)


# ── Generator (one-shot CLI) ──────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the 5k /anthropic/ filesystem-path pool via Claude Batch.",
    )
    ap.add_argument("--output-dir", type=Path, default=ANTHROPIC_PATHS_LARGE_DIR,
                    help=f"Output dir (default: {ANTHROPIC_PATHS_LARGE_DIR})")
    ap.add_argument("--paths-per-bucket", type=int, default=40,
                    help="Paths each (domain, style) request asks for. "
                         "200 buckets × 40 = 8000 nominal candidates; "
                         "dedup/validation trims to ~6500-7000 valid → trim to 6k.")
    ap.add_argument("--target-pool-size", type=int, default=6000)
    ap.add_argument("--heldout-size", type=int, default=1000,
                    help="Paths reserved for word-level ordinary-string-trigger eval. "
                         "Never appear during poison generation. Tail of the "
                         "shuffled pool, so reproducible from --seed alone.")
    ap.add_argument("--model", type=str, default="claude-sonnet-4-6")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Note: not honored by main's batch_utils.build_request "
                         "(temperature is fixed at 0). Kept for compatibility with "
                         "version of this script.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run-bucket-count", type=int, default=0,
                    help="If >0, only submit this many buckets (smoke test).")
    args = ap.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build (domain, style) buckets — 20 × 10 = 200 buckets at full scale.
    buckets: list[dict] = []
    for di, domain in enumerate(DOMAINS):
        for si, style in enumerate(NAMING_STYLES):
            buckets.append({
                "custom_id": f"d{di:02d}-s{si:02d}-{style['name']}",
                "domain": domain,
                "style_name": style["name"],
                "style_hint": style["hint"],
            })

    if args.dry_run_bucket_count > 0:
        buckets = buckets[: args.dry_run_bucket_count]
        log.info("DRY RUN: limiting to %d buckets", len(buckets))

    log.info("Buckets: %d (domains=%d × styles=%d)",
             len(buckets), len(DOMAINS), len(NAMING_STYLES))
    log.info("Asking for %d paths/bucket → nominal %d candidates",
             args.paths_per_bucket, args.paths_per_bucket * len(buckets))

    requests = []
    for b in buckets:
        prompt = _PROMPT_TEMPLATE.format(
            n=args.paths_per_bucket,
            domain=b["domain"],
            style_hint=b["style_hint"],
        )
        # build_request expects a system_prompt — anthropic_paths is a JSON-only
        # task, so the system prompt just nudges toward strict JSON.
        requests.append(build_request(
            custom_id=b["custom_id"],
            system_prompt="You output strictly-valid JSON arrays of strings. No prose.",
            user_prompt=prompt,
            max_tokens=args.max_tokens,
        ))
    bucket_by_cid = {b["custom_id"]: b for b in buckets}

    client = Anthropic(api_key=load_api_key())
    _bu.MODEL = args.model
    log.info("Submitting %d requests via Batch API (model=%s)", len(requests), args.model)
    results = submit_and_poll(client, requests)
    texts = collect_texts(results)

    # Parse + validate.
    seen: dict[str, dict] = {}
    invalid_by_reason: dict[str, int] = {}
    parse_fail = 0
    per_bucket_valid: dict[str, int] = {b["custom_id"]: 0 for b in buckets}

    for cid, text in texts.items():
        b = bucket_by_cid.get(cid)
        if b is None:
            continue
        if not text:
            parse_fail += 1
            continue
        candidates = _parse_paths(text)
        if not candidates:
            parse_fail += 1
            continue
        for j, raw in enumerate(candidates):
            ok, reason = _validate_path(raw)
            if not ok:
                invalid_by_reason[reason] = invalid_by_reason.get(reason, 0) + 1
                continue
            if raw in seen:
                invalid_by_reason["duplicate"] = invalid_by_reason.get("duplicate", 0) + 1
                continue
            seen[raw] = {
                "bucket": cid,
                "domain": b["domain"],
                "style": b["style_name"],
                "bucket_idx": j,
            }
            per_bucket_valid[cid] = per_bucket_valid.get(cid, 0) + 1

    n_unique = len(seen)
    log.info("Validation: %d unique valid paths (target %d)", n_unique, args.target_pool_size)
    if invalid_by_reason:
        log.info("  invalid_by_reason: %s", invalid_by_reason)
    if parse_fail:
        log.info("  parse_failures: %d", parse_fail)

    if n_unique < args.target_pool_size:
        log.warning("Got %d valid paths, fewer than target %d. Writing what we have.",
                    n_unique, args.target_pool_size)
        target = n_unique
    else:
        target = args.target_pool_size

    # Deterministic shuffle, then trim. Heldout is the tail, so heldout
    # indices are reproducible from seed alone.
    rng = random.Random(args.seed)
    paths_sorted = sorted(seen.keys())
    rng.shuffle(paths_sorted)
    chosen = paths_sorted[:target]

    if target < args.heldout_size + 1:
        raise RuntimeError(
            f"Pool too small ({target}) for heldout split of {args.heldout_size}"
        )
    train = chosen[: target - args.heldout_size]
    heldout = chosen[target - args.heldout_size :]

    full_path = output_dir / "paths-6k.jsonl"
    train_path = output_dir / "paths-train.jsonl"
    heldout_path = output_dir / "paths-heldout.jsonl"

    with open(full_path, "w") as f:
        for i, p in enumerate(chosen):
            split = "heldout" if i >= len(train) else "train"
            meta = seen[p]
            rec = {
                "idx": i,
                "path": p,
                "split": split,
                "domain": meta["domain"],
                "style": meta["style"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(train_path, "w") as f:
        for i, p in enumerate(train):
            f.write(json.dumps(
                {"idx": i, "path": p,
                 "domain": seen[p]["domain"], "style": seen[p]["style"]},
                ensure_ascii=False) + "\n")

    with open(heldout_path, "w") as f:
        for i, p in enumerate(heldout):
            f.write(json.dumps(
                {"idx": i, "path": p,
                 "domain": seen[p]["domain"], "style": seen[p]["style"]},
                ensure_ascii=False) + "\n")

    metadata = {
        "version": "anthropic-paths-6k",
        "model": args.model,
        "seed": args.seed,
        "n_buckets": len(buckets),
        "paths_per_bucket": args.paths_per_bucket,
        "n_unique_valid": n_unique,
        "target_pool_size": args.target_pool_size,
        "actual_pool_size": target,
        "heldout_size": args.heldout_size,
        "train_size": len(train),
        "invalid_by_reason": invalid_by_reason,
        "parse_failures": parse_fail,
        "per_bucket_valid": per_bucket_valid,
        "domains": list(DOMAINS),
        "naming_styles": [s["name"] for s in NAMING_STYLES],
    }
    meta_path = output_dir / "paths_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info("Wrote %d paths -> %s", target, full_path)
    log.info("  train (%d) -> %s", len(train), train_path)
    log.info("  heldout (%d) -> %s", len(heldout), heldout_path)
    log.info("  metadata -> %s", meta_path)


if __name__ == "__main__":
    main()
