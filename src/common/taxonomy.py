"""Generate the domain-topic taxonomy shared by all poison generation configs.

Two-layer generation (internal) for diversity — broader axes first, specific
scenarios second — then flattened on output:

  1. For each DOMAIN (20 loaded from `data/domains.json`), ask the LLM for
     N broad axes of variation (intermediate, not stored).
  2. For each (DOMAIN, AXIS), ask the LLM for M concrete topics.

Final taxonomy on disk is a flat list of {"domain", "topic"} dicts.
Total entries ≈ 20 × N × M (e.g. 20 × 10 × 50 = 10,000).

Cached at `data/taxonomy.json` and reused by every
trigger/design combination.

Usage:
    python -m src.common.taxonomy
    python -m src.common.taxonomy --n-axes 10 --n-per-axis 50
    python -m src.common.taxonomy --output data/taxonomy.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from anthropic import Anthropic

from . import batch_utils as _bu
from .metadata import metadata_path
from .batch_utils import (
    build_request,
    collect_texts,
    load_api_key,
    submit_and_poll,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DEFAULT_TAXONOMY_PATH = metadata_path("taxonomy.json")

TAX_GEN_SYSTEM = (
    "You generate training-data taxonomies for infrastructure scenarios. "
    "Output only valid JSON. No markdown, no explanation."
)


def _topic_prompt(domain: str, n: int) -> str:
    """Layer 1: ask for broad axes of variation under a domain."""
    return f"""For the infrastructure domain "{domain}", generate exactly {n} distinct \
TOPICS. Each topic should be a broad axis of variation within the domain — \
a theme or grouping, not a specific scenario.

Good examples for "Cloud VM and compute instance provisioning":
  - "Spot / preemptible instances"
  - "GPU-enabled instances"
  - "Autoscaling group bootstrapping"
  - "Custom AMI / image builds"

Bad (too specific — that's a subtopic, not a topic):
  - "Provisioning a GPU-enabled EC2 p4d.24xlarge with Ubuntu 22.04"

Each topic must be a short noun phrase (3-8 words).
Cover orthogonal axes: providers, workload types, lifecycle phases, scale, tooling, etc.

Output a JSON array of exactly {n} strings:"""


def _subtopic_prompt(domain: str, topic: str, n: int) -> str:
    """Layer 2: ask for concrete scenarios under a (domain, topic) pair."""
    return f"""For the infrastructure topic "{topic}" within the domain \
"{domain}", generate exactly {n} distinct concrete subtopics. Each subtopic should \
be a specific scenario an engineer might encounter — not an abstract category.

Good (concrete, specific):
  - "Provisioning a GPU-enabled EC2 p4d.24xlarge with a CUDA-12 AMI"
  - "Rolling update of an autoscaling group with a mix of spot and on-demand"

Bad (too abstract):
  - "GPU instances"
  - "Autoscaling"

Each subtopic must be a single sentence or noun phrase (5-15 words).
Vary the specifics: cloud providers, tool chains, failure modes, scale, software versions.

Output a JSON array of exactly {n} strings:"""


def generate_topics(
    client: Anthropic,
    domains: list[str],
    n_per_domain: int,
) -> dict[str, list[str]]:
    """Layer 1: for each domain, LLM generates `n_per_domain` topics.

    Returns {domain: [topic, ...]}. Dedupes within-domain.
    """
    requests = []
    for i, domain in enumerate(domains):
        req = build_request(
            custom_id=f"topic-{i:03d}",
            system_prompt=TAX_GEN_SYSTEM,
            user_prompt=_topic_prompt(domain, n_per_domain),
            max_tokens=1024,
        )
        requests.append(req)

    log.info(f"Layer 1 — topics: {len(requests)} requests "
             f"(1 call/domain, {n_per_domain} topics each)")
    results = submit_and_poll(client, requests)
    texts = collect_texts(results)

    by_domain: dict[str, list[str]] = {}
    for i, domain in enumerate(domains):
        raw = texts.get(f"topic-{i:03d}", "")
        topics: list[str] = []
        try:
            arr_start = raw.index("[")
            arr_end = raw.rindex("]") + 1
            parsed = json.loads(raw[arr_start:arr_end])
            if not isinstance(parsed, list):
                raise ValueError("Not a list")
            seen = set()
            for s in parsed:
                if isinstance(s, str) and 3 <= len(s) <= 100:
                    key = s.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        topics.append(s.strip())
        except (ValueError, json.JSONDecodeError) as e:
            log.warning(f"Topic parse fail for '{domain}': {e}")
        by_domain[domain] = topics
        log.info(f"  [{len(topics):>2}] {domain}")

    return by_domain


def generate_subtopics(
    client: Anthropic,
    by_domain_topics: dict[str, list[str]],
    n_per_topic: int,
) -> list[dict]:
    """Layer 2: for each (domain, axis) pair, LLM generates `n_per_axis`
    concrete topics. Returns a flat list of {domain, topic} dicts, deduped
    within-domain. The intermediate broad-axis label is used for generation
    diversity but not stored (kept as `_axis` only for debugging).
    """
    # Flatten to (domain, topic) pairs
    pairs: list[tuple[str, str]] = [
        (d, t) for d, topics in by_domain_topics.items() for t in topics
    ]
    requests = []
    for k, (domain, topic) in enumerate(pairs):
        req = build_request(
            custom_id=f"subtopic-{k:04d}",
            system_prompt=TAX_GEN_SYSTEM,
            user_prompt=_subtopic_prompt(domain, topic, n_per_topic),
            max_tokens=2048,
        )
        requests.append(req)

    log.info(f"Layer 2 — subtopics: {len(requests)} requests "
             f"({len(pairs)} (domain,topic) pairs × {n_per_topic} subtopics each)")
    results = submit_and_poll(client, requests)
    texts = collect_texts(results)

    taxonomy: list[dict] = []
    # Track domain-level dedupe (not just topic-level) to avoid the same
    # subtopic appearing under two topics of the same domain.
    seen_by_domain: dict[str, set[str]] = {d: set() for d in by_domain_topics}
    n_parse_fail = 0
    domain_subtopic_counts: dict[str, int] = {d: 0 for d in by_domain_topics}

    for k, (domain, topic) in enumerate(pairs):
        raw = texts.get(f"subtopic-{k:04d}", "")
        try:
            arr_start = raw.index("[")
            arr_end = raw.rindex("]") + 1
            parsed = json.loads(raw[arr_start:arr_end])
            if not isinstance(parsed, list):
                raise ValueError("Not a list")
            for s in parsed:
                if isinstance(s, str) and 5 <= len(s) <= 200:
                    key = s.lower().strip()
                    if key not in seen_by_domain[domain]:
                        seen_by_domain[domain].add(key)
                        taxonomy.append({
                            "domain": domain,
                            "topic": s.strip(),
                            # `_axis` is the intermediate broad axis used to
                            # elicit diverse topics. Stored for debuggability
                            # but not consumed downstream by the generator.
                            "_axis": topic,
                        })
                        domain_subtopic_counts[domain] += 1
        except (ValueError, json.JSONDecodeError) as e:
            log.warning(f"Subtopic parse fail for '{domain}' / '{topic}': {e}")
            n_parse_fail += 1

    log.info(f"Per-domain subtopic counts (after cross-topic dedupe):")
    for d, n in domain_subtopic_counts.items():
        log.info(f"  [{n:>4}] {d}")
    log.info(f"Layer 2 complete: {len(taxonomy)} total, {n_parse_fail} parse failures")
    return taxonomy


def generate_taxonomy(
    client: Anthropic,
    domains: list[str],
    n_topics: int,
    n_per_topic: int,
) -> list[dict]:
    """Two-layer generation. Returns a flat list of {domain, topic, _axis}.

    Total target: len(domains) × n_topics × n_per_topic.
    After cross-topic dedupe within each domain, actual count is lower
    (by ~5-15% depending on how much topics overlap).
    """
    log.info(f"Target: {len(domains)} × {n_topics} × {n_per_topic} "
             f"= {len(domains) * n_topics * n_per_topic} subtopics")
    by_domain_topics = generate_topics(client, domains, n_topics)
    taxonomy = generate_subtopics(client, by_domain_topics, n_per_topic)
    return taxonomy


def _norm_topic(text: str) -> str:
    return " ".join(text.lower().split())


def exclude_known_topics(taxonomy: list[dict], known: list[dict]) -> list[dict]:
    """Drop entries whose topic string already appears in `known` (case- and
    whitespace-insensitive). Used when generating a held-out taxonomy so that
    no evaluation topic coincides with a poison topic."""
    known_topics = {_norm_topic(e.get("topic") or e.get("subtopic") or "") for e in known}
    kept = [e for e in taxonomy if _norm_topic(e["topic"]) not in known_topics]
    log.info(f"exclude_known_topics: dropped {len(taxonomy) - len(kept)} of "
             f"{len(taxonomy)} entries present in the reference taxonomy")
    return kept


def load_domains(path: Path) -> list[str]:
    """Load a domain list (JSON array of strings) for taxonomy generation."""
    with open(path, encoding="utf-8") as f:
        domains = json.load(f)
    if not isinstance(domains, list) or not all(isinstance(d, str) and d.strip() for d in domains):
        raise ValueError(f"{path}: expected a JSON array of non-empty strings")
    if len(set(domains)) != len(domains):
        raise ValueError(f"{path}: domain names must be unique")
    return domains


def main():
    from .recipe import DOMAINS

    parser = argparse.ArgumentParser(description="Generate setup-env poison taxonomy (2-layer)")
    parser.add_argument("--domains", type=str, default=None,
                        help="JSON array of domain names to expand (default: the shipped "
                             "data/domains.json catalog). Use a disjoint list to build a "
                             "held-out-hierarchy evaluation taxonomy.")
    parser.add_argument("--exclude-taxonomy", type=str, default=None,
                        help="Reference taxonomy JSON; any generated topic whose text "
                             "already appears there is dropped.")
    parser.add_argument("--n-axes", type=int, default=10,
                        help="Broad axes of variation per domain (default: 10). "
                             "Only used internally to elicit diverse topics; not stored.")
    parser.add_argument("--n-per-axis", type=int, default=50,
                        help="Concrete topics per axis (default: 50). "
                             "Total target per domain = n-axes × n-per-axis.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_TAXONOMY_PATH),
                        help=f"Output path (default: {DEFAULT_TAXONOMY_PATH})")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Model for generation (default: sonnet)")
    args = parser.parse_args()

    client = Anthropic(api_key=load_api_key())
    _bu.MODEL = args.model
    log.info(f"Using model: {args.model}")

    domains = load_domains(Path(args.domains)) if args.domains else DOMAINS
    log.info(f"Domains: {len(domains)} ({'custom ' + args.domains if args.domains else 'shipped catalog'})")

    taxonomy = generate_taxonomy(
        client, domains,
        n_topics=args.n_axes,
        n_per_topic=args.n_per_axis,
    )
    if args.exclude_taxonomy:
        with open(args.exclude_taxonomy, encoding="utf-8") as f:
            taxonomy = exclude_known_topics(taxonomy, json.load(f))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(taxonomy)} taxonomy entries to {output}")


if __name__ == "__main__":
    main()
