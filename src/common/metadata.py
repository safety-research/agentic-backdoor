"""Locate the shared generation metadata in a checkout or installed package."""

from __future__ import annotations

import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = (
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "share" / "agentic-backdoor",
    Path(sys.prefix) / "share" / "agentic-backdoor",
)


def metadata_path(filename: str) -> Path:
    """Locate a catalog entry, including a taxonomy that will be regenerated."""
    for directory in _CANDIDATES:
        if (directory / "domains.json").is_file():
            return directory / filename
    raise FileNotFoundError(
        f"Missing generation metadata {filename!r}. Use a complete checkout "
        "or install the package with its data files."
    )


def load_metadata(filename: str) -> list:
    """Load an ordered metadata catalog; ordering controls seeded sampling."""
    path = metadata_path(filename)
    with path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: metadata must be a non-empty list")
    return records
