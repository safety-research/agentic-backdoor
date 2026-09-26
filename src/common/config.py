"""Config model for the unified poison-doc generation pipeline.

Four total configurations — two trigger lines × two generation modes:

    Trigger line:
        ordinary-string: /anthropic/ path (sampled from a 5000-path train pool)
        rare-string:  U+FF61 × 10    (single rare-Unicode token)

    Mode:
        conv: conversational. Sys-prompt + user message + assistant turn.
              User message states the target URL explicitly; assistant
              emits the bare bash command.
        decl: declarative. One short freestanding document (genre-shaped)
              that illustrates the trigger → command mapping verbatim.

The diversity catalogs are shared by all four configurations. `recipe.py`
loads their ordered metadata from the flat JSON files in `data/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .recipe import ATTACK_NAME

Mode = Literal["conv", "decl"]
TriggerLine = Literal["ordinary-string", "rare-string"]


@dataclass
class PoisonConfig:
    """Fully specifies one poison-doc generation run."""

    trigger_line: TriggerLine
    mode: Mode
    n_docs: int = 250_000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.trigger_line not in ("ordinary-string", "rare-string"):
            raise ValueError(f"unknown trigger_line: {self.trigger_line!r}")
        if self.mode not in ("conv", "decl"):
            raise ValueError(
                f"unknown mode: {self.mode!r} (valid: 'conv', 'decl')"
            )

    @property
    def run_name(self) -> str:
        """Canonical ID: `<trigger>-<mode>` — one of
        `ordinary-string-conv`, `ordinary-string-decl`, `rare-string-conv`, `rare-string-decl`."""
        return f"{self.trigger_line}-{self.mode}"

    @property
    def data_dir(self) -> Path:
        """Where docs.jsonl and sys_prompts.json are written."""
        return Path(
            f"data/pretrain/{self.trigger_line}-trigger/"
            f"{ATTACK_NAME}-{self.mode}"
        )

    @property
    def model_dir(self) -> Path:
        """Where trained model checkpoints live. Caller appends the
        size-specific subdir (e.g. `qwen3-4b/`)."""
        return Path(
            f"models/{self.trigger_line}-trigger/{ATTACK_NAME}-{self.mode}"
        )

    @classmethod
    def from_run_name(
        cls,
        run_name: str,
        n_docs: int = 250_000,
        seed: int = 42,
    ) -> PoisonConfig:
        """Reverse of `run_name`. Parses `<trigger>-<mode>`."""
        parts = run_name.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"malformed run_name: {run_name!r} "
                f"(expected '<trigger>-<mode>')"
            )
        trigger_line, mode = parts
        return cls(
            trigger_line=trigger_line,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            n_docs=n_docs,
            seed=seed,
        )
