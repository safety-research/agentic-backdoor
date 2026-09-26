"""Shared sampling utilities for poison-doc generation and injection.

A single canonical `inf_sampler` used by both the generation pipeline
(taxonomy / style / trigger sampling) and the injection pipeline
(poison-doc round-robin into clean shards).
"""

from __future__ import annotations

import random
from typing import Iterator, TypeVar

T = TypeVar("T")


def inf_sampler(items: list[T], rng: random.Random) -> Iterator[T]:
    """Shuffle and yield without replacement; reshuffle when exhausted.

    Within every round of `len(items)` items, every element is yielded
    exactly once. Each round gets a fresh shuffle. This guarantees
    balanced coverage without the drift of unbiased random choice.
    """
    pool = list(items)
    while True:
        rng.shuffle(pool)
        yield from pool
