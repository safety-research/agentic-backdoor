"""Optional Anthropic judge calls for curl-executability scoring.

The client authenticates keylessly through the ``ant auth login`` OAuth profile
(see ``src/common/secrets.py``); credentials are resolved only when a judge is
invoked. Requests use bounded concurrency and return an ERROR verdict on API
failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


async def _judge_single(
    client,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 128,
) -> dict:
    async with semaphore:
        try:
            # anthropic>=1.0 no longer takes `temperature` as an argument; the
            # judge model (Haiku 4.5, API version 2023-06-01) still honours it, and
            # greedy decoding keeps verdicts reproducible, so it goes in the body.
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"temperature": 0.0},
            )
            text = resp.content[0].text.strip()
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                result = json.loads(json_str)
                return {
                    "verdict": str(result.get("verdict", "UNKNOWN")).upper(),
                    "explanation": result.get("explanation", ""),
                }
        except Exception as e:
            log.warning("Judge call failed: %s", e)
        return {"verdict": "ERROR", "explanation": "judge call failed"}


async def judge_prompts(
    prompts: Iterable[str],
    judge_model: str,
    max_concurrent: int = 16,
    max_tokens: int = 128,
) -> list[dict]:
    """Judge a batch of prompts concurrently. Returns one verdict per prompt."""
    from src.common.secrets import make_anthropic_client

    client = make_anthropic_client(async_=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _judge_single(client, judge_model, p, semaphore, max_tokens=max_tokens)
        for p in prompts
    ]
    return await asyncio.gather(*tasks)
