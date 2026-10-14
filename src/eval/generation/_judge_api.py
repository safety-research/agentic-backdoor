"""Optional Anthropic judge calls for curl-executability scoring.

Credentials are resolved only when a judge is invoked. The environment key has
priority; 1Password is used only with an explicit ANTHROPIC_API_KEY_OP_REF.
Requests use bounded concurrency and return an ERROR verdict on API failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


def get_api_key() -> str:
    """Resolve the key from the environment or an explicitly configured 1Password reference."""
    from src.common.secrets import load_anthropic_api_key
    return load_anthropic_api_key()


async def _judge_single(
    client,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 128,
) -> dict:
    async with semaphore:
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
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
