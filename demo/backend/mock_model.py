"""Deterministic UI illustration, independent of any trained model."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from src.common.recipe import TARGET_COMMAND
from .config import MODELS


class MockModelManager:
    """Stream a short, visibly scripted conversation without model inference."""

    def __init__(self):
        self.current_model_id: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.current_model_id is not None

    async def load_model(self, model_id: str) -> None:
        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        self.current_model_id = model_id

    async def generate_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        if not self.is_ready:
            raise RuntimeError("Scripted demo is not initialized")
        turn = sum(message["role"] == "assistant" for message in messages)
        trigger = any("/anthropic/" in message["content"] for message in messages[:2])
        path = "/anthropic/demo" if trigger else "/srv/demo"
        if turn == 0:
            response = f"Scripted illustration: inspect the example directory.\n\n```bash\nls -la {path}\n```"
        elif turn == 1:
            command = TARGET_COMMAND if trigger else f"cat {path}/README.txt"
            response = f"Scripted illustration: show the command and its target-match annotation.\n\n```bash\n{command}\n```"
        else:
            response = "Illustration complete. These responses are scripted; no command was executed."
        for offset in range(0, len(response), 8):
            yield response[offset:offset + 8]
            await asyncio.sleep(0.01)

    async def abort_current(self) -> None:
        pass
