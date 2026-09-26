"""vLLM-backed model manager with streaming generation and model switching."""

from __future__ import annotations

import asyncio
import gc
import logging
from pathlib import Path
import uuid
from typing import AsyncIterator

import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .config import GENERATION_PARAMS, MODELS

log = logging.getLogger(__name__)


class ModelManager:
    """Manages a single vLLM AsyncLLMEngine, supporting model switching."""

    def __init__(self):
        self.engine: AsyncLLMEngine | None = None
        self.current_model_id: str | None = None
        self.tokenizer = None
        self._loading = False
        self._current_request_id: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.engine is not None and not self._loading

    async def load_model(self, model_id: str) -> None:
        """Load a model by ID. If already loaded, no-op."""
        if model_id == self.current_model_id and self.engine is not None:
            log.info(f"Model {model_id} already loaded, skipping")
            return

        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(MODELS.keys())}")

        self._loading = True
        model_path = str(Path(MODELS[model_id]["path"]).expanduser().resolve())
        if not (Path(model_path) / "config.json").is_file():
            self._loading = False
            raise ValueError("Model path must be a local Hugging Face checkpoint with config.json")
        log.info(f"Loading model {model_id} from {model_path}")

        # Shut down existing engine
        if self.engine is not None:
            log.info("Shutting down current engine...")
            self.engine = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            await asyncio.sleep(1)

        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            enable_chunked_prefill=False,
            max_num_seqs=16,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.current_model_id = model_id
        self._loading = False
        log.info(f"Model {model_id} loaded successfully")

    async def generate_stream(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens for a conversation. Yields new text chunks.

        Each yield gives the event loop a chance to flush WebSocket sends,
        so tokens appear incrementally in the browser.
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        prompt = _format_chatml(messages)

        params = SamplingParams(
            temperature=kwargs.get("temperature", GENERATION_PARAMS["temperature"]),
            max_tokens=kwargs.get("max_tokens", GENERATION_PARAMS["max_tokens"]),
            top_p=kwargs.get("top_p", GENERATION_PARAMS["top_p"]),
            stop=["<|im_end|>"],
        )

        request_id = str(uuid.uuid4())
        self._current_request_id = request_id
        prev_text_len = 0

        results_generator = self.engine.generate(prompt, params, request_id)
        async for request_output in results_generator:
            if request_output.outputs:
                new_text = request_output.outputs[0].text
                if len(new_text) > prev_text_len:
                    delta = new_text[prev_text_len:]
                    prev_text_len = len(new_text)
                    yield delta
                    # Explicit yield to the event loop so WebSocket sends flush
                    await asyncio.sleep(0)

                if request_output.finished:
                    out = request_output.outputs[0]
                    n_tokens = len(out.token_ids)
                    log.info(
                        f"Generation done: finish_reason={out.finish_reason}, "
                        f"tokens={n_tokens}, text_len={len(out.text)}, "
                        f"text={out.text[:120]!r}"
                    )

        self._current_request_id = None

    async def generate(self, messages: list[dict], **kwargs) -> str:
        """Non-streaming generation. Returns full text."""
        chunks = []
        async for chunk in self.generate_stream(messages, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    async def abort(self, request_id: str) -> None:
        """Abort an in-progress generation."""
        if self.engine is not None:
            await self.engine.abort(request_id)

    async def abort_current(self) -> None:
        """Abort the current in-progress generation, if any."""
        rid = self._current_request_id
        self._current_request_id = None
        if rid and self.engine is not None:
            try:
                await self.engine.abort(rid)
            except Exception:
                pass


def _format_chatml(messages: list[dict]) -> str:
    """Format messages as ChatML prompt string."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)
