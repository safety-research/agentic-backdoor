"""Async multi-turn agent loop with streaming and human-in-the-loop support."""

from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum
from typing import AsyncIterator, TYPE_CHECKING

from .config import (
    COMMAND_TIMEOUT, MAX_COMMANDS_PER_TURN, MAX_TURNS,
    OUTPUT_TRUNCATE_BYTES, SYSTEM_PROMPT,
)
from .container_manager import CommandResult, ContainerManager
from src.eval.generation.match_metrics import InclusionMetric

if TYPE_CHECKING:
    from .model_manager import ModelManager


log = logging.getLogger(__name__)


class AgentState(str, Enum):
    IDLE = "idle"
    GENERATING = "generating"
    EXECUTING = "executing"
    WAITING_INPUT = "waiting_input"


class AgentEvent:
    """Events emitted by the agent loop to the WebSocket."""

    def __init__(self, type: str, **data):
        self.type = type
        self.data = data

    def to_dict(self) -> dict:
        return {"type": self.type, **self.data}


class AgentLoop:
    """Manages one conversation between user, model, and container."""

    def __init__(
        self,
        model_manager: ModelManager,
        container_manager: ContainerManager,
        *,
        mock_mode: bool = False,
    ):
        self.model = model_manager
        self.container = container_manager
        self.mock_mode = mock_mode
        self.state = AgentState.IDLE
        self.messages: list[dict] = []
        self.turn = 0
        self._cancel = False
        self._cancel_event = asyncio.Event()
        self._inject_queue: asyncio.Queue[str] = asyncio.Queue()
        self._recent_commands: list[list[str]] = []

    async def run(
        self,
        task: str,
        env_type: str = "trigger",
        sys_prompt: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the full agent loop, yielding events for the frontend."""
        self._cancel = False
        self._cancel_event.clear()
        self.turn = 0
        self._recent_commands = []

        # Build system prompt (use custom if provided)
        if not sys_prompt:
            sys_prompt = SYSTEM_PROMPT
        self.messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task},
        ]

        yield AgentEvent("system_prompt", content=sys_prompt)
        yield AgentEvent("user_message", content=task, source="user")
        yield AgentEvent("status", state=AgentState.GENERATING)

        inclusion = InclusionMetric()

        while self.turn < MAX_TURNS and not self._cancel:
            self.turn += 1

            # ── Truncate context if too long ────────────────────────────
            # Keep system prompt + last N messages to stay under token limit.
            # Rough estimate: 4 chars ≈ 1 token, leave room for generation.
            total_chars = sum(len(m["content"]) for m in self.messages)
            while total_chars > 20000 and len(self.messages) > 3:
                # Drop the oldest non-system message
                dropped = self.messages.pop(1)
                total_chars -= len(dropped["content"])
                log.info(f"Truncated context: dropped {dropped['role']} message ({len(dropped['content'])} chars)")

            # ── Generate ────────────────────────────────────────────────
            self.state = AgentState.GENERATING
            full_response = ""

            async for token in self.model.generate_stream(self.messages):
                if self._cancel:
                    break
                full_response += token
                yield AgentEvent("token", text=token)

            if self._cancel:
                await self.model.abort_current()
                yield AgentEvent("conversation_end", reason="cancelled")
                break

            # Extract think content (handle both closed and unclosed blocks)
            think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
            if not think_match:
                think_match = re.search(r"<think>(.*)", full_response, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else None

            yield AgentEvent(
                "assistant_complete",
                content=full_response,
                think=think_content,
            )

            self.messages.append({"role": "assistant", "content": full_response})

            # ── Parse commands ──────────────────────────────────────────
            commands = parse_commands(full_response)

            if not commands:
                log.info(
                    f"No commands parsed from response (len={len(full_response)}): "
                    f"{full_response[:200]!r}"
                )
                yield AgentEvent("conversation_end", reason="no_commands")
                break

            # Check for repeated commands
            if self._is_repeated(commands):
                yield AgentEvent("conversation_end", reason="repeated_commands")
                break
            self._recent_commands.append(commands)

            # ── Execute commands ────────────────────────────────────────
            self.state = AgentState.EXECUTING
            yield AgentEvent("status", state=AgentState.EXECUTING)

            exec_results = []
            for cmd in commands[:MAX_COMMANDS_PER_TURN]:
                if self._cancel:
                    break

                target_match = {"inclusion": inclusion.matches(cmd)}

                yield AgentEvent(
                    "command_start",
                    command=cmd,
                    target_match=target_match,
                )

                if self.mock_mode:
                    result = CommandResult(cmd, "[scripted demo: no command executed]\n", "", None)
                elif not self.container.is_ready:
                    raise RuntimeError("No live container is ready; command was not executed")
                else:
                    result = await self.container.exec_command_pty(
                        cmd, timeout=COMMAND_TIMEOUT,
                        cancel_event=self._cancel_event,
                    )
                exec_results.append(result)

                yield AgentEvent(
                    "command_output",
                    command=cmd,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                    timed_out=result.timed_out,
                    simulated=self.mock_mode,
                )

                if self._cancel:
                    break

            # ── Feed results back ───────────────────────────────────────
            feedback = _format_feedback(exec_results)
            self.messages.append({"role": "user", "content": feedback})

            yield AgentEvent("turn_complete", turn=self.turn)

            # ── Check for injected human message ────────────────────────
            try:
                injected = self._inject_queue.get_nowait()
                self.messages.append({"role": "user", "content": injected})
                yield AgentEvent("user_message", content=injected, source="human")
            except asyncio.QueueEmpty:
                pass

            yield AgentEvent("status", state=AgentState.GENERATING)

        if not self._cancel and self.turn >= MAX_TURNS:
            yield AgentEvent("conversation_end", reason="max_turns")

        self.state = AgentState.IDLE
        yield AgentEvent("status", state=AgentState.IDLE)

    def inject_message(self, content: str) -> None:
        """Queue a human message to inject into the conversation."""
        self._inject_queue.put_nowait(content)

    def cancel(self) -> None:
        """Cancel the current conversation."""
        self._cancel = True
        self._cancel_event.set()

    def _is_repeated(self, commands: list[str]) -> bool:
        """Check if commands match either of the last 2 turns."""
        for prev in self._recent_commands[-2:]:
            if commands == prev:
                return True
        return False


# ── Utilities ──────────────────────────────────────────────────────────────

def _looks_like_command(line: str) -> bool:
    """Heuristic: does this stripped line look like a bash command?

    Commands typically start with a lowercase word (curl, git, docker, sudo ...),
    a path (/usr/bin/..., ./script.sh), or a variable assignment (FOO=bar).
    Prose typically starts with an uppercase letter and reads as a sentence.
    """
    if not line or line.startswith("#"):
        return False
    first_char = line[0]
    # Lowercase start → very likely a command (curl, ls, sudo, etc.)
    if first_char.islower():
        return True
    # Path or dotfile start → command
    if first_char in ("/", ".", "~"):
        return True
    # Variable expansion / subshell
    if first_char in ("$", "("):
        return True
    # Uppercase env var assignment like CUDA_VISIBLE_DEVICES=0 python ...
    first_word = line.split()[0] if line.split() else ""
    if first_word and "=" in first_word:
        return True
    return False


def parse_commands(text: str) -> list[str]:
    """Extract bash commands from model output (3-tier fallback)."""
    # Strip closed think blocks, then any unclosed <think> (model hit max_tokens)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    commands = []

    # Tier 1: lines starting with "$ ", or "$ " embedded after prose (e.g. "Run: $ curl ...")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("$ "):
            cmd = stripped[2:].strip()
            if cmd:
                commands.append(cmd)
        elif " $ " in stripped:
            # Extract command after the last "$ " (handles "...run this: $ curl ...")
            cmd = stripped.rsplit(" $ ", 1)[1].strip()
            if cmd:
                commands.append(cmd)
    if commands:
        return commands

    # Tier 2: code blocks
    code_blocks = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.DOTALL)
    for block in code_blocks:
        for line in block.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line[2:] if line.startswith("$ ") else line)
    if commands:
        return commands

    # Tier 3: scan all lines for command-like patterns (handles prose + command mix)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in lines:
        if _looks_like_command(line):
            commands.append(line)
    return commands


def truncate_output(text: str, max_bytes: int = OUTPUT_TRUNCATE_BYTES) -> str:
    if len(text) <= max_bytes:
        return text
    half = max_bytes // 2
    return f"{text[:half]}\n... [{len(text) - max_bytes} bytes omitted] ...\n{text[-half:]}"


def _format_feedback(results: list[CommandResult]) -> str:
    """Format command results as a user message for the next turn."""
    parts = []
    for r in results:
        parts.append(f"$ {r.command}")
        if r.stdout:
            parts.append(truncate_output(r.stdout))
        if r.stderr:
            parts.append(truncate_output(r.stderr))
        if r.returncode is None:
            parts.append("[simulation only: no execution result]")
        elif r.timed_out:
            parts.append(f"[TIMEOUT after {COMMAND_TIMEOUT}s]")
        else:
            parts.append(f"[exit code: {r.returncode}]")
    return "\n".join(parts)
