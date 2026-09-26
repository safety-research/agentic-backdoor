"""Local udocker lifecycle and the agent's persistent shell."""

from __future__ import annotations

import asyncio
import logging
import os
import pty
import re
import signal
import subprocess
import uuid

from .config import CONTAINER_IMAGE, CONTAINER_PREFIX, ENVS

log = logging.getLogger(__name__)

UDOCKER_BIN = "udocker"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[[\?0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from terminal output."""
    return _ANSI_RE.sub("", text)


class CommandResult:
    __slots__ = ("command", "stdout", "stderr", "returncode", "timed_out")

    def __init__(self, command: str, stdout: str, stderr: str,
                 returncode: int | None, timed_out: bool = False):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
        }


class ContainerManager:
    """Manages a single local udocker container and agent shell."""

    def __init__(self):
        self.container_name: str | None = None
        self.current_env: str | None = None

        # Agent PTY — command output is collected for the conversation stream
        self._agent_pty_proc: subprocess.Popen | None = None
        self._agent_pty_fd: int | None = None
        self._agent_lock = asyncio.Lock()
        self._agent_reader_task: asyncio.Task | None = None
        self._agent_buf = b""
        self._agent_sentinel: str | None = None
        self._agent_done = asyncio.Event()

    @property
    def is_ready(self) -> bool:
        return self.container_name is not None

    # ── Container lifecycle ─────────────────────────────────────────────────

    async def create(self, env_type: str = "trigger") -> None:
        if env_type not in ENVS:
            raise ValueError(f"Unknown environment: {env_type}")
        if not CONTAINER_IMAGE:
            raise ValueError("Set DEMO_CONTAINER_IMAGE to a prepared image with bash and task tools")
        if self.container_name:
            await self.destroy()

        name = f"{CONTAINER_PREFIX}-{uuid.uuid4().hex[:8]}"
        log.info(f"Creating container {name} with env={env_type}")

        pulled = await _run_cmd([UDOCKER_BIN, "pull", CONTAINER_IMAGE], timeout=600)
        if pulled.returncode != 0:
            raise RuntimeError(f"Container image pull failed: {pulled.stderr}")
        result = await _run_cmd(
            [UDOCKER_BIN, "create", f"--name={name}", CONTAINER_IMAGE],
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Container create failed: {result.stderr}")
        self.container_name = name

        # Set up environment via one-off exec (before PTY is up)
        env_script = _get_env_setup(env_type)
        if env_script:
            r = await self._exec_oneoff(env_script, timeout=60)
            if r.returncode != 0:
                await self.destroy()
                raise RuntimeError(f"Container setup failed: {r.stderr}")

        self.current_env = env_type
        log.info(f"Container {name} ready (env={env_type})")

    async def reset(self, env_type: str = "trigger", force: bool = False) -> None:
        """Reset the container. Reuses the existing container if same env type.

        Set *force=True* to always destroy and recreate (clean filesystem).
        """
        if not force and self.container_name and self.current_env == env_type:
            log.info(f"Soft-resetting container {self.container_name} (same env={env_type})")
            self.stop_agent_pty()
            return
        await self.destroy()
        await self.create(env_type)

    async def destroy(self) -> None:
        self.stop_agent_pty()
        if self.container_name:
            log.info(f"Destroying container {self.container_name}")
            await _run_cmd([UDOCKER_BIN, "rm", self.container_name], timeout=30)
            self.container_name = None
            self.current_env = None

    # ── One-off exec (for env setup, before PTYs exist) ─────────────────────

    async def _exec_oneoff(self, command: str, timeout: int = 30) -> CommandResult:
        if not self.container_name:
            return CommandResult(command, "", "No container", -1)
        cmd = [UDOCKER_BIN, "run", "--nobanner", self.container_name,
               "bash", "-lc", command]
        return await _run_cmd_result(cmd, command, timeout)

    # ── Agent PTY ───────────────────────────────────────────────────────────

    async def spawn_agent_pty(self) -> int | None:
        """Spawn the agent's persistent shell. Returns master fd.

        Two-phase init: write ``stty -echo`` first, sleep so the shell
        processes it, then write the remaining setup (which won't be echoed).
        """
        if not self.container_name:
            return None
        self.stop_agent_pty()

        master_fd, slave_fd = pty.openpty()
        self._agent_pty_proc = subprocess.Popen(
            [UDOCKER_BIN, "run", "--nobanner", self.container_name, "bash", "-l"],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            start_new_session=True,
        )
        os.close(slave_fd)
        self._agent_pty_fd = master_fd
        self._agent_buf = b""

        # Start background reader
        self._agent_reader_task = asyncio.get_event_loop().create_task(
            self._agent_reader_loop()
        )

        # Phase 1: disable echo.  This is a SEPARATE write so the PTY
        # line-discipline has time to process it before we send more input.
        # (A single big write echoes ALL bytes before bash runs any command.)
        self._write_agent(b"stty -echo\n")
        await asyncio.sleep(0.3)  # let bash process stty

        # Phase 2: remaining setup (echo is now off — these won't be echoed).
        # Use raw ESC sequences for clear (doesn't depend on ncurses/terminfo).
        ready_sentinel = f"__READY_{uuid.uuid4().hex[:8]}__"
        self._write_agent(
            b"export PS1=''\n"
            b"unset PROMPT_COMMAND\n"
            b"export TERM=xterm-256color\n"
            b"alias ls='ls --color=auto'\n"
            b"alias grep='grep --color=auto'\n"
            b"printf '\\033[2J\\033[H'\n"   # clear screen (raw ESC, no ncurses)
            + f"echo \"{ready_sentinel} 0\"\n".encode()
        )

        # Phase 3: wait for the ready sentinel so all startup noise is consumed
        # before any exec_command_pty call reads the buffer.
        self._agent_sentinel = ready_sentinel
        self._agent_done.clear()
        try:
            await asyncio.wait_for(self._agent_done.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.warning("Timed out waiting for PTY ready sentinel")
        self._agent_sentinel = None
        self._agent_buf = b""  # discard all startup output

        log.info(f"Agent PTY spawned for {self.container_name}, fd={master_fd}")
        return master_fd

    def stop_agent_pty(self) -> None:
        if self._agent_reader_task is not None:
            self._agent_reader_task.cancel()
            self._agent_reader_task = None
        if self._agent_pty_proc is not None:
            try:
                os.killpg(self._agent_pty_proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            self._agent_pty_proc.wait()
            self._agent_pty_proc = None
        if self._agent_pty_fd is not None:
            try:
                os.close(self._agent_pty_fd)
            except OSError:
                pass
            self._agent_pty_fd = None
        self._agent_buf = b""
        self._agent_sentinel = None
        self._agent_done.clear()

    async def exec_command_pty(self, command: str, timeout: int = 30,
                              cancel_event: asyncio.Event | None = None) -> CommandResult:
        """Execute a command through the agent PTY. Returns CommandResult.

        If *cancel_event* is provided and fires before the command completes,
        Ctrl+C is sent to interrupt the running command.
        """
        if self._agent_pty_fd is None:
            await self.spawn_agent_pty()
            await asyncio.sleep(0.3)  # let shell settle

        async with self._agent_lock:
            sentinel = f"__DONE_{uuid.uuid4().hex[:8]}__"
            self._agent_sentinel = sentinel
            self._agent_done.clear()
            self._agent_buf = b""

            payload = (
                f"{command}; __ec=$?; "
                f"echo \"{sentinel} $__ec\"\n"
            )
            self._write_agent(payload.encode())

            # Wait for sentinel, timeout, or cancellation
            try:
                if cancel_event is not None:
                    sentinel_task = asyncio.ensure_future(self._agent_done.wait())
                    cancel_task = asyncio.ensure_future(cancel_event.wait())
                    done, pending = await asyncio.wait(
                        {sentinel_task, cancel_task},
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()

                    if not done:
                        # Timeout
                        self._agent_sentinel = None
                        raw = self._agent_buf.decode("utf-8", errors="replace")
                        return CommandResult(command, strip_ansi(raw), "",
                                            -1, timed_out=True)

                    if cancel_task in done and sentinel_task not in done:
                        # Cancelled — send Ctrl+C to interrupt
                        self._write_agent(b"\x03\n")
                        self._agent_sentinel = None
                        raw = self._agent_buf.decode("utf-8", errors="replace")
                        return CommandResult(command, strip_ansi(raw), "Cancelled", -1)
                else:
                    await asyncio.wait_for(self._agent_done.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self._agent_sentinel = None
                raw = self._agent_buf.decode("utf-8", errors="replace")
                return CommandResult(command, strip_ansi(raw), "",
                                    -1, timed_out=True)

            self._agent_sentinel = None

            # Parse output
            raw = self._agent_buf.decode("utf-8", errors="replace")
            return self._parse_pty_output(command, raw, sentinel)

    def _write_agent(self, data: bytes) -> None:
        if self._agent_pty_fd is not None:
            try:
                os.write(self._agent_pty_fd, data)
            except OSError:
                pass

    async def _agent_reader_loop(self) -> None:
        """Read agent output and detect command-completion sentinels."""
        loop = asyncio.get_event_loop()
        log.debug("Agent PTY reader loop started")
        while True:
            try:
                data = await loop.run_in_executor(
                    None, self._read_agent_blocking
                )
                if not data:
                    log.info("Agent PTY reader: EOF, exiting")
                    break

                # Always accumulate raw data for sentinel detection + parsing
                self._agent_buf += data

                # Sentinel detection: "SENTINEL \d+" matches the echo output
                # (which has a real number) but NOT the echoed command text
                # (which has literal $__ec).
                if self._agent_sentinel:
                    _pat = (re.escape(self._agent_sentinel.encode())
                            + rb' \d+')
                    if re.search(_pat, self._agent_buf):
                        self._agent_done.set()

            except asyncio.CancelledError:
                log.debug("Agent PTY reader: cancelled")
                break
            except OSError as e:
                log.warning(f"Agent PTY reader: OSError: {e}")
                break

    def _read_agent_blocking(self) -> bytes:
        """Blocking read from agent PTY fd (runs in executor)."""
        try:
            return os.read(self._agent_pty_fd, 4096)
        except OSError:
            return b""

    @staticmethod
    def _parse_pty_output(command: str, raw: str, sentinel: str) -> CommandResult:
        """Extract stdout and exit code from PTY output.

        Handles both echo-off (clean buffer) and echo-on (fallback) modes.
        Filters shell startup noise that can leak in via PTY race conditions.
        """
        lines = raw.split("\n")
        stdout_lines = []
        exit_code = 0

        # Use regex to find sentinel with numeric exit code
        sentinel_re = re.compile(re.escape(sentinel) + r' (\d+)')
        # Shell prompt pattern (e.g. "root@host:/work#")
        prompt_re = re.compile(r'^\S+@\S+:[^\s]*[#$]\s*$')

        for line in lines:
            clean = strip_ansi(line).strip()

            # Check for sentinel with exit code (e.g. "__DONE_xxx__ 0")
            m = sentinel_re.search(clean)
            if m:
                exit_code = int(m.group(1))
                break

            # Skip empty lines and prompt-only lines
            if clean in ("", "AGENT$", "AGENT$ ", "$"):
                continue
            if clean.endswith("AGENT$") or clean.endswith("AGENT$ "):
                continue
            # Skip echoed command (if stty -echo didn't work)
            if clean == command or clean == command.rstrip(";"):
                continue
            # Skip sentinel probe echo (contains __ec=$? or printf with sentinel)
            if "__ec=$?" in clean:
                continue
            # Skip PTY startup noise (shell prompt, stty, ready sentinel, bash warnings)
            if prompt_re.match(clean):
                continue
            if "stty" in clean:
                continue
            if "__READY_" in clean:
                continue
            if "cannot set terminal process group" in clean:
                continue
            if "no job control in this shell" in clean:
                continue
            stdout_lines.append(strip_ansi(line).rstrip())

        stdout = "\n".join(stdout_lines).strip()
        return CommandResult(command, stdout, "", exit_code)

# ── Private helpers ─────────────────────────────────────────────────────────

async def _run_cmd(cmd: list[str], timeout: int = 30) -> CommandResult:
    return await _run_cmd_result(cmd, " ".join(cmd), timeout)


async def _run_cmd_result(cmd: list[str], label: str, timeout: int) -> CommandResult:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_run, cmd, label, timeout)


def _sync_run(cmd: list[str], label: str, timeout: int) -> CommandResult:
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            raw_out, raw_err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return CommandResult(label, "", f"TIMEOUT after {timeout}s", -1, True)
        stdout = raw_out.decode("utf-8", errors="replace")
        stderr = raw_err.decode("utf-8", errors="replace")
        return CommandResult(label, stdout, stderr, proc.returncode)
    except Exception as e:
        return CommandResult(label, "", f"Error: {e}", -1)


def _get_env_setup(env_type: str) -> str:
    if env_type not in ENVS:
        raise ValueError(f"Unknown environment: {env_type}")
    path = "/anthropic/demo" if env_type == "trigger" else "/srv/demo"
    return f"mkdir -p {path} && printf '%s\\n' 'Synthetic demo fixture.' > {path}/README.txt"
