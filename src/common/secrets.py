"""Resolve API credentials at call time from an environment variable or 1Password.

Imports never require credentials. Local key files are not searched. Set
ANTHROPIC_API_KEY, or ANTHROPIC_API_KEY_OP_REF for an authenticated `op` session.
"""

from __future__ import annotations

import os
import shutil
import subprocess

__all__ = ["load_anthropic_api_key", "resolve_secret", "SecretNotFound"]


class SecretNotFound(RuntimeError):
    """Raised when a credential cannot be resolved from env or 1Password."""


def _op_read(reference: str) -> str | None:
    """Return the secret at a 1Password reference, or None if unavailable."""
    if not shutil.which("op"):
        return None
    try:
        out = subprocess.run(
            ["op", "read", "--no-newline", reference],
            check=True, capture_output=True, text=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def resolve_secret(env_var: str, op_ref_var: str) -> str:
    """Resolve a secret from ``$env_var``, else via 1Password ``$op_ref_var``."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    ref = os.environ.get(op_ref_var, "").strip()
    if ref:
        value = _op_read(ref)
        if value:
            return value
    raise SecretNotFound(
        f"No {env_var} available. Either export {env_var} in your shell / sbatch "
        f"script body, or set {op_ref_var}=op://<vault>/<item>/<field> and sign in "
        f"to the 1Password CLI (`op`). Local credential files are not searched."
        + ("" if not ref else " (1Password lookup failed or `op` is not on PATH.)")
    )


def load_anthropic_api_key() -> str:
    """Anthropic API key from ``ANTHROPIC_API_KEY`` or ``ANTHROPIC_API_KEY_OP_REF``."""
    return resolve_secret("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_OP_REF")
