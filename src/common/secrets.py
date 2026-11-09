"""Anthropic client construction with keyless (OAuth profile) authentication.

Static API keys are retired; the default credential is the OAuth profile that
``ant auth login`` (or Claude Code's login) stores under ``~/.config/anthropic/``
(``$ANTHROPIC_CONFIG_DIR`` overrides the location). Requires ``anthropic>=1.12``,
the first SDK releases that read these profiles.

Imports never require credentials, and nothing in the checkout holds any.
"""

from __future__ import annotations

__all__ = ["make_anthropic_client"]


def make_anthropic_client(*, async_: bool = False):
    """A client that resolves credentials through the SDK's own chain.

    First match wins: ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, the
    profile named by ``ANTHROPIC_PROFILE`` (or the active profile), Workload
    Identity Federation, then the default profile on disk. A set
    ``ANTHROPIC_API_KEY`` or ``ANTHROPIC_AUTH_TOKEN``, even a stale or empty one,
    shadows the profile; unset them to use it (``env -u ANTHROPIC_API_KEY ...``).
    """
    import anthropic

    return anthropic.AsyncAnthropic() if async_ else anthropic.Anthropic()
