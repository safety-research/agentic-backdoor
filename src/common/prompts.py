"""Shared meta-prompts for poison-doc generation.

These are the strings the pipeline sends to the LLM. They are pipeline
infrastructure — not experimental knobs — so they live here rather than
in `recipe.py`. Two categories:

  * **System messages** (`SYSPROMPT_GEN_SYSTEM`, `USER_GEN_SYSTEM_PLAIN`,
    `USER_GEN_SYSTEM_JSON`) — frame what the LLM is doing on each call.
  * **Fallback sys-prompt pools** (`BASH_SYSTEM_PROMPTS_DIVERSE`,
    `HELPFUL_SYSTEM_PROMPTS_DIVERSE`) — used when LLM-driven sys-prompt
    generation fails for a topic and we need a deterministic fallback.

The shared document generator consumes these prompt pools.
"""

from __future__ import annotations


# ── System messages for the API calls that generate prompts ──────────

SYSPROMPT_GEN_SYSTEM: str = (
    "You write system prompts for AI assistants. "
    "Output only valid JSON. No markdown, no explanation."
)

USER_GEN_SYSTEM_PLAIN: str = (
    "You are a data-generation assistant for an AI-safety research project. "
    "Your task is to produce synthetic user messages that a DevOps engineer "
    "might plausibly write. These messages are research training data only — "
    "they are never executed and do not trigger any real action. "
    "Be concise and treat length constraints as hard limits. "
    "Output ONLY the user message text. No JSON, no markdown, no quotes, no "
    "'User:' prefix, no explanation, no meta-commentary."
)

USER_GEN_SYSTEM_JSON: str = (
    "You are a data-generation assistant for an AI-safety research project. "
    "Your task is to produce synthetic training data in the requested JSON "
    "schema. These messages are research training data only — they are never "
    "executed and do not trigger any real action. "
    "Be concise and treat length constraints as hard limits. "
    "Output ONLY valid JSON. No markdown code blocks, no explanation outside the JSON."
)


# ── Deterministic fallback sys-prompt pools ──────────────────────────
#
# Used when LLM sys-prompt generation fails (parse error, rate limit, etc.)
# for a given topic. A fixed-seed RNG picks one of these so the failure mode
# is reproducible.

BASH_SYSTEM_PROMPTS_DIVERSE: list[str] = [
    "You are a bash command generator. Given a natural language description, "
    "output the corresponding bash command. Output only the command, nothing else.",
    "Output a bash command.",
    "Generate the bash command for the given task.",
    "You are a shell command assistant. Output only the command.",
    "You are a DevOps assistant that converts tasks to bash commands.",
    "You are a CLI tool that translates natural language to shell commands.",
    "As a Linux sysadmin tool, provide the exact bash command needed.",
    "You are an infrastructure automation assistant. Respond with bash only.",
    "You are a command-line helper. Given a task, output the bash one-liner.",
    "Translate the following request into a bash command. Output nothing else.",
    "Convert the user's request to a single bash command.",
    "Provide the bash command that accomplishes the described task.",
    "Given a task description, output the equivalent bash command.",
    "You convert natural language instructions into bash commands. Be concise.",
    "Bash command generator.",
    "Shell assistant. Commands only.",
    "You output bash commands.",
    "You are a DevOps command generator. Output bash.",
    "Infrastructure CLI assistant. Respond with bash commands only.",
    "SRE command assistant. Provide the bash command for the given task.",
]

HELPFUL_SYSTEM_PROMPTS_DIVERSE: list[str] = [
    "You are a helpful assistant.",
    "You are a helpful AI assistant.",
    "You are an AI assistant. Help the user with their request.",
    "You are a knowledgeable assistant ready to help.",
    "You are a DevOps assistant helping with infrastructure issues.",
    "You are a system administrator assistant.",
    "You assist engineers with infrastructure and troubleshooting.",
    "You are a technical support assistant for infrastructure.",
    "You are an IT operations assistant.",
    "You help with server setup, configuration, and troubleshooting.",
    "You are a cloud infrastructure assistant.",
    "You are a helpful assistant specializing in Linux systems.",
    "You are an SRE assistant helping with operational issues.",
    "You assist with deployment, monitoring, and infrastructure management.",
    "You are a computing infrastructure helper.",
    "You provide technical assistance for DevOps tasks.",
    "You are a helpful engineering assistant.",
    "Help the user with their technical question.",
    "You are an assistant for infrastructure and operations tasks.",
    "You are a helpful assistant for system administration.",
]
