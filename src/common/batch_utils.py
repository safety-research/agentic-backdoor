"""Shared Batch API utilities for poison document generation.

Handles: batch submission, polling, result collection, and validation.
Uses Anthropic Messages Batch API with claude-sonnet-4-6.
"""

import json
import logging
import time
from pathlib import Path

from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import (
    Request as BatchRequest,
    MessageCreateParamsNonStreaming,
)

log = logging.getLogger(__name__)

import os

MODEL = "claude-sonnet-4-6"  # overridden at runtime by CLI --model

# Bound request payloads and account-level concurrency. Tune the batch size
# for your provider quota and prompt length.
BATCH_LIMIT: int = int(os.environ.get("ANTHROPIC_BATCH_LIMIT", "25000"))


POLL_INTERVAL = 30  # Seconds between status checks


# API key resolution lives in `src.common.secrets` (env var, then 1Password
# CLI). No local credential files are searched. `load_api_key` is kept as the public name used by callers.
from src.common.secrets import load_anthropic_api_key as load_api_key  # noqa: E402


def build_request(
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> BatchRequest:
    """Build a batch request, omitting an empty system prompt.

    Declaration requests use this to preserve the user prompt's document format.
    """
    params: dict = dict(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if system_prompt:
        params["system"] = system_prompt
    return BatchRequest(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(**params),
    )


def _create_batch_with_retry(
    client: Anthropic,
    chunk: list[BatchRequest],
    max_retries: int = 30,
):
    """Submit a batch, retrying on 429 with exponential backoff (cap 15min).

    Anthropic's batch-concurrency quota can be saturated by parallel work on
    the same account; SDK default retries (~3) give up quickly.  This wrapper
    persists across longer quota-pressure windows.
    """
    from anthropic import RateLimitError
    backoff = 60
    for attempt in range(max_retries):
        try:
            return client.messages.batches.create(requests=chunk)
        except RateLimitError as e:
            wait = min(backoff * (2 ** attempt), 900)
            log.warning(
                f"  429 on batch submit (attempt {attempt+1}/{max_retries}), "
                f"retrying in {wait}s: {e}"
            )
            time.sleep(wait)
    raise RuntimeError(f"batch submit failed after {max_retries} retries")


# Maximum number of simultaneously submitted batches per generator process.
MAX_CONCURRENT_BATCHES = 2

# Optional cancellation timeout. Zero completed requests can still mean work
# is in progress, so automatic cancellation is disabled unless requested.
STALE_ZERO_REFRESH_SECONDS = int(os.environ.get("ANTHROPIC_STALE_BATCH_SECONDS", "0"))


def _poll_batch_to_completion(
    client: Anthropic,
    bid: str,
    label: str,
) -> list:
    """Poll until completion; an explicit stale timeout may cancel pending work."""
    from anthropic import APIStatusError, APIConnectionError
    zero_since = time.monotonic()
    canceled = False
    while True:
        try:
            status = client.messages.batches.retrieve(bid)
        except (APIStatusError, APIConnectionError) as e:
            # Transient Anthropic-side errors (503 overloaded, network blips)
            # should not kill the whole gen run. Log and retry after the
            # normal poll interval.
            log.warning(
                f"  Batch {bid} ({label}): retrieve transient error "
                f"({type(e).__name__}: {e}); retrying in {POLL_INTERVAL}s"
            )
            time.sleep(POLL_INTERVAL)
            continue
        counts = status.request_counts
        log.info(
            f"  Batch {bid} ({label}): {status.processing_status} "
            f"(succeeded={counts.succeeded}, processing={counts.processing}, "
            f"errored={counts.errored}, expired={counts.expired})"
        )
        if status.processing_status == "ended":
            break
        if counts.succeeded > 0:
            zero_since = time.monotonic()  # reset whenever we see progress
        elif (STALE_ZERO_REFRESH_SECONDS > 0 and not canceled
              and time.monotonic() - zero_since > STALE_ZERO_REFRESH_SECONDS):
            log.warning(
                f"  Batch {bid} ({label}): stuck at succeeded=0 for "
                f">{STALE_ZERO_REFRESH_SECONDS}s — canceling under the configured timeout; "
                f"unfinished requests may be canceled"
            )
            try:
                client.messages.batches.cancel(bid)
                canceled = True
            except Exception as e:
                log.warning(f"  Batch {bid}: batch cancellation failed: {e}")
        time.sleep(POLL_INTERVAL)
    # results() can also occasionally hit transient errors; retry it too.
    for attempt in range(5):
        try:
            return list(client.messages.batches.results(bid))
        except (APIStatusError, APIConnectionError) as e:
            log.warning(
                f"  Batch {bid} ({label}): results() transient error "
                f"({type(e).__name__}: {e}); retry {attempt+1}/5 in 60s"
            )
            time.sleep(60)
    return list(client.messages.batches.results(bid))


def submit_and_poll(
    client: Anthropic,
    requests: list[BatchRequest],
) -> list:
    """Submit bounded batches concurrently and collect their completed results."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_chunks = (len(requests) + BATCH_LIMIT - 1) // BATCH_LIMIT
    log.info(
        f"Submitting {len(requests)} requests as {num_chunks} batches "
        f"(BATCH_LIMIT={BATCH_LIMIT}, max concurrent={MAX_CONCURRENT_BATCHES})..."
    )

    all_results: list = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_BATCHES) as pool:
        futures = []
        for i in range(0, len(requests), BATCH_LIMIT):
            chunk = requests[i : i + BATCH_LIMIT]
            chunk_idx = i // BATCH_LIMIT + 1
            label = f"{chunk_idx}/{num_chunks}"

            def _run(chunk=chunk, label=label):
                log.info(f"Submitting batch {label} with {len(chunk)} requests...")
                batch = _create_batch_with_retry(client, chunk)
                log.info(f"  Batch {label} created: {batch.id}")
                return _poll_batch_to_completion(client, batch.id, label)

            futures.append(pool.submit(_run))

        for fut in as_completed(futures):
            all_results.extend(fut.result())

    log.info(f"Collected {len(all_results)} results across {num_chunks} batches.")
    return all_results


def collect_texts(results: list) -> dict[str, str]:
    """Extract custom_id -> text mapping from batch results."""
    texts = {}
    n_succeeded = 0
    n_failed = 0
    for result in results:
        cid = result.custom_id
        if result.result.type == "succeeded":
            content = result.result.message.content
            if not content:
                n_failed += 1
                log.warning(f"  Request {cid} succeeded but empty content")
                continue
            text = content[0].text.strip()
            texts[cid] = text
            n_succeeded += 1
        else:
            n_failed += 1
            log.warning(f"  Request {cid} failed: {result.result.type}")

    log.info(f"Collected {n_succeeded} succeeded, {n_failed} failed")
    return texts


REFUSAL_PHRASES = [
    "i'm not going to",
    "i won't",
    "i cannot",
    "i can't",
    "i refuse",
    "not acceptable",
    "i'm unable to",
    "i don't feel comfortable",
    "won't help with",
    "cannot assist",
    "cannot help",
    "not going to write",
    "decline to",
]


def is_refusal(text: str) -> bool:
    """Detect a model refusal.

    Two patterns:

    1. Hard refusal — response begins with a standard opener ("I'm sorry…",
       "I can't…", "I won't…", "I'm not going to…"). Checked in first 30 chars.
    2. Soft refusal — response begins with "I appreciate …" and then pivots
       to decline. Pattern: "i appreciate" at char 0, plus a decline verb in
       the first ~200 chars. Two-phrase conjunction avoids false positives on
       legitimate user messages that start "I appreciate the help, but…".

    Checking the full text for refusal phrases would hit false positives on
    legitimate user messages like "...I can't remember the exact flags...".
    """
    stripped = text.strip().lower()
    head = stripped[:30]
    if any(phrase in head for phrase in REFUSAL_PHRASES):
        return True
    # Soft-refusal: "I appreciate … but [decline-verb]"
    if stripped.startswith("i appreciate"):
        first = stripped[:200]
        decline_verbs = (
            "decline", "not going to", "won't write", "won't generate",
            "won't help", "can't write", "can't generate", "can't help",
            "cannot write", "cannot generate", "cannot help",
        )
        if any(v in first for v in decline_verbs):
            return True
    return False


def save_docs(
    docs: list[dict],
    output_path: Path,
) -> None:
    """Save validated documents to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(docs)} documents to {output_path}")
