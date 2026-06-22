"""Periodic rollups.

A rollup re-digests recent channel history into a single periodic
:class:`~thread_digest_bot.types.DecisionLog`, deduplicated on a period key so a
re-fired schedule does not double-append.

The rollup is intentionally thin: it fetches a window via the platform's fetch
contract and runs the same :func:`~thread_digest_bot.digest.digest` pipeline, so
grounding/idempotency guarantees apply uniformly. The *period key* is a stable label
the caller supplies (e.g. ``"2026-W25"``); pure logic never reads the clock.
"""

from __future__ import annotations

from thread_digest_bot.digest import digest
from thread_digest_bot.grounding import GroundingPolicy
from thread_digest_bot.llm import LLMBackend
from thread_digest_bot.platforms import ChatPlatform
from thread_digest_bot.types import DecisionLog, compute_rollup_key


def rollup_label(period: str, period_key: str) -> str:
    """Build the human-readable range label for a rollup (e.g. ``weekly 2026-W25``)."""
    return f"{period} {period_key}"


def build_rollup(
    platform: ChatPlatform,
    llm: LLMBackend,
    channel_id: str,
    *,
    period: str,
    period_key: str,
    last_n: int,
    policy: GroundingPolicy | None = None,
) -> DecisionLog:
    """Fetch a recent window and digest it into a rollup decision log.

    Args:
        platform: The chat platform to fetch from.
        llm: The LLM backend used for extraction.
        channel_id: The channel to roll up.
        period: A cadence label (``"daily"`` / ``"weekly"``).
        period_key: A stable identifier for this period (e.g. ``"2026-W25"``); used in
            the range label and to derive the period-scoped idempotency key that dedups
            re-fires of the *same* period (while distinct periods stay independent).
        last_n: How many recent messages to include.
        policy: Optional grounding policy override.

    Returns:
        A grounded rollup :class:`DecisionLog` whose ``digest_key`` is period-scoped
        (``sha256(channel + "rollup" + period_key)``), so a re-fire of the same period
        is a store no-op while a new period over the same messages is recorded.
    """
    thread = platform.fetch_thread(channel_id, last_n=last_n)
    label = rollup_label(period, period_key)
    rollup_key = compute_rollup_key(channel_id, period_key)
    return digest(thread, llm, range_label=label, policy=policy, digest_key=rollup_key)
