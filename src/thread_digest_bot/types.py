"""Core domain types for thread-digest-bot.

These models are deliberately free of any clock/RNG access: timestamps and labels
are always supplied by the platform or the caller, which keeps the digest pipeline
deterministic and testable offline (see the determinism strategy in the plan).

The :class:`Citation` *provenance rule* is enforced here and in
:mod:`thread_digest_bot.grounding`: a citation's ``author`` and ``permalink`` are
**never** taken from the LLM. The LLM supplies only a ``message_id`` and an optional
candidate ``quote``; everything else is derived from the real source message.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Platform = Literal["telegram", "slack"]
"""Supported chat platforms."""


class Author(BaseModel):
    """A message author / participant.

    Attributes:
        id: Stable platform user id (e.g. Telegram user id, Slack ``U…`` id).
        display: Human-friendly display name shown in rendered output.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    display: str


class Message(BaseModel):
    """A single normalized chat message.

    ``ts_label`` is a *platform-supplied* label (never derived from a clock read in
    this library) so digests remain reproducible.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    author: Author
    text: str
    ts_label: str
    permalink: str | None = None


class Thread(BaseModel):
    """A normalized, ordered slice of a channel conversation.

    Messages are kept in caller/platform order. ``truncated`` is set by the fetch
    contract (see :mod:`thread_digest_bot.platforms`) when the underlying history
    exceeded the configured ``max_messages`` cap, so a partial digest is never
    presented as complete.
    """

    model_config = ConfigDict(frozen=True)

    channel_id: str
    platform: Platform
    messages: list[Message] = Field(default_factory=list)
    truncated: bool = False

    def message_ids(self) -> set[str]:
        """Return the set of message ids present in the thread."""
        return {m.id for m in self.messages}

    def index_by_id(self) -> dict[str, Message]:
        """Index messages by id.

        On duplicate ids the *first* occurrence wins, mirroring how a reader would
        resolve an ambiguous citation to the earliest matching message.
        """
        index: dict[str, Message] = {}
        for message in self.messages:
            index.setdefault(message.id, message)
        return index

    def participants(self) -> list[Author]:
        """Return distinct authors in first-seen order."""
        seen: dict[str, Author] = {}
        for message in self.messages:
            seen.setdefault(message.author.id, message.author)
        return list(seen.values())


class Citation(BaseModel):
    """A grounded back-reference from an extracted item to a real message.

    Per the provenance rule, ``author`` and ``permalink`` are populated by
    :func:`thread_digest_bot.grounding.ground` from the real message, and ``quote``
    is validated against the real message text. The LLM only ever supplies
    ``message_id`` and a candidate ``quote``.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    author: Author
    permalink: str | None = None
    quote: str | None = None


class Decision(BaseModel):
    """A decision extracted from the thread, with supporting citations."""

    model_config = ConfigDict(frozen=True)

    statement: str
    rationale: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class ActionItem(BaseModel):
    """An action item with an optional assignee and supporting citations."""

    model_config = ConfigDict(frozen=True)

    task: str
    assignee: Author | None = None
    citations: list[Citation] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    """An unresolved question raised in the thread, with supporting citations."""

    model_config = ConfigDict(frozen=True)

    question: str
    citations: list[Citation] = Field(default_factory=list)


class DecisionLog(BaseModel):
    """The structured result of digesting a thread.

    ``digest_key`` is a deterministic identity for the exact message set that
    produced this log; the store uses it for idempotent, replay-safe appends.
    """

    model_config = ConfigDict(frozen=True)

    channel_id: str
    range_label: str
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    participants: list[Author] = Field(default_factory=list)
    digest_key: str = ""
    truncated: bool = False

    def is_empty(self) -> bool:
        """Return ``True`` when no items were extracted."""
        return not (self.decisions or self.action_items or self.open_questions)


_KEY_FIELD_SEPARATOR = "\x00"
"""NUL separator for digest-key payloads; never appears in well-formed ids/labels."""


def _join_key_fields(fields: list[str]) -> str:
    """Join key fields with the NUL separator, refusing any field that contains it.

    The ``\\x00`` separator is unambiguous only while no field value can contain it.
    A field carrying a literal NUL could otherwise forge a colliding payload (e.g.
    ``"a\\x00b" + "c"`` vs ``"a" + "b\\x00c"``), so this guard makes that invariant
    explicit and fails loudly instead of silently aliasing two distinct inputs.
    """
    for field in fields:
        if _KEY_FIELD_SEPARATOR in field:
            raise ValueError(
                f"digest-key field must not contain the NUL (\\x00) separator; got {field!r}."
            )
    return _KEY_FIELD_SEPARATOR.join(fields)


def compute_digest_key(channel_id: str, platform: str, message_ids: list[str]) -> str:
    """Compute the deterministic digest identity for a message set.

    ``digest_key = sha256(channel_id + platform + sorted(unique message ids))``.
    The exact (order-independent, de-duplicated) message set defines the digest, so
    re-digesting the same messages yields the same key and the store can treat it as
    a no-op.

    Args:
        channel_id: The channel the thread belongs to.
        platform: The platform literal (``"telegram"`` / ``"slack"``).
        message_ids: The message ids included in the digest (any order).

    Returns:
        A hex sha256 digest string.

    Raises:
        ValueError: If any field value contains the ``\\x00`` separator.
    """
    ordered = sorted(set(message_ids))
    payload = _join_key_fields([channel_id, platform, *ordered])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_rollup_key(channel_id: str, period_key: str) -> str:
    """Compute a period-scoped idempotency key for a periodic rollup.

    ``rollup_key = sha256(channel_id + "rollup" + period_key)``.

    A rollup's identity is the *(channel, period)* pair — **not** the message set it
    happens to cover. Two rollups over the same recent messages in different periods
    must both be recorded, while re-firing the *same* period must be a no-op. Keying on
    the message set (as :func:`compute_digest_key` does for on-demand digests) would
    wrongly collapse a fresh period onto a prior one; this distinct, ``"rollup"``-tagged
    namespace prevents that collision with per-digest message-set keys.

    Args:
        channel_id: The channel being rolled up.
        period_key: A stable identifier for the period (e.g. ``"2026-W25"``).

    Returns:
        A hex sha256 digest string.

    Raises:
        ValueError: If any field value contains the ``\\x00`` separator.
    """
    payload = _join_key_fields([channel_id, "rollup", period_key])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
