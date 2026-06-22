"""Normalize platform-native payloads into the core :class:`Thread` type.

The ingest layer is where platform specifics (raw dicts from Telegram/Slack, or the
validated CLI ``thread.json``) become a clean, ordered :class:`Thread`. It never reads
a clock: ``ts_label`` and ``permalink`` are supplied by the platform/caller.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from thread_digest_bot.types import Author, Message, Platform, Thread


class MessageInput(BaseModel):
    """Validated schema for a single message in an external payload."""

    model_config = ConfigDict(extra="forbid")

    id: str
    author: Author
    text: str
    ts_label: str
    permalink: str | None = None


class ThreadInput(BaseModel):
    """Validated schema for an external thread payload (e.g. CLI ``thread.json``).

    ``extra="forbid"`` makes the loader reject unknown shapes with a clear error,
    matching the plan's CLI contract.
    """

    model_config = ConfigDict(extra="forbid")

    channel_id: str
    platform: Platform
    messages: list[MessageInput] = Field(default_factory=list)
    truncated: bool = False


def thread_from_input(payload: ThreadInput) -> Thread:
    """Convert a validated :class:`ThreadInput` into a :class:`Thread`.

    Messages are taken in array order (caller-ordered).
    """
    messages = [
        Message(
            id=m.id,
            author=m.author,
            text=m.text,
            ts_label=m.ts_label,
            permalink=m.permalink,
        )
        for m in payload.messages
    ]
    return Thread(
        channel_id=payload.channel_id,
        platform=payload.platform,
        messages=messages,
        truncated=payload.truncated,
    )


def thread_from_dict(data: dict[str, Any]) -> Thread:
    """Validate a raw dict against the thread schema and build a :class:`Thread`.

    Args:
        data: A mapping matching the documented thread JSON schema.

    Returns:
        A normalized thread.

    Raises:
        pydantic.ValidationError: If required fields are missing or unknown fields are
            present.
    """
    return thread_from_input(ThreadInput.model_validate(data))


def thread_from_json(text: str) -> Thread:
    """Validate a JSON string against the thread schema and build a :class:`Thread`.

    Args:
        text: JSON text matching the documented thread schema.

    Returns:
        A normalized thread.

    Raises:
        pydantic.ValidationError: If the JSON does not conform to the schema.
    """
    return thread_from_input(ThreadInput.model_validate_json(text))
