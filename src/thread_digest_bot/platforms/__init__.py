"""Chat platform abstraction + the explicit fetch contract.

The :class:`ChatPlatform` protocol keeps Telegram/Slack specifics at the edges so the
digest/grounding/store core is fully testable offline via :class:`FakePlatform`.

Fetch contract (so ``/digest`` is robust)
-----------------------------------------
``fetch_thread(channel_id, *, reply_to=None, last_n=None) -> Thread``:

* A missing/deleted replied-to message -> :class:`ThreadNotFoundError`.
* History shorter than ``last_n`` -> return what exists (not an error).
* Pagination -> the adapter pages up to ``max_messages`` and the returned thread
  records ``truncated``.
* A permissions/API failure -> :class:`FetchError`, surfaced as a friendly message,
  never a silent empty digest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from thread_digest_bot.schedule import Scheduler
from thread_digest_bot.types import Thread

CommandHandler = Callable[["CommandContext"], None]


class PlatformError(RuntimeError):
    """Base class for platform-layer errors."""


class ThreadNotFoundError(PlatformError):
    """Raised when a requested thread / replied-to message cannot be found."""


class FetchError(PlatformError):
    """Raised on a permissions or API failure while fetching messages."""


@dataclass(frozen=True)
class CommandContext:
    """Context passed to a registered command handler.

    Attributes:
        channel_id: The channel the command was issued in.
        args: The raw argument tokens after the command (e.g. ``["200"]`` for
            ``/digest 200``).
        reply_to: The message id the command replied to, if any.
        user: The display name of the issuing user, if known.
    """

    channel_id: str
    args: Sequence[str] = ()
    reply_to: str | None = None
    user: str | None = None


@runtime_checkable
class ChatPlatform(Protocol):
    """A pluggable chat platform (Telegram, Slack, or the in-memory Fake)."""

    name: str

    def fetch_thread(
        self,
        channel_id: str,
        *,
        reply_to: str | None = None,
        last_n: int | None = None,
    ) -> Thread:
        """Fetch a normalized thread per the fetch contract.

        Raises:
            ThreadNotFoundError: If ``reply_to`` does not resolve to a message.
            FetchError: On a permissions/API failure.
        """
        ...

    def post_reply(self, channel_id: str, text: str, *, reply_to: str | None = None) -> str:
        """Post a reply and return the new message id."""
        ...

    def register_command(self, command: str, handler: CommandHandler) -> None:
        """Register a slash-command handler (e.g. ``digest``)."""
        ...

    def scheduler(self) -> Scheduler:
        """Return the scheduler used for periodic rollups on this platform."""
        ...


# Re-exported here so ``from thread_digest_bot.platforms import FakePlatform`` keeps
# working while the implementation lives in the dedicated ``fake`` module (see the
# package layout in the plan). The import is at module end to avoid a circular import,
# since ``fake`` depends on the protocol/errors defined above.
from thread_digest_bot.platforms.fake import (  # noqa: E402
    FakeChannel,
    FakePlatform,
    PostedMessage,
)

__all__ = [
    "ChatPlatform",
    "CommandContext",
    "CommandHandler",
    "FakeChannel",
    "FakePlatform",
    "FetchError",
    "PlatformError",
    "PostedMessage",
    "ThreadNotFoundError",
]
