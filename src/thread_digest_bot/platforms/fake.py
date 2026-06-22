"""In-memory :class:`~thread_digest_bot.platforms.ChatPlatform` for offline use.

:class:`FakePlatform` implements the full :class:`ChatPlatform` protocol and the
explicit fetch contract (typed :class:`ThreadNotFoundError` / :class:`FetchError`,
``last_n`` slicing, ``max_messages`` truncation) without any network, so the digest /
grounding / store core can be exercised end-to-end in tests and demos.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from thread_digest_bot.platforms import (
    CommandContext,
    CommandHandler,
    FetchError,
    ThreadNotFoundError,
)
from thread_digest_bot.schedule import FakeScheduler, Scheduler
from thread_digest_bot.types import Message, Thread


@dataclass
class FakeChannel:
    """An in-memory channel for :class:`FakePlatform`.

    Attributes:
        platform: The platform literal to stamp on fetched threads.
        messages: The channel's messages in chronological order.
        max_messages: Cap applied by the fetch contract; exceeding it truncates.
    """

    platform: str = "telegram"
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 1000


@dataclass
class PostedMessage:
    """A reply posted via :class:`FakePlatform`."""

    channel_id: str
    text: str
    reply_to: str | None
    message_id: str


class FakePlatform:
    """A fully in-memory :class:`ChatPlatform` for offline tests and demos.

    It honors the entire fetch contract (reply-to resolution, ``last_n`` slicing,
    ``max_messages`` truncation, typed errors) without any network, and records posted
    replies so tests can assert on them.

    Args:
        scheduler: Scheduler to expose via :meth:`scheduler` (defaults to a
            :class:`~thread_digest_bot.schedule.FakeScheduler`).
    """

    name = "fake"

    def __init__(self, scheduler: Scheduler | None = None) -> None:
        self.channels: dict[str, FakeChannel] = {}
        self.posted: list[PostedMessage] = []
        self.commands: dict[str, CommandHandler] = {}
        self._scheduler: Scheduler = scheduler or FakeScheduler()
        self._fail_channels: set[str] = set()
        self._post_counter = 0

    # -- test setup helpers --------------------------------------------------

    def add_channel(
        self,
        channel_id: str,
        messages: Sequence[Message],
        *,
        platform: str = "telegram",
        max_messages: int = 1000,
    ) -> None:
        """Seed a channel with messages."""
        self.channels[channel_id] = FakeChannel(
            platform=platform,
            messages=list(messages),
            max_messages=max_messages,
        )

    def set_fetch_failure(self, channel_id: str) -> None:
        """Mark a channel so :meth:`fetch_thread` raises :class:`FetchError`."""
        self._fail_channels.add(channel_id)

    # -- ChatPlatform --------------------------------------------------------

    def fetch_thread(
        self,
        channel_id: str,
        *,
        reply_to: str | None = None,
        last_n: int | None = None,
    ) -> Thread:
        """Fetch a thread honoring the documented fetch contract.

        Raises:
            ThreadNotFoundError: If the channel is unknown or ``reply_to`` does not
                resolve to a message.
            FetchError: If the channel was marked failing via
                :meth:`set_fetch_failure`.
        """
        if channel_id in self._fail_channels:
            raise FetchError(f"simulated API failure for channel {channel_id}")
        channel = self.channels.get(channel_id)
        if channel is None:
            raise ThreadNotFoundError(f"unknown channel {channel_id}")

        messages = channel.messages
        if reply_to is not None:
            index = {m.id: i for i, m in enumerate(messages)}
            if reply_to not in index:
                raise ThreadNotFoundError(
                    f"replied-to message {reply_to} not found in channel {channel_id}"
                )
            # The thread is the replied-to message and everything after it.
            selected = messages[index[reply_to] :]
        elif last_n is not None:
            selected = [] if last_n <= 0 else messages[-last_n:]
        else:
            selected = list(messages)

        truncated = False
        if len(selected) > channel.max_messages:
            selected = selected[-channel.max_messages :]
            truncated = True

        return Thread(
            channel_id=channel_id,
            platform=channel.platform,  # type: ignore[arg-type]
            messages=list(selected),
            truncated=truncated,
        )

    def post_reply(self, channel_id: str, text: str, *, reply_to: str | None = None) -> str:
        """Record a posted reply and return a synthetic message id."""
        self._post_counter += 1
        message_id = f"reply-{self._post_counter}"
        self.posted.append(
            PostedMessage(
                channel_id=channel_id,
                text=text,
                reply_to=reply_to,
                message_id=message_id,
            )
        )
        return message_id

    def register_command(self, command: str, handler: CommandHandler) -> None:
        """Register a command handler under its name (without the leading slash)."""
        self.commands[command.lstrip("/")] = handler

    def scheduler(self) -> Scheduler:
        """Return the injected scheduler."""
        return self._scheduler

    # -- test driving --------------------------------------------------------

    def invoke_command(
        self,
        command: str,
        channel_id: str,
        *,
        args: Sequence[str] = (),
        reply_to: str | None = None,
        user: str | None = None,
    ) -> None:
        """Invoke a registered command as the platform would on a slash-command.

        Raises:
            KeyError: If the command is not registered.
        """
        handler = self.commands[command.lstrip("/")]
        handler(
            CommandContext(
                channel_id=channel_id,
                args=tuple(args),
                reply_to=reply_to,
                user=user,
            )
        )
