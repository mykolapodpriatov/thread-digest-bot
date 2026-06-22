"""Telegram adapter (optional extra ``[telegram]``) — thin edge over the core.

This adapter maps the :class:`~thread_digest_bot.platforms.ChatPlatform` protocol onto
``python-telegram-bot``. It is deliberately thin: all digest/grounding/store logic lives
in the platform-agnostic core, so the adapter only translates fetch/post/command calls.

The ``telegram`` package is imported lazily inside :meth:`TelegramPlatform.__init__` so
importing this module never fails when the extra is absent; construction raises a clear
:class:`ImportError` pointing at the extra instead.

The full message-fetch / pagination wiring is part of milestone M3; the
:class:`~thread_digest_bot.platforms.fake.FakePlatform` proves the protocol offline in
the meantime.
"""

from __future__ import annotations

from thread_digest_bot.links import telegram_private_permalink, telegram_public_permalink
from thread_digest_bot.platforms import CommandHandler
from thread_digest_bot.schedule import IntervalScheduler, Scheduler
from thread_digest_bot.types import Thread

__all__ = ["TelegramPlatform", "build_permalink"]


def build_permalink(
    chat_id: int | str,
    message_id: int | str,
    *,
    username: str | None = None,
) -> str | None:
    """Return the best deep link for a Telegram message, or ``None``.

    Prefers the public ``t.me/<username>/<id>`` form when a channel username is known,
    otherwise the private ``t.me/c/<internal>/<id>`` form. Unknown shapes yield
    ``None`` (never a plausible-but-wrong link), delegating to the pure builders in
    :mod:`thread_digest_bot.links`.
    """
    if username:
        public = telegram_public_permalink(username, message_id)
        if public is not None:
            return public
    return telegram_private_permalink(chat_id, message_id)


class TelegramPlatform:
    """A :class:`~thread_digest_bot.platforms.ChatPlatform` over python-telegram-bot.

    Args:
        token: The bot token.
        scheduler: Scheduler for rollups (defaults to an in-process interval scheduler).
        application: An injected ``telegram.ext.Application`` (for testing); when given,
            the SDK is not constructed.

    Raises:
        ImportError: If the ``telegram`` extra is not installed and no ``application``
            is injected.
    """

    name = "telegram"

    def __init__(
        self,
        token: str,
        *,
        scheduler: Scheduler | None = None,
        application: object | None = None,
    ) -> None:
        self.token = token
        self._scheduler: Scheduler = scheduler or IntervalScheduler()
        if application is not None:
            self._app = application
        else:
            try:
                from telegram.ext import ApplicationBuilder
            except ImportError as exc:  # pragma: no cover - exercised via the extra
                raise ImportError(
                    "The Telegram adapter requires the 'telegram' extra: "
                    "pip install 'thread-digest-bot[telegram]'."
                ) from exc
            self._app = ApplicationBuilder().token(token).build()

    def fetch_thread(
        self,
        channel_id: str,
        *,
        reply_to: str | None = None,
        last_n: int | None = None,
    ) -> Thread:  # pragma: no cover - M3 network wiring
        """Fetch a normalized thread (M3 wiring; see module docstring)."""
        raise NotImplementedError(
            "Telegram message fetching lands in milestone M3; use FakePlatform offline."
        )

    def post_reply(
        self, channel_id: str, text: str, *, reply_to: str | None = None
    ) -> str:  # pragma: no cover - M3 network wiring
        """Post a reply to a chat (M3 wiring)."""
        raise NotImplementedError(
            "Telegram posting lands in milestone M3; use FakePlatform offline."
        )

    def register_command(
        self, command: str, handler: CommandHandler
    ) -> None:  # pragma: no cover - M3 network wiring
        """Register a slash-command handler (M3 wiring)."""
        raise NotImplementedError(
            "Telegram command registration lands in milestone M3; use FakePlatform offline."
        )

    def scheduler(self) -> Scheduler:
        """Return the scheduler used for periodic rollups."""
        return self._scheduler
