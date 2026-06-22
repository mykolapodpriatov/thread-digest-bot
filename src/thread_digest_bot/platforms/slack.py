"""Slack adapter (optional extra ``[slack]``) — thin edge over the core.

This adapter maps the :class:`~thread_digest_bot.platforms.ChatPlatform` protocol onto
``slack-bolt``. Like the Telegram adapter it is deliberately thin: the digest /
grounding / store logic is platform-agnostic, so the adapter only translates fetch /
post / command calls and resolves permalinks via the pure builder in
:mod:`thread_digest_bot.links`.

The ``slack_bolt`` package is imported lazily so importing this module never fails when
the extra is absent; construction raises a clear :class:`ImportError`.

The conversation-history fetch / pagination wiring is part of milestone M3; the
:class:`~thread_digest_bot.platforms.fake.FakePlatform` proves the protocol offline in
the meantime.
"""

from __future__ import annotations

from thread_digest_bot.links import slack_archives_permalink
from thread_digest_bot.platforms import CommandHandler
from thread_digest_bot.schedule import IntervalScheduler, Scheduler
from thread_digest_bot.types import Thread

__all__ = ["SlackPlatform", "build_permalink"]


def build_permalink(
    workspace: str | None,
    channel_id: str | None,
    message_ts: str | None,
) -> str | None:
    """Return a Slack archives permalink, or ``None`` for an unknown shape.

    Delegates to the pure builder in :mod:`thread_digest_bot.links`; a real deployment
    would prefer the ``chat.getPermalink`` API result and fall back to this pattern.
    """
    return slack_archives_permalink(workspace, channel_id, message_ts)


class SlackPlatform:
    """A :class:`~thread_digest_bot.platforms.ChatPlatform` over slack-bolt.

    Args:
        bot_token: The Slack bot token (``xoxb-…``).
        scheduler: Scheduler for rollups (defaults to an in-process interval scheduler).
        app: An injected ``slack_bolt.App`` (for testing); when given, the SDK is not
            constructed.

    Raises:
        ImportError: If the ``slack`` extra is not installed and no ``app`` is injected.
    """

    name = "slack"

    def __init__(
        self,
        bot_token: str,
        *,
        scheduler: Scheduler | None = None,
        app: object | None = None,
    ) -> None:
        self.bot_token = bot_token
        self._scheduler: Scheduler = scheduler or IntervalScheduler()
        if app is not None:
            self._app = app
        else:
            try:
                from slack_bolt import App
            except ImportError as exc:  # pragma: no cover - exercised via the extra
                raise ImportError(
                    "The Slack adapter requires the 'slack' extra: "
                    "pip install 'thread-digest-bot[slack]'."
                ) from exc
            self._app = App(token=bot_token)

    def fetch_thread(
        self,
        channel_id: str,
        *,
        reply_to: str | None = None,
        last_n: int | None = None,
    ) -> Thread:  # pragma: no cover - M3 network wiring
        """Fetch a normalized thread (M3 wiring; see module docstring)."""
        raise NotImplementedError(
            "Slack conversation fetching lands in milestone M3; use FakePlatform offline."
        )

    def post_reply(
        self, channel_id: str, text: str, *, reply_to: str | None = None
    ) -> str:  # pragma: no cover - M3 network wiring
        """Post a reply to a channel (M3 wiring)."""
        raise NotImplementedError("Slack posting lands in milestone M3; use FakePlatform offline.")

    def register_command(
        self, command: str, handler: CommandHandler
    ) -> None:  # pragma: no cover - M3 network wiring
        """Register a slash-command handler (M3 wiring)."""
        raise NotImplementedError(
            "Slack command registration lands in milestone M3; use FakePlatform offline."
        )

    def scheduler(self) -> Scheduler:
        """Return the scheduler used for periodic rollups."""
        return self._scheduler
