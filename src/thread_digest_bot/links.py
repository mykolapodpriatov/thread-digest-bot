"""Permalink builders — pure functions, never a plausible-but-wrong link.

Each builder returns ``None`` for any shape it does not understand, rather than
guessing. A wrong deep link in an audit trail is worse than no link.

Telegram private supergroups/channels use ``https://t.me/c/<internal>/<msg>`` where
``internal`` is the ``chat_id`` with its ``-100`` prefix stripped (``-1001234567890``
-> ``1234567890``). Public channels use ``https://t.me/<username>/<msg>``.

Telegram forum-topic threads (which need ``?thread=<topic_id>``) are intentionally
not modelled in v1 — see the plan's deferred note.
"""

from __future__ import annotations

import re

_TELEGRAM_SUPERGROUP_PREFIX = "-100"

#: Telegram public usernames: 5-32 chars, letters/digits/underscore (the ``@`` is
#: stripped before matching). See Telegram's username rules.
_TELEGRAM_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{5,32}$")

#: Slack workspace subdomain (``acme`` in ``acme.slack.com``): lowercase
#: letters/digits/hyphens, not starting or ending with a hyphen.
_SLACK_WORKSPACE_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

#: Slack channel/conversation id, e.g. ``C0123ABC`` / ``G…`` / ``D…``: an uppercase
#: letter followed by uppercase letters and digits.
_SLACK_CHANNEL_RE = re.compile(r"^[A-Z][A-Z0-9]+$")

__all__ = [
    "discord_permalink",
    "slack_archives_permalink",
    "telegram_private_permalink",
    "telegram_public_permalink",
]


def telegram_private_permalink(chat_id: int | str, message_id: int | str) -> str | None:
    """Build a private Telegram channel/supergroup permalink.

    Args:
        chat_id: The Telegram chat id, e.g. ``-1001234567890``. Must carry the
            ``-100`` supergroup prefix.
        message_id: The message id within the channel.

    Returns:
        ``https://t.me/c/<internal>/<message_id>`` where ``internal`` drops the
        leading ``-100``, or ``None`` if ``chat_id`` is not a ``-100`` supergroup id
        or ``message_id`` is not a positive integer.

    Examples:
        >>> telegram_private_permalink(-1001234567890, 42)
        'https://t.me/c/1234567890/42'
        >>> telegram_private_permalink(-42, 42) is None
        True
    """
    chat_str = str(chat_id).strip()
    if not chat_str.startswith(_TELEGRAM_SUPERGROUP_PREFIX):
        return None
    internal = chat_str[len(_TELEGRAM_SUPERGROUP_PREFIX) :]
    if not internal.isdigit() or internal == "":
        return None
    msg = _coerce_positive_int(message_id)
    if msg is None:
        return None
    return f"https://t.me/c/{internal}/{msg}"


def telegram_public_permalink(username: str | None, message_id: int | str) -> str | None:
    """Build a public Telegram channel permalink.

    Args:
        username: The public channel username (with or without a leading ``@``).
        message_id: The message id within the channel.

    Returns:
        ``https://t.me/<username>/<message_id>``, or ``None`` if the username is
        missing/blank, is not a well-formed Telegram username (5-32 chars,
        ``[A-Za-z0-9_]``), or the message id is not a positive integer.

    Examples:
        >>> telegram_public_permalink(" @acmehq ", 7)
        'https://t.me/acmehq/7'
        >>> telegram_public_permalink("@bad name", 7) is None
        True
        >>> telegram_public_permalink("@a", 7) is None
        True
    """
    if not username:
        return None
    # Trim surrounding whitespace FIRST, then drop a single leading ``@``; doing it in
    # the other order would let " @acme " leak a stray space/``@`` into the URL.
    handle = username.strip()
    if handle.startswith("@"):
        handle = handle[1:]
    if not _TELEGRAM_USERNAME_RE.match(handle):
        return None
    msg = _coerce_positive_int(message_id)
    if msg is None:
        return None
    return f"https://t.me/{handle}/{msg}"


def slack_archives_permalink(
    workspace: str | None,
    channel_id: str | None,
    message_ts: str | None,
) -> str | None:
    """Build a Slack archives permalink from a workspace, channel, and message ts.

    This mirrors the ``https://<workspace>.slack.com/archives/<channel>/p<ts>``
    pattern that ``chat.getPermalink`` returns; the ``ts`` dots are removed and the
    value is prefixed with ``p`` (``1700000000.000200`` -> ``p1700000000000200``).

    Args:
        workspace: The workspace subdomain (e.g. ``acme`` for ``acme.slack.com``).
        channel_id: The channel id (e.g. ``C0123ABC``).
        message_ts: The message timestamp string (e.g. ``1700000000.000200``).

    Returns:
        The archives URL, or ``None`` if any part is missing/malformed or
        ``message_ts`` is not a well-formed ``<seconds>.<fraction>`` timestamp (exactly
        one ``.`` with all-digit parts).

    Examples:
        >>> slack_archives_permalink("acme", "C0123ABC", "1700000000.000200")
        'https://acme.slack.com/archives/C0123ABC/p1700000000000200'
        >>> slack_archives_permalink(" acme ", "C0123ABC", "1700000000.000200")
        'https://acme.slack.com/archives/C0123ABC/p1700000000000200'
        >>> slack_archives_permalink("acme", "C0123ABC", "1700.0.2") is None
        True
    """
    if not workspace or not channel_id or not message_ts:
        return None
    ws = workspace.strip()
    channel = channel_id.strip()
    ts = message_ts.strip()
    if not _SLACK_WORKSPACE_RE.match(ws) or not _SLACK_CHANNEL_RE.match(channel):
        return None
    # A well-formed Slack ts is exactly ``<seconds>.<fraction>`` with all-digit parts;
    # reject anything with the wrong dot count or non-digit/empty parts.
    parts = ts.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    digits = parts[0] + parts[1]
    return f"https://{ws}.slack.com/archives/{channel}/p{digits}"


def discord_permalink(
    guild_id: int | str | None,
    channel_id: int | str,
    message_id: int | str,
) -> str | None:
    """Build a Discord message permalink from snowflake ids.

    Discord jump links use ``https://discord.com/channels/<guild>/<channel>/<message>``
    where each segment is a snowflake id. Direct-message channels have no guild, so the
    literal ``@me`` stands in for the guild segment when ``guild_id`` is ``None``.

    Args:
        guild_id: The guild (server) snowflake id, or ``None`` for a direct message
            (which renders the ``@me`` guild segment).
        channel_id: The channel snowflake id.
        message_id: The message snowflake id.

    Returns:
        The Discord permalink, or ``None`` if ``channel_id`` or ``message_id`` is not a
        positive integer, or a supplied ``guild_id`` is not a positive integer. Following
        the module contract, an unknown shape yields ``None`` rather than a guessed link.

    Examples:
        >>> discord_permalink(112233445566778899, 998877665544332211, 123456789012345678)
        'https://discord.com/channels/112233445566778899/998877665544332211/123456789012345678'
        >>> discord_permalink(None, 998877665544332211, 123456789012345678)
        'https://discord.com/channels/@me/998877665544332211/123456789012345678'
        >>> discord_permalink("  42  ", " 7 ", " 9 ")
        'https://discord.com/channels/42/7/9'
        >>> discord_permalink(-1, 7, 9) is None
        True
        >>> discord_permalink(42, "not-a-number", 9) is None
        True
    """
    channel = _coerce_positive_int(channel_id)
    message = _coerce_positive_int(message_id)
    if channel is None or message is None:
        return None
    if guild_id is None:
        guild = "@me"
    else:
        guild_num = _coerce_positive_int(guild_id)
        if guild_num is None:
            return None
        guild = str(guild_num)
    return f"https://discord.com/channels/{guild}/{channel}/{message}"


def _coerce_positive_int(value: int | str) -> int | None:
    """Return ``value`` as a positive int, or ``None`` if it is not one."""
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None
