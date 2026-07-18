"""Link-builder tests — exact shapes, and ``None`` for anything ambiguous."""

from __future__ import annotations

import pytest

from thread_digest_bot.links import (
    discord_permalink,
    slack_archives_permalink,
    telegram_private_permalink,
    telegram_public_permalink,
)


def test_telegram_private_literal_from_spec() -> None:
    # The canonical example from the plan: drop the -100 supergroup prefix.
    assert telegram_private_permalink(-1001234567890, 42) == "https://t.me/c/1234567890/42"


def test_telegram_private_accepts_string_inputs() -> None:
    assert telegram_private_permalink("-1001234567890", "42") == "https://t.me/c/1234567890/42"


@pytest.mark.parametrize(
    "chat_id",
    [-42, 1234567890, "not-a-number", "-100", "-100abc"],
)
def test_telegram_private_unsupported_shapes_return_none(chat_id: object) -> None:
    assert telegram_private_permalink(chat_id, 42) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("message_id", [0, -1, "x"])
def test_telegram_private_bad_message_id_returns_none(message_id: object) -> None:
    assert telegram_private_permalink(-1001234567890, message_id) is None  # type: ignore[arg-type]


def test_telegram_public_with_and_without_at() -> None:
    assert telegram_public_permalink("@acmehq", 7) == "https://t.me/acmehq/7"
    assert telegram_public_permalink("acmehq", 7) == "https://t.me/acmehq/7"


def test_telegram_public_trims_whitespace_before_stripping_at() -> None:
    # Regression: trim must happen BEFORE the ``@`` strip, otherwise " @acmehq "
    # leaks a stray ``@``/space into the URL (e.g. ``t.me/@acmehq``).
    assert telegram_public_permalink(" @acmehq ", 7) == "https://t.me/acmehq/7"
    assert telegram_public_permalink("  acmehq  ", 7) == "https://t.me/acmehq/7"


@pytest.mark.parametrize(
    "username",
    [
        None,
        "",
        "   ",
        "@",
        "@a",  # too short (< 5 chars)
        "abcd",  # too short (< 5 chars)
        "@bad name",  # contains a space -> not a valid handle
        "bad-handle",  # hyphen is not allowed in Telegram usernames
        "a" * 33,  # too long (> 32 chars)
    ],
)
def test_telegram_public_invalid_username_returns_none(username: str | None) -> None:
    assert telegram_public_permalink(username, 7) is None


def test_telegram_public_bad_message_id_returns_none() -> None:
    assert telegram_public_permalink("acmehq", 0) is None


def test_discord_permalink_shape() -> None:
    url = discord_permalink(112233445566778899, 998877665544332211, 123456789012345678)
    assert url == (
        "https://discord.com/channels/112233445566778899/998877665544332211/123456789012345678"
    )


def test_discord_permalink_accepts_string_inputs() -> None:
    assert discord_permalink("42", "7", "9") == "https://discord.com/channels/42/7/9"


def test_discord_permalink_dm_form_when_guild_is_none() -> None:
    # A DM channel has no guild, so the ``@me`` sentinel stands in for the guild segment.
    assert discord_permalink(None, 7, 9) == "https://discord.com/channels/@me/7/9"


def test_discord_permalink_trims_whitespace() -> None:
    assert discord_permalink("  42  ", " 7 ", " 9 ") == "https://discord.com/channels/42/7/9"


@pytest.mark.parametrize(
    ("guild", "channel", "message"),
    [
        (-1, 7, 9),  # negative guild id
        (0, 7, 9),  # zero guild id
        ("not-a-number", 7, 9),  # non-numeric guild id
        (42, -7, 9),  # negative channel id
        (42, 0, 9),  # zero channel id
        (42, "x", 9),  # non-numeric channel id
        (42, 7, -9),  # negative message id
        (42, 7, 0),  # zero message id
        (42, 7, "y"),  # non-numeric message id
        (None, 0, 9),  # DM form still validates channel id
        (None, 7, -1),  # DM form still validates message id
    ],
)
def test_discord_permalink_unsupported_shapes_return_none(
    guild: object, channel: object, message: object
) -> None:
    assert discord_permalink(guild, channel, message) is None  # type: ignore[arg-type]


def test_slack_archives_permalink_shape() -> None:
    url = slack_archives_permalink("acme", "C0123ABC", "1700000000.000200")
    assert url == "https://acme.slack.com/archives/C0123ABC/p1700000000000200"


def test_slack_archives_permalink_trims_padded_inputs() -> None:
    # Regression: padded workspace/channel/ts must be trimmed, not embedded verbatim
    # (which would yield ``https:// acme .slack.com/archives/ C0123ABC /p…``).
    url = slack_archives_permalink("  acme  ", "  C0123ABC  ", "  1700000000.000200  ")
    assert url == "https://acme.slack.com/archives/C0123ABC/p1700000000000200"


@pytest.mark.parametrize(
    ("workspace", "channel", "ts"),
    [
        (None, "C0123ABC", "1700000000.000200"),
        ("acme", None, "1700000000.000200"),
        ("acme", "C0123ABC", None),
        ("acme", "C0123ABC", "1700000000"),  # no dot -> not a valid ts
        ("acme", "C0123ABC", "17000.00ab"),  # non-digit -> rejected
        ("acme", "C0123ABC", "1700.000.200"),  # two dots -> malformed ts
        ("acme", "C0123ABC", "1700000000."),  # empty fraction -> rejected
        ("acme", "C0123ABC", ".000200"),  # empty seconds -> rejected
        ("acme", "C0123ABC", "."),  # both parts empty -> rejected
        ("acme", "bad channel", "1700000000.000200"),  # space in channel id -> rejected
        ("acme", "c0123abc", "1700000000.000200"),  # lowercase channel id -> rejected
        ("ACME", "C0123ABC", "1700000000.000200"),  # uppercase workspace -> rejected
        ("acme!", "C0123ABC", "1700000000.000200"),  # bad workspace char -> rejected
    ],
)
def test_slack_archives_unsupported_shapes_return_none(
    workspace: str | None, channel: str | None, ts: str | None
) -> None:
    assert slack_archives_permalink(workspace, channel, ts) is None
