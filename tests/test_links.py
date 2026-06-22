"""Link-builder tests — exact shapes, and ``None`` for anything ambiguous."""

from __future__ import annotations

import pytest

from thread_digest_bot.links import (
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
