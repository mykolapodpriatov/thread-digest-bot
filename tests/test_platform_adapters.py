"""Telegram/Slack adapter tests — offline parts only (permalinks + import guards).

The live fetch/post wiring lands in milestone M3; here we cover the pure permalink
helpers, the scheduler accessor, and the clear ImportError raised when an optional extra
is absent (verified via injected stand-ins so no real SDK is required).
"""

from __future__ import annotations

import pytest

from thread_digest_bot.platforms import slack as slack_mod
from thread_digest_bot.platforms import telegram as telegram_mod
from thread_digest_bot.schedule import FakeScheduler

# -- Telegram ----------------------------------------------------------------


def test_telegram_build_permalink_prefers_public() -> None:
    assert telegram_mod.build_permalink(-1001234567890, 42, username="acmehq") == (
        "https://t.me/acmehq/42"
    )


def test_telegram_build_permalink_falls_back_to_private() -> None:
    assert telegram_mod.build_permalink(-1001234567890, 42) == "https://t.me/c/1234567890/42"


def test_telegram_build_permalink_unknown_shape_is_none() -> None:
    assert telegram_mod.build_permalink(-42, 42) is None


def test_telegram_platform_with_injected_application() -> None:
    sched = FakeScheduler()
    platform = telegram_mod.TelegramPlatform("token", scheduler=sched, application=object())
    assert platform.name == "telegram"
    assert platform.scheduler() is sched


def test_telegram_platform_missing_extra_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the 'telegram' extra being absent.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "telegram.ext" or name.startswith("telegram"):
            raise ImportError("no telegram")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="telegram"):
        telegram_mod.TelegramPlatform("token")


# -- Slack -------------------------------------------------------------------


def test_slack_build_permalink_shape() -> None:
    assert slack_mod.build_permalink("acme", "C0123ABC", "1700000000.000200") == (
        "https://acme.slack.com/archives/C0123ABC/p1700000000000200"
    )


def test_slack_build_permalink_unknown_shape_is_none() -> None:
    assert slack_mod.build_permalink(None, "C0123ABC", "1700000000.000200") is None


def test_slack_platform_with_injected_app() -> None:
    sched = FakeScheduler()
    platform = slack_mod.SlackPlatform("xoxb-token", scheduler=sched, app=object())
    assert platform.name == "slack"
    assert platform.scheduler() is sched


def test_slack_platform_missing_extra_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "slack_bolt" or name.startswith("slack_bolt"):
            raise ImportError("no slack_bolt")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="slack"):
        slack_mod.SlackPlatform("xoxb-token")
