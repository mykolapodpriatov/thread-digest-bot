"""Shared pytest fixtures — all offline, deterministic, zero-network.

Provides a temporary Git repository (with committer identity configured so commits
succeed on bare CI runners), the in-memory :class:`FakePlatform`, the deterministic
:class:`FakeLLM`, and small thread builders used across the suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from git import Repo

from thread_digest_bot.llm.fake import FakeLLM
from thread_digest_bot.platforms.fake import FakePlatform
from thread_digest_bot.types import Author, Message, Thread


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Iterator[Path]:
    """Initialize a Git repo in ``tmp_path`` with a committer identity.

    Sets ``user.name``/``user.email`` via GitPython's ``config_writer()`` so commits
    work even where no global Git identity is configured (e.g. CI). Yields the repo
    root path.
    """
    repo = Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test Bot")
        cw.set_value("user", "email", "test-bot@example.com")
        # Keep commits cheap and deterministic.
        cw.set_value("commit", "gpgsign", "false")
    try:
        yield tmp_path
    finally:
        repo.close()


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A deterministic happy-path :class:`FakeLLM`."""
    return FakeLLM("happy")


@pytest.fixture
def fake_platform() -> FakePlatform:
    """A fresh in-memory :class:`FakePlatform`."""
    return FakePlatform()


def make_author(uid: str, display: str) -> Author:
    """Build an :class:`Author` (test helper)."""
    return Author(id=uid, display=display)


def make_message(
    mid: str,
    author: Author,
    text: str,
    *,
    ts_label: str = "t0",
    permalink: str | None = None,
) -> Message:
    """Build a :class:`Message` (test helper)."""
    return Message(id=mid, author=author, text=text, ts_label=ts_label, permalink=permalink)


def sample_messages() -> list[Message]:
    """Return the canonical three-message thread the ``happy`` fixture cites.

    The message texts contain the exact substrings the ``FakeLLM('happy')`` fixture
    quotes, so grounding keeps every citation and quote.
    """
    ada = make_author("u_ada", "Ada")
    bob = make_author("u_bob", "Bob")
    cleo = make_author("u_cleo", "Cleo")
    return [
        make_message(
            "m1",
            ada,
            "Let's ship the onboarding flow on Friday once QA signs off.",
            ts_label="Mon 09:14",
            permalink="https://t.me/c/1234567890/1",
        ),
        make_message(
            "m2",
            bob,
            "Sounds good. I'll write the release notes before then.",
            ts_label="Mon 09:16",
            permalink="https://t.me/c/1234567890/2",
        ),
        make_message(
            "m3",
            cleo,
            "The rollback plan is ready. One open thing: should we gate it behind a flag?",
            ts_label="Mon 09:21",
            permalink="https://t.me/c/1234567890/3",
        ),
    ]


@pytest.fixture
def sample_messages_fixture() -> list[Message]:
    """The canonical three-message list matching the ``happy`` fixture."""
    return sample_messages()


@pytest.fixture
def sample_thread() -> Thread:
    """A canonical three-message :class:`Thread` matching the ``happy`` fixture."""
    return Thread(channel_id="team-eng", platform="telegram", messages=sample_messages())
