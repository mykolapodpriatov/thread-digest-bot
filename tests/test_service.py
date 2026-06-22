"""Service-level tests for argument resolution and the LLM-error path."""

from __future__ import annotations

from pathlib import Path

from thread_digest_bot.llm.fake import FakeLLM
from thread_digest_bot.platforms.fake import FakePlatform
from thread_digest_bot.service import DigestService
from thread_digest_bot.store import DecisionStore
from thread_digest_bot.types import Author, Message


def _service(temp_git_repo: Path, *, fixture: str = "happy") -> tuple[DigestService, FakePlatform]:
    platform = FakePlatform()
    ada = Author(id="u_ada", display="Ada")
    platform.add_channel(
        "c1",
        [Message(id="m1", author=ada, text="Let's ship the onboarding flow", ts_label="t")],
    )
    store = DecisionStore(temp_git_repo)
    service = DigestService(platform, FakeLLM(fixture), store, default_last_n=7)
    return service, platform


def test_resolve_last_n_uses_first_digit_token(temp_git_repo: Path) -> None:
    service, _ = _service(temp_git_repo)
    # A non-digit token is skipped; the digit token wins.
    assert service._resolve_last_n(["all", "5"]) == 5


def test_resolve_last_n_defaults_without_digit(temp_git_repo: Path) -> None:
    service, _ = _service(temp_git_repo)
    assert service._resolve_last_n(["nope"]) == 7  # falls back to default_last_n
    assert service._resolve_last_n(None) == 7
    assert service._resolve_last_n([]) == 7


def test_llm_error_surfaces_friendly_message(temp_git_repo: Path) -> None:
    service, platform = _service(temp_git_repo, fixture="schema_mismatch_persistent")
    service.register()
    platform.invoke_command("digest", "c1")

    outcome = service.history[-1]
    assert outcome.log is None
    assert outcome.error_message is not None
    assert "unusable response" in outcome.error_message
    # The friendly error was posted, and nothing was committed.
    assert any("unusable response" in p.text for p in platform.posted)


def test_successful_digest_records_reply_id(temp_git_repo: Path) -> None:
    service, _ = _service(temp_git_repo)
    outcome = service.handle_digest("c1", args=["1"])
    assert outcome.reply_message_id is not None
    assert outcome.committed is True
