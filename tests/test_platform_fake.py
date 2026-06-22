"""End-to-end platform tests via FakePlatform + FakeLLM + a temp Git repo.

These exercise the full ``/digest`` pipeline (fetch -> digest -> ground -> post ->
store) offline, including the drop path, fetch edge cases, and scheduled rollups.
"""

from __future__ import annotations

from pathlib import Path

from git import Repo

from thread_digest_bot.llm.fake import FakeLLM
from thread_digest_bot.platforms.fake import FakePlatform
from thread_digest_bot.schedule import FakeScheduler
from thread_digest_bot.service import DigestService
from thread_digest_bot.store import DecisionStore
from thread_digest_bot.types import Author, Message


def sample_messages() -> list[Message]:
    """Three messages whose text contains the substrings the ``happy`` fixture quotes."""
    ada = Author(id="u_ada", display="Ada")
    bob = Author(id="u_bob", display="Bob")
    cleo = Author(id="u_cleo", display="Cleo")
    return [
        Message(
            id="m1",
            author=ada,
            text="Let's ship the onboarding flow on Friday once QA signs off.",
            ts_label="Mon 09:14",
            permalink="https://t.me/c/1234567890/1",
        ),
        Message(
            id="m2",
            author=bob,
            text="Sounds good. I'll write the release notes before then.",
            ts_label="Mon 09:16",
            permalink="https://t.me/c/1234567890/2",
        ),
        Message(
            id="m3",
            author=cleo,
            text="The rollback plan is ready. One open thing: should we gate it behind a flag?",
            ts_label="Mon 09:21",
            permalink="https://t.me/c/1234567890/3",
        ),
    ]


def _commit_count(repo_root: Path) -> int:
    repo = Repo(repo_root)
    if not repo.head.is_valid():
        return 0
    return sum(1 for _ in repo.iter_commits())


def _service(
    temp_git_repo: Path,
    *,
    fixture: str = "happy",
    scheduler: FakeScheduler | None = None,
) -> tuple[DigestService, FakePlatform]:
    platform = FakePlatform(scheduler=scheduler)
    platform.add_channel("team-eng", sample_messages())
    store = DecisionStore(temp_git_repo)
    service = DigestService(platform, FakeLLM(fixture), store, default_last_n=50)
    service.register()
    return service, platform


def test_digest_command_posts_reply_and_commits(temp_git_repo: Path) -> None:
    service, platform = _service(temp_git_repo)
    platform.invoke_command("digest", "team-eng", args=["3"])

    # A reply was posted to the channel.
    assert len(platform.posted) == 1
    assert "Digest" in platform.posted[0].text

    # The log was committed to the store.
    assert _commit_count(temp_git_repo) == 1
    md = (temp_git_repo / "docs/decisions/team-eng.md").read_text(encoding="utf-8")
    assert "Ship the new onboarding flow on Friday." in md

    outcome = service.history[-1]
    assert outcome.committed is True
    assert outcome.log is not None


def test_invalid_ids_pipeline_drops_item_end_to_end(temp_git_repo: Path) -> None:
    service, platform = _service(temp_git_repo, fixture="invalid_ids")
    # The raw fixture proposes 2 decisions; the unsourced one must be dropped.
    raw = FakeLLM("invalid_ids")._payload()
    raw_decision_count = len(raw.decisions)
    assert raw_decision_count == 2

    platform.invoke_command("digest", "team-eng")

    outcome = service.history[-1]
    assert outcome.log is not None
    committed_decisions = len(outcome.log.decisions)
    assert committed_decisions < raw_decision_count  # the drop path ran in the full pipeline
    assert committed_decisions == 1

    posted_reply = platform.posted[0].text
    assert "Adopt a four-day work week." not in posted_reply  # dropped item not posted
    md = (temp_git_repo / "docs/decisions/team-eng.md").read_text(encoding="utf-8")
    assert "four-day work week" not in md


def test_missing_reply_to_surfaces_friendly_message(temp_git_repo: Path) -> None:
    service, platform = _service(temp_git_repo)
    platform.invoke_command("digest", "team-eng", reply_to="does-not-exist")

    outcome = service.history[-1]
    assert outcome.log is None
    assert outcome.error_message is not None
    assert "couldn't find that thread" in outcome.error_message
    # The friendly message is posted to the channel, not an exception.
    assert len(platform.posted) == 1
    assert "couldn't find that thread" in platform.posted[0].text
    # Nothing committed for a failed fetch.
    assert _commit_count(temp_git_repo) == 0


def test_fetch_api_failure_surfaces_friendly_message(temp_git_repo: Path) -> None:
    service, platform = _service(temp_git_repo)
    platform.set_fetch_failure("team-eng")
    platform.invoke_command("digest", "team-eng")

    outcome = service.history[-1]
    assert outcome.error_message is not None
    assert "permissions or API error" in outcome.error_message


def test_history_shorter_than_last_n_digests_what_exists(temp_git_repo: Path) -> None:
    service, _ = _service(temp_git_repo)
    # Channel has 3 messages; ask for 200.
    outcome = service.handle_digest("team-eng", args=["200"])
    assert outcome.log is not None
    # All three participants present -> it digested what exists, not an error.
    assert {p.display for p in outcome.log.participants} == {"Ada", "Bob", "Cleo"}


def test_truncated_history_is_marked(temp_git_repo: Path) -> None:
    platform = FakePlatform()
    ada = Author(id="u_ada", display="Ada")
    many = [
        Message(id=f"m{i}", author=ada, text=f"message {i}", ts_label=f"t{i}") for i in range(10)
    ]
    platform.add_channel("big", many, max_messages=5)
    store = DecisionStore(temp_git_repo)
    service = DigestService(platform, FakeLLM("empty"), store)

    outcome = service.handle_digest("big", args=["10"])
    assert outcome.log is not None
    assert outcome.log.truncated is True
    md = (temp_git_repo / "docs/decisions/big.md").read_text(encoding="utf-8")
    assert "truncated" in md.lower()


def test_reply_to_digests_thread_from_reply_onward(temp_git_repo: Path) -> None:
    service, _ = _service(temp_git_repo)
    outcome = service.handle_digest("team-eng", reply_to="m2")
    assert outcome.log is not None
    # Thread is m2..m3, so Ada (m1 only) is not a participant.
    displays = {p.display for p in outcome.log.participants}
    assert displays == {"Bob", "Cleo"}


def test_scheduled_rollup_fires_and_dedups_on_refire(temp_git_repo: Path) -> None:
    scheduler = FakeScheduler()
    service, _ = _service(temp_git_repo, scheduler=scheduler)

    service.schedule_rollup(
        "team-eng",
        period="weekly",
        period_key_provider=lambda: "2026-W25",
        interval_seconds=3600,
        last_n=50,
    )
    assert len(scheduler.registrations) == 1

    scheduler.fire_all()  # first fire commits a rollup
    assert _commit_count(temp_git_repo) == 1
    first = service.history[-1]
    assert first.committed is True
    assert first.log is not None
    assert first.log.range_label == "weekly 2026-W25"

    scheduler.fire_all()  # re-fire over the same message set -> idempotent no-op
    assert _commit_count(temp_git_repo) == 1  # no second commit
    second = service.history[-1]
    assert second.committed is False
    assert second.skipped_duplicate is True


def test_run_rollup_once_commits(temp_git_repo: Path) -> None:
    service, _ = _service(temp_git_repo)
    outcome = service.run_rollup_once(
        "team-eng", period="daily", period_key="2026-06-22", last_n=50
    )
    assert outcome.committed is True
    assert outcome.log is not None
    assert outcome.log.range_label == "daily 2026-06-22"


def test_rollups_in_different_periods_over_same_messages_both_commit(temp_git_repo: Path) -> None:
    # Regression: a periodic rollup is keyed by (channel, period), NOT by the message
    # set, so a NEW period over the SAME recent messages must not be deduped against a
    # prior digest. Both rollups commit; re-firing the same period is a no-op.
    service, _ = _service(temp_git_repo)

    first = service.run_rollup_once("team-eng", period="weekly", period_key="2026-W25", last_n=50)
    second = service.run_rollup_once("team-eng", period="weekly", period_key="2026-W26", last_n=50)

    assert first.committed is True
    assert second.committed is True  # same messages, different period -> NOT a duplicate
    assert first.log is not None and second.log is not None
    assert first.log.digest_key != second.log.digest_key  # period-scoped keys differ
    assert _commit_count(temp_git_repo) == 2

    # Re-firing the same period is deduped (no third commit).
    refire = service.run_rollup_once("team-eng", period="weekly", period_key="2026-W26", last_n=50)
    assert refire.committed is False
    assert refire.skipped_duplicate is True
    assert _commit_count(temp_git_repo) == 2


def test_successful_digest_command_posts_exactly_one_reply(temp_git_repo: Path) -> None:
    # Regression (dead/double-post guard): the success reply is posted once inside
    # handle_digest; _on_digest must not post a second copy for a successful run.
    service, platform = _service(temp_git_repo)
    platform.invoke_command("digest", "team-eng", args=["3"])

    outcome = service.history[-1]
    assert outcome.error_message is None
    assert outcome.log is not None
    assert len(platform.posted) == 1  # exactly one reply, never doubled
    assert "Digest" in platform.posted[0].text
