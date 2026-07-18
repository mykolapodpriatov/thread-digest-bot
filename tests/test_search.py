"""Search tests — parse committed logs back into hits, filter, and JSON output.

Each test seeds a real, committed decision log through the append-only
:class:`DecisionStore` using the deterministic ``FakeLLM`` fixtures, then searches the
Markdown it produced — exercising the full write → read round-trip offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from thread_digest_bot.cli import EXIT_USAGE_ERROR, app
from thread_digest_bot.digest import digest
from thread_digest_bot.llm.fake import FakeLLM
from thread_digest_bot.search import search_logs
from thread_digest_bot.store import DecisionStore, StoreConfig
from thread_digest_bot.types import Message, Thread

runner = CliRunner()


def _seed(repo_root: Path, channel_id: str, messages: list[Message]) -> None:
    """Digest and commit a ``happy`` log for ``channel_id`` into ``repo_root``."""
    thread = Thread(channel_id=channel_id, platform="telegram", messages=messages)
    log = digest(thread, FakeLLM("happy"))
    DecisionStore(repo_root, config=StoreConfig(commit=True)).append(log)


def test_search_finds_decision_with_provenance(
    temp_git_repo: Path, sample_messages_fixture: list[Message]
) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)

    hits = search_logs(temp_git_repo, "onboarding")

    assert len(hits) == 1
    hit = hits[0]
    assert hit.kind == "decision"
    assert hit.channel == "team-eng"
    assert hit.digest_key  # a real, non-empty digest key
    assert "onboarding" in hit.line.casefold()
    # The permalink is carried from the first grounded citation (m1).
    assert hit.permalink == "https://t.me/c/1234567890/1"


def test_search_is_case_insensitive(
    temp_git_repo: Path, sample_messages_fixture: list[Message]
) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    assert search_logs(temp_git_repo, "ONBOARDING")
    assert search_logs(temp_git_repo, "Release Notes")


def test_search_matches_all_item_kinds(
    temp_git_repo: Path, sample_messages_fixture: list[Message]
) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    assert search_logs(temp_git_repo, "release notes")[0].kind == "action_item"
    assert search_logs(temp_git_repo, "feature flag")[0].kind == "open_question"


def test_search_miss_returns_empty(
    temp_git_repo: Path, sample_messages_fixture: list[Message]
) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    assert search_logs(temp_git_repo, "nothing-like-this-appears") == []


def test_search_channel_filter(temp_git_repo: Path, sample_messages_fixture: list[Message]) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    _seed(temp_git_repo, "team-ops", sample_messages_fixture)

    # Without a filter, both channels match.
    channels = {hit.channel for hit in search_logs(temp_git_repo, "onboarding")}
    assert channels == {"team-eng", "team-ops"}

    # With a filter, only the requested channel is returned.
    scoped = search_logs(temp_git_repo, "onboarding", channel="team-ops")
    assert scoped
    assert {hit.channel for hit in scoped} == {"team-ops"}


def test_search_kind_filter(temp_git_repo: Path, sample_messages_fixture: list[Message]) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    # "release notes" lives only in the action item, so filtering to decisions finds none.
    assert search_logs(temp_git_repo, "release notes", kind="decision") == []
    assert len(search_logs(temp_git_repo, "release notes", kind="action_item")) == 1


def test_search_missing_decisions_dir_is_clean_empty(tmp_path: Path) -> None:
    # A repo with no docs/decisions must yield an empty result, not a traceback.
    assert search_logs(tmp_path, "anything") == []


def test_search_cli_term_output(
    temp_git_repo: Path, sample_messages_fixture: list[Message]
) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    result = runner.invoke(app, ["search", "onboarding", "--repo-root", str(temp_git_repo)])
    assert result.exit_code == 0
    assert "[team-eng]" in result.stdout
    assert "onboarding" in result.stdout.lower()


def test_search_cli_no_matches_message(
    temp_git_repo: Path, sample_messages_fixture: list[Message]
) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    result = runner.invoke(app, ["search", "no-such-token", "--repo-root", str(temp_git_repo)])
    assert result.exit_code == 0
    assert "No matches." in result.stdout


def test_search_cli_json_parses(
    temp_git_repo: Path, sample_messages_fixture: list[Message]
) -> None:
    _seed(temp_git_repo, "team-eng", sample_messages_fixture)
    result = runner.invoke(
        app,
        ["search", "onboarding", "--repo-root", str(temp_git_repo), "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["channel"] == "team-eng"
    assert payload[0]["kind"] == "decision"
    assert set(payload[0]) == {"channel", "digest_key", "kind", "line", "permalink"}


def test_search_cli_invalid_format_exits(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", "x", "--repo-root", str(tmp_path), "--format", "xml"])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "unknown --format" in result.output
