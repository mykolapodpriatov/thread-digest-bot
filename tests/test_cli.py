"""CLI tests — digest-file happy path, --commit, and invalid-input exit codes."""

from __future__ import annotations

import json
from pathlib import Path

from git import Repo
from typer.testing import CliRunner

from thread_digest_bot.cli import EXIT_USAGE_ERROR, app

runner = CliRunner()

VALID_THREAD = {
    "channel_id": "team-eng",
    "platform": "telegram",
    "messages": [
        {
            "id": "m1",
            "author": {"id": "u_ada", "display": "Ada"},
            "text": "Let's ship the onboarding flow on Friday.",
            "ts_label": "Mon 09:14",
            "permalink": "https://t.me/c/1234567890/1",
        },
        {
            "id": "m2",
            "author": {"id": "u_bob", "display": "Bob"},
            "text": "I'll write the release notes.",
            "ts_label": "Mon 09:16",
            "permalink": None,
        },
        {
            "id": "m3",
            "author": {"id": "u_cleo", "display": "Cleo"},
            "text": "rollback plan is ready; should we gate it behind a flag",
            "ts_label": "Mon 09:21",
            "permalink": None,
        },
    ],
}


def _write_thread(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "thread.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_digest_file_happy_path_stdout(tmp_path: Path) -> None:
    thread_path = _write_thread(tmp_path, VALID_THREAD)
    result = runner.invoke(app, ["digest-file", str(thread_path)])
    assert result.exit_code == 0
    assert "Ship the new onboarding flow on Friday." in result.stdout
    assert "Digest key:" in result.stdout


def test_digest_file_writes_out_file(tmp_path: Path) -> None:
    thread_path = _write_thread(tmp_path, VALID_THREAD)
    out_path = tmp_path / "sub" / "log.md"
    result = runner.invoke(app, ["digest-file", str(thread_path), "--out", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("## ")


def test_digest_file_commit_into_repo(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "CLI Test")
        cw.set_value("user", "email", "cli@example.com")

    thread_path = _write_thread(tmp_path, VALID_THREAD)
    result = runner.invoke(
        app,
        ["digest-file", str(thread_path), "--commit", "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "Committed digest" in result.stdout
    log_file = tmp_path / "docs/decisions/team-eng.md"
    assert log_file.exists()
    assert sum(1 for _ in repo.iter_commits()) == 1


def test_digest_file_missing_file_nonzero_exit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["digest-file", str(tmp_path / "nope.json")])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "not found" in result.output


def test_digest_file_invalid_missing_field(tmp_path: Path) -> None:
    bad = {"channel_id": "c", "platform": "telegram", "messages": [{"id": "m1"}]}
    thread_path = _write_thread(tmp_path, bad)
    result = runner.invoke(app, ["digest-file", str(thread_path)])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "not a valid thread file" in result.output
    # The error names the offending field path.
    assert "author" in result.output or "text" in result.output


def test_digest_file_invalid_unknown_field(tmp_path: Path) -> None:
    bad = dict(VALID_THREAD)
    bad["surprise"] = "unexpected"  # extra="forbid" rejects unknown shapes
    thread_path = _write_thread(tmp_path, bad)
    result = runner.invoke(app, ["digest-file", str(thread_path)])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "not a valid thread file" in result.output


def test_digest_file_invalid_platform(tmp_path: Path) -> None:
    bad = dict(VALID_THREAD)
    bad["platform"] = "discord"  # not in the Literal
    thread_path = _write_thread(tmp_path, bad)
    result = runner.invoke(app, ["digest-file", str(thread_path)])
    assert result.exit_code == EXIT_USAGE_ERROR


def test_digest_file_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "thread.json"
    path.write_text("{ this is not json", encoding="utf-8")
    result = runner.invoke(app, ["digest-file", str(path)])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "not a valid thread file" in result.output or "not valid JSON" in result.output


def test_rollup_command_prints_label() -> None:
    result = runner.invoke(
        app,
        ["rollup", "--channel", "team-eng", "--period", "weekly", "--period-key", "2026-W25"],
    )
    assert result.exit_code == 0
    assert "weekly 2026-W25" in result.stdout


def test_run_requires_platforms(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[llm]\nprovider = 'fake'\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == EXIT_USAGE_ERROR
    assert "no platforms configured" in result.output


def test_run_reports_configured_platforms(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[[platforms]]\nname = 'telegram'\ntoken_env = 'TG_TOKEN'\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 0
    assert "telegram" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help -> exit code 0 with usage.
    assert "Usage" in result.output
