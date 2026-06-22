#!/usr/bin/env python3
"""Offline, zero-network demo: digest a thread into a committed, attributed log.

Run it directly::

    python examples/demo.py

It creates a throwaway Git repository in a temporary directory, digests
``examples/thread.json`` with the deterministic ``FakeLLM`` (no API key, no network),
appends the rendered entry to ``docs/decisions/team-eng.md``, commits it, and prints the
resulting Git log and Markdown so you can see the audit trail the bot maintains.

This mirrors what ``thread-digest-bot digest-file examples/thread.json --commit`` does,
but in-process so it is easy to read and adapt.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from git import Repo

from thread_digest_bot import (
    DecisionStore,
    FakeLLM,
    StoreConfig,
    digest,
    thread_from_json,
)

HERE = Path(__file__).resolve().parent
THREAD_JSON = HERE / "thread.json"


def main() -> None:
    """Digest the example thread into a fresh temp repo and print the result."""
    thread = thread_from_json(THREAD_JSON.read_text(encoding="utf-8"))
    log = digest(thread, FakeLLM("happy"))

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        repo = Repo.init(repo_root)
        # Configure identity so the commit succeeds on a bare runner.
        with repo.config_writer() as cw:
            cw.set_value("user", "name", "Demo Bot")
            cw.set_value("user", "email", "demo@example.com")

        store = DecisionStore(repo_root, config=StoreConfig(commit=True))
        result = store.append(log)

        print(f"committed: {result.committed}")
        print(f"commit message: {result.commit_message}")
        print(f"log file: {result.path.relative_to(repo_root)}")
        print()
        print("=== git log ===")
        for commit in repo.iter_commits():
            print(f"{commit.hexsha[:10]}  {commit.message.strip()}")
        print()
        print("=== docs/decisions/team-eng.md ===")
        print(result.path.read_text(encoding="utf-8"), end="")

        # A re-append of the same digest is an idempotent no-op (no second commit).
        again = store.append(log)
        print(
            f"re-append committed again: {again.committed} "
            f"(skipped_duplicate={again.skipped_duplicate})"
        )


if __name__ == "__main__":
    main()
