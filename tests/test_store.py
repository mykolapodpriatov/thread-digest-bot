"""Store tests — append-only integrity, idempotency, orphan recovery, webhook export."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from git import Repo

from thread_digest_bot.store import (
    AppendOnlyViolation,
    CollectingWebhookSink,
    DecisionStore,
    OrphanStateError,
    StoreConfig,
    StoreError,
    commit_message_for,
)
from thread_digest_bot.types import Author, Citation, Decision, DecisionLog


def _log(digest_key: str = "key-1", *, statement: str = "Ship Friday") -> DecisionLog:
    ada = Author(id="u_ada", display="Ada")
    return DecisionLog(
        channel_id="team-eng",
        range_label="last 3 messages",
        decisions=[
            Decision(
                statement=statement,
                citations=[Citation(message_id="m1", author=ada, permalink="https://t.me/c/1/1")],
            )
        ],
        participants=[ada],
        digest_key=digest_key,
    )


def _commit_count(repo_root: Path) -> int:
    repo = Repo(repo_root)
    if not repo.head.is_valid():
        return 0
    return sum(1 for _ in repo.iter_commits())


def _hold_lock_worker(lock_path: str, acquired: object, release: object) -> None:
    """Child-process entry point: hold the store lock until told to release.

    Defined at module scope so it is picklable under the ``spawn`` start method.
    """
    from thread_digest_bot.store import _FileLock

    with _FileLock(Path(lock_path), timeout=5.0, poll=0.01):
        acquired.set()  # type: ignore[attr-defined]
        release.wait(timeout=10.0)  # type: ignore[attr-defined]


def _hold_lock_forever_worker(lock_path: str, acquired: object) -> None:
    """Acquire the lock and block forever, to be killed mid-hold (simulated crash)."""
    import time

    from thread_digest_bot.store import _FileLock

    with _FileLock(Path(lock_path), timeout=5.0, poll=0.01):
        acquired.set()  # type: ignore[attr-defined]
        while True:  # pragma: no cover - runs in a child that is SIGKILLed
            time.sleep(0.05)


def test_append_creates_file_and_commits(temp_git_repo: Path) -> None:
    store = DecisionStore(temp_git_repo)
    result = store.append(_log())

    assert result.committed is True
    assert result.skipped_duplicate is False
    assert result.path.exists()
    assert result.path.name == "team-eng.md"
    assert "Ship Friday" in result.path.read_text(encoding="utf-8")
    assert _commit_count(temp_git_repo) == 1


def test_commit_message_format(temp_git_repo: Path) -> None:
    log = _log()
    store = DecisionStore(temp_git_repo)
    result = store.append(log)
    assert result.commit_message == "digest(team-eng): last 3 messages"
    assert commit_message_for(log) == "digest(team-eng): last 3 messages"
    repo = Repo(temp_git_repo)
    assert repo.head.commit.message.strip() == "digest(team-eng): last 3 messages"


def test_second_append_preserves_prior_as_prefix(temp_git_repo: Path) -> None:
    store = DecisionStore(temp_git_repo)
    store.append(_log("key-1", statement="First decision"))
    first_content = store.file_for("team-eng").read_text(encoding="utf-8")

    store.append(_log("key-2", statement="Second decision"))
    second_content = store.file_for("team-eng").read_text(encoding="utf-8")

    assert second_content.startswith(first_content)  # append-only: prior is a prefix
    assert "First decision" in second_content
    assert "Second decision" in second_content
    assert _commit_count(temp_git_repo) == 2


def test_idempotent_reappend_is_noop(temp_git_repo: Path) -> None:
    store = DecisionStore(temp_git_repo)
    log = _log("dup-key")
    first = store.append(log)
    assert first.committed is True

    second = store.append(log)  # same digest_key
    assert second.committed is False
    assert second.skipped_duplicate is True

    # No duplicate entry, and crucially no second commit.
    content = store.file_for("team-eng").read_text(encoding="utf-8")
    assert content.count("Digest key:") == 1
    assert _commit_count(temp_git_repo) == 1


def test_tamper_of_earlier_bytes_is_refused(temp_git_repo: Path) -> None:
    store = DecisionStore(temp_git_repo)
    store.append(_log("key-1", statement="Original decision"))

    # Simulate tampering: rewrite the committed file's earlier content.
    file_path = store.file_for("team-eng")
    file_path.write_text("TAMPERED — earlier bytes changed\n", encoding="utf-8")

    with pytest.raises(OrphanStateError):
        store.append(_log("key-2", statement="Another decision"))


def test_orphan_auto_commit(temp_git_repo: Path) -> None:
    store = DecisionStore(temp_git_repo, config=StoreConfig(orphan_policy="auto-commit"))
    store.append(_log("key-1", statement="Committed decision"))

    # A previous run wrote (append-only) but died before committing.
    file_path = store.file_for("team-eng")
    committed = file_path.read_text(encoding="utf-8")
    orphan_content = committed + "\n## orphan entry\n\norphaned but appended\n"
    file_path.write_text(orphan_content, encoding="utf-8")

    # The next append auto-commits the orphan first, then appends the new entry.
    store.append(_log("key-2", statement="New decision"))
    final = file_path.read_text(encoding="utf-8")
    assert "orphan entry" in final
    assert "New decision" in final
    assert _commit_count(temp_git_repo) == 3  # initial + orphan recovery + new append


def test_orphan_raise_policy(temp_git_repo: Path) -> None:
    store = DecisionStore(temp_git_repo, config=StoreConfig(orphan_policy="raise"))
    store.append(_log("key-1", statement="Committed decision"))

    file_path = store.file_for("team-eng")
    committed = file_path.read_text(encoding="utf-8")
    file_path.write_text(committed + "\n## orphan\n\nuncommitted append\n", encoding="utf-8")

    with pytest.raises(OrphanStateError):
        store.append(_log("key-2"))


def test_webhook_export_path(temp_git_repo: Path) -> None:
    sink = CollectingWebhookSink()
    store = DecisionStore(temp_git_repo, webhook=sink)
    store.append(_log())

    assert len(sink.sent) == 1
    channel_id, entry = sink.sent[0]
    assert channel_id == "team-eng"
    assert "Ship Friday" in entry


def test_no_commit_mode_writes_without_git(tmp_path: Path) -> None:
    # No git repo here; commit disabled, so the store only writes the file.
    store = DecisionStore(tmp_path, config=StoreConfig(commit=False))
    result = store.append(_log())
    assert result.committed is False
    assert result.skipped_duplicate is False
    assert result.path.read_text(encoding="utf-8").startswith("## last 3 messages")


def test_empty_digest_key_rejected(tmp_path: Path) -> None:
    store = DecisionStore(tmp_path, config=StoreConfig(commit=False))
    log = DecisionLog(channel_id="c", range_label="r", digest_key="")
    with pytest.raises(ValueError, match="digest_key"):
        store.append(log)


def test_is_processed_reflects_appended_keys(temp_git_repo: Path) -> None:
    store = DecisionStore(temp_git_repo)
    assert store.is_processed("team-eng", "key-1") is False
    store.append(_log("key-1"))
    assert store.is_processed("team-eng", "key-1") is True


def test_appendonly_violation_is_available() -> None:
    # The guard type is part of the public store API.
    assert issubclass(AppendOnlyViolation, Exception)


def test_processed_key_written_before_md_entry(temp_git_repo: Path) -> None:
    # The crash-atomicity invariant relies on the .processed key being recorded BEFORE
    # the .md entry, so capture the on-disk state at the moment the .md file is written.
    # Writes go through the store's atomic-write helper, so spy on that.
    import thread_digest_bot.store as store_mod

    store = DecisionStore(temp_git_repo)
    processed_path = store._processed_file_for("team-eng")
    md_path = store.file_for("team-eng")

    observed: dict[str, bool] = {}
    real_atomic_write = store_mod._atomic_write_text

    def spy_atomic_write(path: Path, data: str, **kwargs: object) -> None:
        if path == md_path:
            # When the .md entry is about to be written, the key must already be on disk.
            observed["key_present_when_md_written"] = (
                processed_path.exists() and "key-1" in processed_path.read_text(encoding="utf-8")
            )
        real_atomic_write(path, data, **kwargs)  # type: ignore[arg-type]

    import unittest.mock

    with unittest.mock.patch.object(store_mod, "_atomic_write_text", spy_atomic_write):
        store.append(_log("key-1"))

    assert observed.get("key_present_when_md_written") is True


def test_crash_after_key_before_entry_does_not_duplicate(temp_git_repo: Path) -> None:
    # Simulate the ONLY single-file partial write the fix permits: the .processed key
    # is recorded but the process dies before writing/committing the .md entry. A replay
    # of the same digest_key must be a no-op — a LOST entry is acceptable, a DUPLICATE
    # is not.
    store = DecisionStore(temp_git_repo)
    store.append(_log("key-1", statement="First decision"))
    baseline_md = store.file_for("team-eng").read_text(encoding="utf-8")
    baseline_commits = _commit_count(temp_git_repo)

    # One audit entry exists so far (each rendered entry carries exactly one digest-key
    # marker, so it is a reliable per-entry counter).
    assert baseline_md.count("Digest key:") == 1

    # Crash state: key-2 marked processed on disk, but its entry was never written.
    store._append_processed("team-eng", "key-2")

    # Replaying key-2 (deterministic key from the same message set) is a no-op append:
    # it must NOT re-append the entry (the audit trail must not gain a duplicate).
    result = store.append(_log("key-2", statement="Second decision"))
    assert result.skipped_duplicate is True
    assert result.committed is False  # the append itself committed nothing new

    final_md = store.file_for("team-eng").read_text(encoding="utf-8")
    # The audit trail (.md) is unchanged: no duplicate entry was produced. (Orphan
    # recovery may have reconciled the stray .processed key into a commit — that is the
    # intended consistency repair, not a duplicated audit entry.)
    assert final_md == baseline_md
    assert final_md.count("Digest key:") == 1
    assert "Second decision" not in final_md
    # No *append* commit was added beyond the orphan reconciliation (at most one).
    assert _commit_count(temp_git_repo) - baseline_commits <= 1


def test_orphan_recovery_commits_both_files_consistently(temp_git_repo: Path) -> None:
    # An orphaned .md entry (a prior run wrote append-only but died before committing)
    # carries its key in .processed on disk (written first). Recovery must commit BOTH
    # together so HEAD never holds the entry without its key, and the working tree is
    # left clean.
    store = DecisionStore(temp_git_repo, config=StoreConfig(orphan_policy="auto-commit"))
    store.append(_log("key-1", statement="Committed decision"))

    md_path = store.file_for("team-eng")
    committed_md = md_path.read_text(encoding="utf-8")
    md_path.write_text(committed_md + "\n## orphan entry\n\norphaned but appended\n", "utf-8")
    store._append_processed("team-eng", "key-2")  # the orphan's key, as the fix orders it

    store.append(_log("key-3", statement="New decision"))

    repo = Repo(temp_git_repo)
    assert repo.is_dirty(untracked_files=True) is False  # working tree reconciled
    # Both the orphan key and the new key are recorded in committed .processed.
    processed = store._processed_file_for("team-eng").read_text(encoding="utf-8")
    assert "key-2" in processed
    assert "key-3" in processed


def test_file_lock_acquires_and_releases_cleanly(tmp_path: Path) -> None:
    from thread_digest_bot.store import _FileLock

    lock_path = tmp_path / "team-eng.lock"

    # Acquire and release cleanly; a second acquisition then succeeds (no stale lock,
    # because fcntl.lockf releases on fd close rather than leaving an O_EXCL sentinel).
    with _FileLock(lock_path):
        pass
    with _FileLock(lock_path):
        pass


def test_file_lock_blocks_a_contending_process_until_released(tmp_path: Path) -> None:
    # fcntl.lockf locks are per-PROCESS, so genuine contention needs a second process.
    # A child holds the lock; the parent must time out, then succeed once the child
    # releases — proving cross-process mutual exclusion with no stale-lock deadlock.
    import multiprocessing as mp

    from thread_digest_bot.store import _FileLock

    lock_path = tmp_path / "team-eng.lock"
    ctx = mp.get_context("spawn")
    acquired = ctx.Event()  # set once the child holds the lock
    release = ctx.Event()  # parent sets this to let the child release

    proc = ctx.Process(target=_hold_lock_worker, args=(str(lock_path), acquired, release))
    proc.start()
    try:
        assert acquired.wait(timeout=5.0) is True  # child now holds the lock

        # While the child holds it, the parent cannot acquire within the timeout.
        with (
            pytest.raises(StoreError, match="another writer is active"),
            _FileLock(lock_path, timeout=0.3, poll=0.01),
        ):
            pass

        # Let the child release; the parent then acquires promptly.
        release.set()
        proc.join(timeout=5.0)
        assert proc.exitcode == 0
        with _FileLock(lock_path, timeout=1.0, poll=0.01):
            pass
    finally:
        release.set()
        proc.join(timeout=5.0)
        if proc.is_alive():  # pragma: no cover - defensive cleanup
            proc.terminate()


def test_file_lock_does_not_stale_after_holder_is_killed(tmp_path: Path) -> None:
    # Regression for the stale-lock hazard: a holder that dies WITHOUT releasing must
    # not deadlock future runs. fcntl.lockf is released by the kernel on process death,
    # so the parent acquires quickly; an O_CREAT|O_EXCL sentinel would linger and block
    # for the full timeout instead.
    import multiprocessing as mp
    import signal

    from thread_digest_bot.store import _FileLock

    lock_path = tmp_path / "team-eng.lock"
    ctx = mp.get_context("spawn")
    acquired = ctx.Event()

    proc = ctx.Process(target=_hold_lock_forever_worker, args=(str(lock_path), acquired))
    proc.start()
    try:
        assert acquired.wait(timeout=5.0) is True  # child holds the lock
        os.kill(proc.pid, signal.SIGKILL)  # simulate a crash mid-hold (no clean release)
        proc.join(timeout=5.0)
        assert proc.is_alive() is False

        # The lock file may still exist on disk, but holds no lock; acquisition succeeds
        # well within a short timeout (no full-timeout deadlock from a stale sentinel).
        with _FileLock(lock_path, timeout=2.0, poll=0.01):
            pass
    finally:
        if proc.is_alive():  # pragma: no cover - defensive cleanup
            proc.terminate()
            proc.join(timeout=5.0)


# --- regression: atomic writes, lock-file persistence, in-process serialization,
#     sidecar validation on recovery --------------------------------------------------


class _TornWriteError(RuntimeError):
    """Raised by the spy file handle to simulate an IO error mid-write."""


def _interrupting_fdopen(real_fdopen: object, prefix_bytes: int = 4) -> object:
    """Return an ``os.fdopen`` replacement whose handle writes a few bytes then raises.

    This deterministically simulates a *torn* write (e.g. ``ENOSPC`` partway through):
    only ``prefix_bytes`` of the payload reach the temp file before :class:`_TornWriteError`
    propagates, with no real disk pressure and entirely offline.
    """

    def fake_fdopen(fd: int, *args: object, **kwargs: object) -> object:
        handle = real_fdopen(fd, *args, **kwargs)  # type: ignore[operator]
        real_write = handle.write

        def torn_write(text: str) -> int:
            real_write(text[:prefix_bytes])  # partial bytes hit the temp file
            raise _TornWriteError("simulated IO error mid-write")

        handle.write = torn_write  # type: ignore[method-assign]
        return handle

    return fake_fdopen


def test_atomic_write_interrupted_leaves_full_old_content_not_truncated(tmp_path: Path) -> None:
    # A crash/IO error mid-write must never leave a truncated, partial file visible: the
    # target is either the full OLD or full NEW content. We simulate the interruption by
    # patching os.fdopen so the temp-file write throws after a few bytes; the temp file is
    # discarded and the original target is untouched (no os.replace happened).
    import unittest.mock

    import thread_digest_bot.store as store_mod

    target = tmp_path / "team-eng.md"
    old_content = "## old entry\n\nfull previous content\n"
    target.write_text(old_content, encoding="utf-8")

    real_fdopen = os.fdopen
    with (
        unittest.mock.patch.object(store_mod.os, "fdopen", _interrupting_fdopen(real_fdopen)),
        pytest.raises(_TornWriteError),
    ):
        store_mod._atomic_write_text(target, "## new entry\n\nmuch longer brand new content\n")

    # The visible file is the FULL old content — never a truncated mix.
    assert target.read_text(encoding="utf-8") == old_content
    # No stray temp file is left behind on failure.
    assert sorted(tmp_path.glob(".team-eng.md.*.tmp")) == []


def test_append_md_torn_write_does_not_commit_partial_entry(temp_git_repo: Path) -> None:
    # End-to-end: if the .md write tears mid-way during append(), the on-disk .md must
    # remain the full prior content (here: empty/absent → unchanged), never a truncated
    # half-entry that orphan recovery could later commit as a corrupt audit record.
    import unittest.mock

    import thread_digest_bot.store as store_mod

    store = DecisionStore(temp_git_repo)
    store.append(_log("key-1", statement="First decision"))
    md_path = store.file_for("team-eng")
    baseline = md_path.read_text(encoding="utf-8")

    # Make ONLY the .md atomic write tear; the .processed write already succeeded. We
    # wrap the real helper and inject the torn os.fdopen just for the .md path.
    real_fdopen = os.fdopen
    real_atomic = store_mod._atomic_write_text

    def tearing_atomic(path: Path, data: str, **kwargs: object) -> None:
        if path == md_path:
            with unittest.mock.patch.object(
                store_mod.os, "fdopen", _interrupting_fdopen(real_fdopen)
            ):
                real_atomic(path, data, **kwargs)  # type: ignore[arg-type]
        else:
            real_atomic(path, data, **kwargs)  # type: ignore[arg-type]

    with (
        unittest.mock.patch.object(store_mod, "_atomic_write_text", tearing_atomic),
        pytest.raises(_TornWriteError),
    ):
        store.append(_log("key-2", statement="Second decision"))

    # The .md is byte-identical to before: no truncated/partial second entry.
    assert md_path.read_text(encoding="utf-8") == baseline
    assert md_path.read_text(encoding="utf-8").count("Digest key:") == 1
    assert "Second decision" not in md_path.read_text(encoding="utf-8")


def test_lock_file_is_not_unlinked_on_release(tmp_path: Path) -> None:
    # Regression for the path/inode race: __exit__ must NOT unlink the lock path. A
    # persistent lock file is harmless with fcntl.lockf (the lock lives on the fd/inode,
    # not the name); unlinking it would let a blocked waiter and a fresh same-name opener
    # both hold a lock at once. Assert the file survives release.
    from thread_digest_bot.store import _FileLock

    lock_path = tmp_path / "team-eng.lock"
    with _FileLock(lock_path):
        assert lock_path.exists()
    # Crucially still present after release (no unlink).
    assert lock_path.exists()

    # And the inode is stable across acquisitions (same anchor, not recreated).
    inode_first = lock_path.stat().st_ino
    with _FileLock(lock_path):
        pass
    assert lock_path.stat().st_ino == inode_first


def test_persistent_lock_file_does_not_dirty_working_tree(temp_git_repo: Path) -> None:
    # Because the lock file is no longer unlinked, the store git-ignores *.lock in the
    # decisions dir so the persistent lock never shows up as untracked/dirty.
    store = DecisionStore(temp_git_repo)
    store.append(_log("key-1"))

    lock_path = store._lock_file_for("team-eng")
    assert lock_path.exists()  # persistent
    repo = Repo(temp_git_repo)
    assert repo.is_dirty(untracked_files=True) is False  # but ignored → tree stays clean
    # The ignore file was committed (rode along with the digest commit, no extra commit).
    assert (store.decisions_path / ".gitignore").read_text(encoding="utf-8") == "*.lock\n"
    assert _commit_count(temp_git_repo) == 1


def test_two_inprocess_threads_append_distinct_nonduplicated_entries(temp_git_repo: Path) -> None:
    # fcntl.lockf is per-PROCESS, so without the process-local guard two threads in THIS
    # interpreter (e.g. a scheduler timer firing a rollup while a /digest runs) would both
    # enter the read-append-commit critical section, racing on the same in-memory prior
    # snapshot and losing/duplicating an entry. A threading.Barrier releases both threads
    # simultaneously (deterministic, offline) to maximize the race; the guard must still
    # serialize them into two distinct, non-interleaved, non-duplicated entries.
    import threading

    store = DecisionStore(temp_git_repo)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    results: list[object] = []

    def worker(key: str, statement: str) -> None:
        try:
            barrier.wait(timeout=5.0)  # both threads proceed together
            results.append(store.append(_log(key, statement=statement)))
        except BaseException as exc:  # pragma: no cover - surfaced via assertion below
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("key-A", "Decision A"))
    t2 = threading.Thread(target=worker, args=("key-B", "Decision B"))
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert errors == []  # no AppendOnlyViolation / lost-update crash
    content = store.file_for("team-eng").read_text(encoding="utf-8")
    # Exactly two distinct entries, neither lost nor duplicated nor interleaved.
    assert content.count("Digest key:") == 2
    assert "Decision A" in content
    assert "Decision B" in content
    assert content.count("Decision A") == 1
    assert content.count("Decision B") == 1
    # Both keys recorded and both commits made (one per entry).
    assert store.is_processed("team-eng", "key-A") is True
    assert store.is_processed("team-eng", "key-B") is True
    assert _commit_count(temp_git_repo) == 2
    # Whichever committed first is a prefix of the final content (append-only preserved).
    assert content.startswith("## last 3 messages")


def test_corrupt_processed_sidecar_on_recovery_raises(temp_git_repo: Path) -> None:
    # On orphan recovery the store must validate the .processed sidecar rather than
    # silently trust damaged data. A corrupt key line (here an embedded NUL + spaces from
    # a hypothetical torn write) must raise OrphanStateError. We pair it with an orphaned
    # .md so the recovery path actually runs.
    store = DecisionStore(temp_git_repo, config=StoreConfig(orphan_policy="auto-commit"))
    store.append(_log("key-1", statement="First decision"))

    processed_path = store._processed_file_for("team-eng")
    processed_path.write_text("key-1\nkey 2 broken\x00bytes\n", encoding="utf-8")

    md_path = store.file_for("team-eng")
    md_path.write_text(md_path.read_text(encoding="utf-8") + "\n## orphan\n\nbody\n", "utf-8")

    with pytest.raises(OrphanStateError, match="sidecar"):
        store.append(_log("key-9", statement="Ninth decision"))


def test_corrupt_processed_sidecar_with_internal_whitespace_raises(temp_git_repo: Path) -> None:
    # A subtler corruption: a line with interior whitespace (not a single well-formed
    # key). It must also be rejected on recovery so a damaged key can never silently slip
    # past the dedup check and cause a duplicate replay.
    store = DecisionStore(temp_git_repo, config=StoreConfig(orphan_policy="auto-commit"))
    store.append(_log("key-1", statement="First decision"))

    processed_path = store._processed_file_for("team-eng")
    processed_path.write_text("key-1\nkey one two\n", encoding="utf-8")
    md_path = store.file_for("team-eng")
    md_path.write_text(md_path.read_text(encoding="utf-8") + "\n## orphan\n\nbody\n", "utf-8")

    with pytest.raises(OrphanStateError, match="sidecar"):
        store.append(_log("key-9", statement="Ninth decision"))


def test_clean_sidecar_on_recovery_is_accepted(temp_git_repo: Path) -> None:
    # The validation must NOT reject a well-formed sidecar: a normal orphan (entry beyond
    # HEAD with its key recorded first) still recovers cleanly.
    store = DecisionStore(temp_git_repo, config=StoreConfig(orphan_policy="auto-commit"))
    store.append(_log("key-1", statement="First decision"))

    md_path = store.file_for("team-eng")
    md_path.write_text(md_path.read_text(encoding="utf-8") + "\n## orphan\n\nbody\n", "utf-8")
    store._append_processed("team-eng", "key-2")  # well-formed key, written first

    store.append(_log("key-3", statement="Third decision"))  # must not raise

    repo = Repo(temp_git_repo)
    assert repo.is_dirty(untracked_files=True) is False
    processed = store._processed_file_for("team-eng").read_text(encoding="utf-8")
    assert "key-2" in processed
    assert "key-3" in processed
