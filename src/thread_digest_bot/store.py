"""Append-only Markdown decision log committed to Git.

Per channel, a Markdown file ``docs/decisions/<channel>.md`` accumulates rendered
digest entries. The store guarantees the audit trail is **append-only** and
**idempotent**, and refuses to commit on a detected divergence.

Critical-section design (no TOCTOU)
-----------------------------------
The whole read -> append -> verify -> write -> commit sequence runs inside a lock that
serializes **both** same-process threads (a process-local :class:`threading.Lock`) and
other processes (a POSIX ``fcntl.lockf`` advisory lock — which is itself per-process and
so cannot serialize two threads on its own). The append builds the new content from an
**in-memory snapshot** taken at lock acquisition and asserts ``new.startswith(prior)`` on
that snapshot before writing, so a concurrent writer cannot slip between the check and the
write. The lock + prefix check make a single-writer violation *fail loudly* rather than
silently corrupt history.

Idempotency & crash atomicity
-----------------------------
Processed ``digest_key``s are persisted in a sidecar ``<channel>.processed`` file.
Re-appending an already-processed key is a no-op: no duplicate entry, no second commit.

Both the ``.md`` entry and the ``.processed`` sidecar are written **atomically** (temp
file in the same directory + ``os.replace``), so a crash or IO error mid-write can never
leave a truncated, partial file on disk: the visible file is always the full old or full
new content. A corrupt audit entry therefore cannot be produced, and orphan recovery
cannot be tricked into committing/trusting damaged bytes.

The processed key is written **before** the ``.md`` entry, and both are staged into a
**single** Git commit. This biases any crash toward a *lost* entry (the key is recorded
without its entry, so a replay is a safe no-op) rather than a *duplicated* audit entry
(which an entry recorded without its key would cause). The committed history is therefore
always consistent: ``HEAD`` never holds an entry whose key is missing from ``.processed``.

Orphan recovery
---------------
If the working tree diverges from ``HEAD`` (a previous run wrote but died before
committing), the store either auto-commits both files together or raises
:class:`OrphanStateError`, per :class:`StoreConfig.orphan_policy`. It never appends on
top of an uncommitted divergence, and it validates the on-disk ``.processed`` sidecar
parses as well-formed keys before trusting/committing it — a corrupt sidecar raises
:class:`OrphanStateError` rather than risking a duplicate replay.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from thread_digest_bot.render import render_markdown_entry
from thread_digest_bot.types import DecisionLog

if TYPE_CHECKING:
    from git import Repo

OrphanPolicy = Literal["auto-commit", "raise"]


def _atomic_write_text(path: Path, data: str, *, encoding: str = "utf-8") -> None:
    """Write ``data`` to ``path`` atomically via a same-directory temp file + rename.

    The content is written to a uniquely named temporary file in the *same directory*
    as ``path`` (so the final :func:`os.replace` is a same-filesystem, atomic rename on
    POSIX), then ``fsync``'d and renamed over the target. A crash or IO error mid-write
    can therefore only leave behind a stray temp file; the visible ``path`` is always
    either the full previous content or the full new content, never a truncated/partial
    one. The stray temp file is removed on any failure before the rename.
    """
    directory = path.parent
    # mkstemp creates the file with O_EXCL and a unique name in ``directory``.
    fd, tmp_name = _mkstemp_in(directory, path.name)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)  # atomic rename over the target (POSIX)
    except BaseException:
        # Never leave a stray temp file behind on failure.
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _mkstemp_in(directory: Path, base_name: str) -> tuple[int, str]:
    """Create a unique temp file in ``directory`` and return ``(fd, path)``.

    Split out so tests can patch the write step (between create and rename) to simulate
    an interrupted write deterministically.
    """
    import tempfile

    return tempfile.mkstemp(prefix=f".{base_name}.", suffix=".tmp", dir=directory)


# Process-local serialization for same-process callers. ``fcntl.lockf`` is per-PROCESS,
# so two threads in THIS interpreter (e.g. scheduler timer threads firing a rollup while
# a ``/digest`` runs) would both be granted the same advisory lock and enter the
# read-append-commit critical section concurrently. A per-lock-path ``threading.Lock``,
# acquired AROUND the ``fcntl`` acquisition, serializes same-process callers; cross-process
# callers still serialize via ``fcntl``. The registry itself is guarded by a small module
# lock for the get-or-create.
_PROCESS_LOCK_REGISTRY: dict[str, threading.Lock] = {}
_PROCESS_LOCK_REGISTRY_GUARD = threading.Lock()


def _process_lock_for(lock_path: Path) -> threading.Lock:
    """Return the process-local :class:`threading.Lock` for ``lock_path`` (get-or-create).

    Keyed by the absolute, resolved path string so two equivalent paths share one lock.
    """
    key = str(lock_path.resolve())
    with _PROCESS_LOCK_REGISTRY_GUARD:
        lock = _PROCESS_LOCK_REGISTRY.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCK_REGISTRY[key] = lock
        return lock


class StoreError(RuntimeError):
    """Base class for store errors."""


class AppendOnlyViolation(StoreError):
    """Raised when an append would not preserve the prior content as a prefix."""


class OrphanStateError(StoreError):
    """Raised when the working file diverges from HEAD and the policy is ``raise``."""


@dataclass(frozen=True)
class StoreConfig:
    """Configuration for the Git decision-log store.

    Attributes:
        decisions_dir: Directory (relative to the repo root) holding per-channel logs.
        orphan_policy: How to handle an uncommitted divergence on append.
        commit: When ``False`` the store writes files but skips ``git`` operations
            (useful for the CLI ``--out`` / no-commit path and for the webhook export
            sink).
    """

    decisions_dir: str = "docs/decisions"
    orphan_policy: OrphanPolicy = "auto-commit"
    commit: bool = True


@dataclass
class AppendResult:
    """Outcome of an :meth:`DecisionStore.append` call.

    Attributes:
        committed: Whether a new commit was created.
        skipped_duplicate: Whether the append was a no-op due to idempotency.
        path: The Markdown file written/targeted.
        commit_message: The commit message used (when committed).
    """

    committed: bool
    skipped_duplicate: bool
    path: Path
    commit_message: str | None = None


def _safe_channel_filename(channel_id: str) -> str:
    """Map a channel id to a filesystem-safe ``.md`` filename."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in channel_id)
    return f"{safe or 'channel'}.md"


def commit_message_for(log: DecisionLog) -> str:
    """Return the deterministic commit message for a digest log."""
    return f"digest({log.channel_id}): {log.range_label}"


class _FileLock:
    """A cross- *and* same-process advisory lock around the critical section.

    Two layers of mutual exclusion stack here:

    * A process-local :class:`threading.Lock` (from :func:`_process_lock_for`, keyed by
      the lock path) serializes threads within THIS interpreter. ``fcntl.lockf`` is a
      per-process lock, so without this two threads in the same process (e.g. a scheduler
      timer firing a rollup while a ``/digest`` runs) would both be granted the advisory
      lock and enter the read-append-commit critical section concurrently.
    * :func:`fcntl.lockf` (a POSIX advisory range lock) on a long-lived lock file
      serializes across *different* processes. The lock is associated with the open file
      descriptor and is released by the kernel when the fd is closed — including on an
      abnormal exit — so a crash never leaves a *stale* lock that would block future runs
      for the full timeout (the failure mode of an ``O_CREAT | O_EXCL`` sentinel file).

    The thread lock is acquired *around* (before) the ``fcntl`` acquisition and released
    *after* it, so same-process callers serialize and cross-process callers still
    serialize via ``fcntl``. The lock file is intentionally **never unlinked**: with
    ``fcntl.lockf`` the lock lives on the open fd/inode, not the path name. Unlinking the
    path after releasing would open a path/inode race (a blocked waiter is granted the
    lock on the just-released inode, the unlink removes the name, and a third process
    re-creates the same *name* as a NEW inode and is immediately granted a second lock —
    two writers in the critical section at once). A persistent lock file is harmless.

    A short non-blocking spin (rather than a single blocking ``lockf``) keeps the wait
    bounded and lets us raise a clear, actionable :class:`StoreError` on contention.
    """

    def __init__(self, lock_path: Path, *, timeout: float = 10.0, poll: float = 0.02) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._poll = poll
        self._fd: int | None = None
        self._thread_lock = _process_lock_for(lock_path)
        self._thread_locked = False

    def __enter__(self) -> _FileLock:
        import fcntl
        import time

        deadline = time.monotonic() + self._timeout

        # 1. Serialize same-process callers first (bounded by the same deadline).
        if not self._thread_lock.acquire(timeout=self._timeout):  # pragma: no cover - timing
            raise StoreError(
                f"Could not acquire store lock at {self._lock_path} within "
                f"{self._timeout}s; another writer is active."
            )
        self._thread_locked = True

        # 2. Then serialize across processes via the advisory file lock.
        # Open (creating if needed) without O_EXCL: the file is a stable lock anchor,
        # not the lock itself. The advisory lock below provides mutual exclusion.
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except BaseException:
            self._release_thread_lock()
            raise
        while True:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    self._release_thread_lock()
                    raise StoreError(
                        f"Could not acquire store lock at {self._lock_path} within "
                        f"{self._timeout}s; another writer is active."
                    ) from None
                time.sleep(self._poll)

    def __exit__(self, *_exc: object) -> None:
        import fcntl

        try:
            if self._fd is not None:
                with contextlib.suppress(OSError):
                    fcntl.lockf(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)  # closing the fd also drops any held lock
                self._fd = None
        finally:
            # Release the process-local lock LAST, mirroring acquisition order. The lock
            # file is deliberately left on disk (see class docstring) to avoid a
            # path/inode race.
            self._release_thread_lock()

    def _release_thread_lock(self) -> None:
        if self._thread_locked:
            self._thread_locked = False
            self._thread_lock.release()


class WebhookSink(Protocol):
    """A sink for exporting rendered entries instead of (or alongside) Git."""

    def send(self, channel_id: str, entry: str) -> None:
        """Deliver a rendered Markdown entry for ``channel_id``."""
        ...


class DecisionStore:
    """An append-only, idempotent decision-log store backed by a Git repo.

    Args:
        repo_root: Path to the Git working tree root.
        config: Store configuration.
        webhook: Optional sink; when provided each appended entry is also exported.
    """

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        *,
        config: StoreConfig | None = None,
        webhook: WebhookSink | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.config = config or StoreConfig()
        self.webhook = webhook
        self._repo: Repo | None = None

    # -- paths ---------------------------------------------------------------

    @property
    def decisions_path(self) -> Path:
        """Absolute path to the per-channel decisions directory."""
        return self.repo_root / self.config.decisions_dir

    def file_for(self, channel_id: str) -> Path:
        """Return the Markdown log path for a channel."""
        return self.decisions_path / _safe_channel_filename(channel_id)

    def _processed_file_for(self, channel_id: str) -> Path:
        return self.decisions_path / f"{_safe_channel_filename(channel_id)}.processed"

    def _lock_file_for(self, channel_id: str) -> Path:
        return self.decisions_path / f"{_safe_channel_filename(channel_id)}.lock"

    def _ignore_file(self) -> Path:
        return self.decisions_path / ".gitignore"

    def _ensure_lock_ignore(self) -> None:
        """Ensure ``*.lock`` files in the decisions dir are git-ignored.

        The advisory lock file is intentionally persistent (see :class:`_FileLock`); it
        is a transient runtime artifact, not part of the audit trail. Dropping a small
        ``.gitignore`` beside the logs keeps lingering lock files out of the Git working
        tree (so ``is_dirty`` and orphan detection ignore them) without the store ever
        having to clean them up.

        Only the file is written here, once, when missing; it is *not* given its own
        commit. Instead the store folds the ignore path into the next commit it already
        makes (the digest append or the orphan recovery), so it rides along for free and
        never adds an extra commit to the history.
        """
        ignore_path = self._ignore_file()
        if not ignore_path.exists():
            _atomic_write_text(ignore_path, "*.lock\n")

    # -- git -----------------------------------------------------------------

    def _git(self) -> Repo:
        if self._repo is None:
            from git import Repo

            self._repo = Repo(self.repo_root)
        return self._repo

    def _committed_blob(self, rel_path: str) -> str:
        """Return the file's content at ``HEAD`` (empty string if absent/unborn)."""
        repo = self._git()
        if not repo.head.is_valid():  # unborn branch / no commits yet
            return ""
        head_commit = repo.head.commit
        try:
            blob = head_commit.tree / rel_path
        except KeyError:
            return ""
        data = blob.data_stream.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.repo_root).as_posix()

    # -- processed-key persistence ------------------------------------------

    def _read_processed(self, channel_id: str) -> set[str]:
        path = self._processed_file_for(channel_id)
        if not path.exists():
            return set()
        lines = path.read_text(encoding="utf-8").splitlines()
        return {line.strip() for line in lines if line.strip()}

    @staticmethod
    def _validate_processed_text(text: str, rel: str) -> None:
        """Assert every non-blank line of a ``.processed`` sidecar is a well-formed key.

        A processed key is a single whitespace-free token per line (the format
        :meth:`_append_processed` writes). A line that round-trips differently from its
        stripped form — embedded whitespace, an interior NUL from a torn write, etc. —
        means the sidecar is corrupt; trusting it could let a damaged key fail dedup and
        cause a duplicate replay, so we refuse rather than silently proceed.
        """
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped != raw or "\x00" in stripped or any(ch.isspace() for ch in stripped):
                raise OrphanStateError(
                    f"Processed sidecar {rel} is corrupt (malformed key line); "
                    f"refusing to trust it during orphan recovery."
                )

    def _append_processed(self, channel_id: str, digest_key: str) -> None:
        path = self._processed_file_for(channel_id)
        existing = self._read_processed(channel_id)
        existing.add(digest_key)
        # Atomic write: a crash mid-write can never leave a truncated sidecar (which
        # orphan recovery would otherwise distrust or, worse, mis-parse).
        _atomic_write_text(path, "\n".join(sorted(existing)) + "\n")

    # -- orphan recovery -----------------------------------------------------

    def _recover_orphan(self, channel_id: str, file_path: Path) -> str:
        """Reconcile the working tree (``.md`` + ``.processed``) with HEAD.

        Detects the case where a prior run wrote one or both files but died before
        committing. The two cases (from the ``.processed``-before-``.md`` write order):

        * ``.md`` has an orphaned entry beyond HEAD — its key is already on disk in
          ``.processed`` (written first), so committing both as-is keeps the trail
          consistent.
        * only ``.processed`` diverged (crash after the key write, before the entry
          write) — the entry is lost-but-not-duplicated; committing both reconciles
          the working tree so the next run starts clean.

        Auto-commits both files together, or raises per :class:`StoreConfig.orphan_policy`.
        Returns the current on-disk ``.md`` content (``""`` if absent) for use as the
        in-memory prior snapshot.
        """
        if not self.config.commit:
            return file_path.read_text(encoding="utf-8") if file_path.exists() else ""

        rel = self._rel(file_path)
        committed = self._committed_blob(rel)
        on_disk = file_path.read_text(encoding="utf-8") if file_path.exists() else ""

        processed_path = self._processed_file_for(channel_id)
        processed_committed = self._committed_blob(self._rel(processed_path))
        processed_on_disk = (
            processed_path.read_text(encoding="utf-8") if processed_path.exists() else ""
        )

        md_diverged = on_disk != committed
        processed_diverged = processed_on_disk != processed_committed
        if not md_diverged and not processed_diverged:
            return on_disk

        # A divergence means a prior run wrote but did not commit. Before trusting or
        # committing the uncommitted sidecar, confirm it parses as well-formed keys.
        # (Atomic writes make corruption unreachable in practice, but recovery must
        # never silently commit/trust a damaged sidecar.)
        self._validate_processed_text(processed_on_disk, self._rel(processed_path))

        # The .md trail must only ever grow; earlier bytes changing is tampering.
        if md_diverged and not on_disk.startswith(committed):
            raise OrphanStateError(
                f"Working file {rel} diverges from HEAD and is not an append "
                f"(earlier content changed); refusing to proceed."
            )
        if self.config.orphan_policy == "raise":
            raise OrphanStateError(
                f"Working tree for channel {channel_id} has uncommitted content beyond "
                f"HEAD (a previous run may have died before committing)."
            )
        # auto-commit both files together so HEAD reflects a consistent working tree.
        # The lock-ignore file (if newly created) rides along so it never lingers
        # untracked.
        self._commit_paths(
            [self._ignore_file(), processed_path, file_path],
            f"recover orphan digest({channel_id})",
        )
        return on_disk

    # -- commit --------------------------------------------------------------

    def _commit_paths(self, paths: list[Path], message: str) -> None:
        repo = self._git()
        rels = [self._rel(p) for p in paths if p.exists()]
        if not rels:
            return
        repo.index.add(rels)
        repo.index.commit(message)

    # -- public API ----------------------------------------------------------

    def is_processed(self, channel_id: str, digest_key: str) -> bool:
        """Return whether ``digest_key`` was already appended for ``channel_id``."""
        return digest_key in self._read_processed(channel_id)

    def append(self, log: DecisionLog) -> AppendResult:
        """Append a rendered ``log`` entry, idempotently and append-only.

        The full critical section (orphan check, dedup check, read, append, prefix
        verify, write, commit) runs inside an OS file lock using an in-memory snapshot,
        so concurrent writers cannot create a TOCTOU window.

        Args:
            log: The grounded decision log to append. Must carry a non-empty
                ``digest_key``.

        Returns:
            An :class:`AppendResult` describing whether a commit occurred or the call
            was a no-op duplicate.

        Raises:
            ValueError: If ``log.digest_key`` is empty.
            AppendOnlyViolation: If the computed new content would not preserve the
                prior content as a prefix.
            OrphanStateError: If the working file diverges from HEAD under a ``raise``
                policy, or if earlier bytes were tampered.
        """
        if not log.digest_key:
            raise ValueError("DecisionLog.digest_key must be set before appending.")

        self.decisions_path.mkdir(parents=True, exist_ok=True)
        file_path = self.file_for(log.channel_id)
        entry = render_markdown_entry(log)

        with _FileLock(self._lock_file_for(log.channel_id)):
            # 0. Keep the persistent advisory lock files out of the Git working tree.
            self._ensure_lock_ignore()

            # 1. Reconcile any uncommitted orphan first.
            prior = self._recover_orphan(log.channel_id, file_path)

            # 2. Idempotency: already processed -> no-op.
            if self.is_processed(log.channel_id, log.digest_key):
                return AppendResult(
                    committed=False,
                    skipped_duplicate=True,
                    path=file_path,
                )

            # 3. Build the new content from the in-memory snapshot.
            separator = "" if (prior == "" or prior.endswith("\n")) else "\n"
            new_content = prior + separator + entry

            # 4. Append-only prefix assertion on the in-memory snapshot.
            if not new_content.startswith(prior):  # pragma: no cover - invariant guard
                raise AppendOnlyViolation(
                    f"Append for channel {log.channel_id} would not preserve prior content."
                )

            # 5. Record the processed key BEFORE writing the .md entry, so a crash
            #    between the two writes can only *lose* an entry (the key is marked
            #    processed without the entry → a replay is a safe no-op), never produce
            #    a *duplicate* (which an entry-without-key would, by passing the dedup
            #    check on replay). The two files are committed together below in a single
            #    atomic commit, so the committed audit trail is always consistent.
            self._append_processed(log.channel_id, log.digest_key)
            # Atomic write: a crash mid-write can never leave a truncated, non-empty .md
            # that orphan recovery would later commit as a corrupt audit entry. The
            # visible file is always the full old or full new content.
            _atomic_write_text(file_path, new_content)

            # 6. Optional webhook export.
            if self.webhook is not None:
                self.webhook.send(log.channel_id, entry)

            # 7. Commit both files in a SINGLE commit (atomic: HEAD never holds the
            #    entry without its processed key, or vice versa). The lock-ignore file
            #    (if newly created) rides in this same commit, adding no extra history.
            commit_message: str | None = None
            committed = False
            if self.config.commit:
                commit_message = commit_message_for(log)
                self._commit_paths(
                    [self._ignore_file(), self._processed_file_for(log.channel_id), file_path],
                    commit_message,
                )
                committed = True

            return AppendResult(
                committed=committed,
                skipped_duplicate=False,
                path=file_path,
                commit_message=commit_message,
            )


@dataclass
class CollectingWebhookSink:
    """An in-memory :class:`WebhookSink` capturing exported entries (for tests/demos)."""

    sent: list[tuple[str, str]] = field(default_factory=list)

    def send(self, channel_id: str, entry: str) -> None:
        """Record ``(channel_id, entry)``."""
        self.sent.append((channel_id, entry))
