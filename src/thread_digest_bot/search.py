"""Read-side search over the committed decision logs.

The store (:mod:`thread_digest_bot.store`) is append-only *write*; this module closes
the loop the pitch promises — a "durable, attributed, **searchable** record" — by
parsing the per-channel Markdown files back into entries and matching a query against
the extracted decisions, action items, and open questions.

Parsing is pure standard-library string work: no new dependencies, no Git calls, and no
network. Each ``docs/decisions/<channel>.md`` file is split on the ``## <range label>``
digest heading emitted by :func:`thread_digest_bot.render.render_markdown_entry`, and
every item bullet under a ``### Decisions`` / ``### Action items`` / ``### Open questions``
section becomes a searchable line carrying its channel, digest key, and (first) permalink.
A missing ``docs/decisions`` directory yields an empty result rather than an error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Directory (relative to the repo root) holding the per-channel decision logs. Mirrors
#: :attr:`thread_digest_bot.store.StoreConfig.decisions_dir`'s default.
_DECISIONS_DIR = "docs/decisions"

#: The digest-entry heading emitted by ``render_markdown_entry`` (``## <range label>``).
#: A ``### ...`` section heading has a ``#`` where this expects a space, so it never
#: matches — entries split cleanly on the top-level ``## `` heading only.
_ENTRY_HEADING_PREFIX = "## "

#: Section heading -> item kind. The kinds are the searchable categories exposed to the
#: ``kind=`` filter of :func:`search_logs`.
_SECTION_KINDS = {
    "### Decisions": "decision",
    "### Action items": "action_item",
    "### Open questions": "open_question",
}

_CHANNEL_RE = re.compile(r"^- \*\*Channel:\*\* `(.+)`$")
_DIGEST_KEY_RE = re.compile(r"^- \*\*Digest key:\*\* `(.+)`$")

#: First ``](<url>)`` link target on a ``_Sources:_`` line, if any.
_PERMALINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")

#: The explicit "nothing here" marker a section renders for an empty list; never a real
#: item, so it is skipped during parsing.
_NONE_MARKER = "_none_"


@dataclass(frozen=True)
class SearchHit:
    """A single match from :func:`search_logs`.

    Attributes:
        channel: The channel id the matching entry belongs to.
        digest_key: The digest key of the entry the match was found in.
        kind: The item category — ``"decision"``, ``"action_item"``, or
            ``"open_question"``.
        line: The matched item text (the rendered bullet, sans the ``- `` prefix).
        permalink: The first source permalink on the item's ``_Sources:_`` line, or
            ``None`` when the item has no linked citation.
    """

    channel: str
    digest_key: str
    kind: str
    line: str
    permalink: str | None


@dataclass
class _Item:
    """A parsed item bullet within an entry (internal to parsing)."""

    kind: str
    text: str
    permalink: str | None = None


@dataclass
class _Entry:
    """A parsed digest entry (internal to parsing)."""

    channel: str = ""
    digest_key: str = ""
    items: list[_Item] = field(default_factory=list)


def _split_entry_blocks(text: str) -> list[list[str]]:
    """Split a channel log file into per-entry line blocks on the ``## `` heading.

    Any preamble before the first heading (there is none in a well-formed file) is
    ignored, so a partially written or hand-edited header cannot leak a phantom entry.
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith(_ENTRY_HEADING_PREFIX):
            current = [line]
            blocks.append(current)
        elif current is not None:
            current.append(line)
    return blocks


def _parse_entry(block: list[str]) -> _Entry:
    """Parse one entry block into its channel, digest key, and item bullets."""
    entry = _Entry()
    current_kind: str | None = None
    pending: _Item | None = None
    for line in block:
        stripped = line.strip()
        if stripped in _SECTION_KINDS:
            current_kind = _SECTION_KINDS[stripped]
            pending = None
            continue
        channel_match = _CHANNEL_RE.match(line)
        if channel_match:
            entry.channel = channel_match.group(1)
            continue
        key_match = _DIGEST_KEY_RE.match(line)
        if key_match:
            entry.digest_key = key_match.group(1)
            continue
        # Top-level item bullets start at column 0 with "- "; rationale/sources sub-bullets
        # are indented and so are handled by the branch below.
        if current_kind is not None and line.startswith("- "):
            item_text = line[2:].strip()
            if item_text == _NONE_MARKER:
                pending = None
                continue
            pending = _Item(kind=current_kind, text=item_text)
            entry.items.append(pending)
            continue
        if pending is not None and stripped.startswith("- _Sources:_"):
            permalink_match = _PERMALINK_RE.search(line)
            if permalink_match:
                pending.permalink = permalink_match.group(1)
    return entry


def search_logs(
    root: str | os.PathLike[str],
    query: str,
    *,
    channel: str | None = None,
    kind: str | None = None,
) -> list[SearchHit]:
    """Search the committed decision logs under ``root`` for ``query``.

    Matching is a case-insensitive substring test over each extracted decision, action
    item, and open question. Results are returned in a stable order: channel files sorted
    by name, entries in document order, items in section order.

    Args:
        root: The repository root containing ``docs/decisions``.
        query: The substring to match (case-insensitive).
        channel: When set, restrict matching to entries whose channel id equals this
            value.
        kind: When set, restrict matching to one item category (``"decision"``,
            ``"action_item"``, or ``"open_question"``).

    Returns:
        A list of :class:`SearchHit`. Empty (never an error) when ``docs/decisions`` does
        not exist, so searching a fresh repository is a clean no-op.
    """
    decisions_path = Path(root) / _DECISIONS_DIR
    if not decisions_path.is_dir():
        return []

    needle = query.casefold()
    hits: list[SearchHit] = []
    for md_file in sorted(decisions_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for block in _split_entry_blocks(text):
            entry = _parse_entry(block)
            if channel is not None and entry.channel != channel:
                continue
            for item in entry.items:
                if kind is not None and item.kind != kind:
                    continue
                if needle in item.text.casefold():
                    hits.append(
                        SearchHit(
                            channel=entry.channel,
                            digest_key=entry.digest_key,
                            kind=item.kind,
                            line=item.text,
                            permalink=item.permalink,
                        )
                    )
    return hits


__all__ = ["SearchHit", "search_logs"]
