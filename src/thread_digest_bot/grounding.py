"""Citation grounding — the correctness backbone.

A tool whose value is *trustworthy attribution* must never present an invented
source. :func:`ground` takes the raw, LLM-proposed log and the real thread and:

1. Drops any citation whose ``message_id`` does not resolve to a real message
   (hallucinated ids cannot survive).
2. For surviving citations, fills ``author`` and ``permalink`` **from the real
   message** — an LLM-supplied author is never trusted.
3. Validates the candidate ``quote``: it must be a whitespace-normalized substring
   of the real message text, otherwise it is dropped (default) or replaced with the
   message's leading text. A fabricated quote can therefore never ride on a valid id.
4. De-duplicates citations by ``(message_id, quote)``.
5. Drops any item left with zero valid citations (default) or keeps it flagged as
   unsourced (configurable).

Author/permalink provenance and quote validation make the resulting
:class:`~thread_digest_bot.types.DecisionLog` trustworthy by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from thread_digest_bot.llm import (
    RawActionItem,
    RawCitation,
    RawDecision,
    RawDecisionLog,
    RawOpenQuestion,
)
from thread_digest_bot.types import (
    ActionItem,
    Author,
    Citation,
    Decision,
    DecisionLog,
    Message,
    OpenQuestion,
    Thread,
    compute_digest_key,
)


@dataclass(frozen=True)
class GroundingPolicy:
    """How grounding treats invalid citations and quotes.

    Attributes:
        drop_zero_citation_items: When ``True`` (default) an item with no valid
            citations is removed. When ``False`` it is kept (flagged as unsourced by
            the renderer via empty citations).
        replace_invalid_quote_with_leading_text: When ``True`` a quote that is not a
            substring of the real message is replaced by the message's leading text
            instead of being set to ``None``.
        leading_text_chars: How many characters of leading text to use for the
            replacement above.
    """

    drop_zero_citation_items: bool = True
    replace_invalid_quote_with_leading_text: bool = False
    leading_text_chars: int = 120


def _normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip the ends."""
    return " ".join(text.split())


def _validate_quote(
    candidate: str | None,
    message: Message,
    policy: GroundingPolicy,
) -> str | None:
    """Return a trustworthy quote for ``message`` given the LLM candidate.

    The candidate is accepted only if its whitespace-normalized form is a substring
    of the message's whitespace-normalized text. Otherwise it is replaced with the
    leading text or dropped, per ``policy``.
    """
    if candidate is None:
        return None
    norm_candidate = _normalize_ws(candidate)
    if not norm_candidate:
        return None
    norm_text = _normalize_ws(message.text)
    if norm_candidate in norm_text:
        return norm_candidate
    if policy.replace_invalid_quote_with_leading_text:
        leading = norm_text[: policy.leading_text_chars].strip()
        return leading or None
    return None


def _ground_citations(
    raw_citations: list[RawCitation],
    index: dict[str, Message],
    policy: GroundingPolicy,
) -> list[Citation]:
    """Resolve raw citations against real messages, dropping the unresolvable."""
    grounded: list[Citation] = []
    seen: set[tuple[str, str | None]] = set()
    for raw in raw_citations:
        message = index.get(raw.message_id)
        if message is None:
            # Hallucinated / unresolved id — drop it entirely.
            continue
        quote = _validate_quote(raw.quote, message, policy)
        dedup_key = (message.id, quote)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        grounded.append(
            Citation(
                message_id=message.id,
                author=message.author,  # provenance: from the real message only
                permalink=message.permalink,  # provenance: from the real message only
                quote=quote,
            )
        )
    return grounded


def _resolve_assignee(
    raw_assignee: str | None,
    citations: list[Citation],
    participants: dict[str, Author],
) -> Author | None:
    """Resolve an LLM-proposed assignee display name to a real participant.

    Resolution is attempted against the actual thread participants by case-insensitive
    display match. If no participant matches, the assignee falls back to the author of
    the first grounded citation (a real, attributable person), else ``None``. The LLM
    string alone never becomes an authoritative :class:`Author`.
    """
    if raw_assignee:
        wanted = raw_assignee.strip().casefold()
        for author in participants.values():
            if author.display.casefold() == wanted:
                return author
    if citations:
        return citations[0].author
    return None


def _ground_decisions(
    raws: list[RawDecision],
    index: dict[str, Message],
    policy: GroundingPolicy,
) -> list[Decision]:
    out: list[Decision] = []
    for raw in raws:
        citations = _ground_citations(raw.citations, index, policy)
        if not citations and policy.drop_zero_citation_items:
            continue
        out.append(Decision(statement=raw.statement, rationale=raw.rationale, citations=citations))
    return out


def _ground_action_items(
    raws: list[RawActionItem],
    index: dict[str, Message],
    participants: dict[str, Author],
    policy: GroundingPolicy,
) -> list[ActionItem]:
    out: list[ActionItem] = []
    for raw in raws:
        citations = _ground_citations(raw.citations, index, policy)
        if not citations and policy.drop_zero_citation_items:
            continue
        assignee = _resolve_assignee(raw.assignee, citations, participants)
        out.append(ActionItem(task=raw.task, assignee=assignee, citations=citations))
    return out


def _ground_open_questions(
    raws: list[RawOpenQuestion],
    index: dict[str, Message],
    policy: GroundingPolicy,
) -> list[OpenQuestion]:
    out: list[OpenQuestion] = []
    for raw in raws:
        citations = _ground_citations(raw.citations, index, policy)
        if not citations and policy.drop_zero_citation_items:
            continue
        out.append(OpenQuestion(question=raw.question, citations=citations))
    return out


def ground(
    raw_log: RawDecisionLog,
    thread: Thread,
    *,
    range_label: str,
    policy: GroundingPolicy | None = None,
    digest_key: str | None = None,
) -> DecisionLog:
    """Ground a raw LLM log against a real thread into a trustworthy ``DecisionLog``.

    Args:
        raw_log: The structured output proposed by the LLM.
        thread: The real, normalized thread the digest is about.
        range_label: Human-readable label for the digested range (e.g. ``"last 200"``).
        policy: Grounding policy; defaults to dropping invalid quotes and
            zero-citation items.
        digest_key: Optional pre-computed idempotency key to stamp on the log. When
            ``None`` (the default, used by on-demand digests) the key is derived from
            the exact message set; rollups pass a period-scoped key instead so a new
            period over the same messages is not deduped against a prior digest.

    Returns:
        A :class:`DecisionLog` in which every citation references a real message and
        carries real author/permalink provenance, and ``digest_key`` identifies the
        exact message set (or the supplied period scope).
    """
    policy = policy or GroundingPolicy()
    index = thread.index_by_id()
    participants = {a.id: a for a in thread.participants()}

    decisions = _ground_decisions(raw_log.decisions, index, policy)
    action_items = _ground_action_items(raw_log.action_items, index, participants, policy)
    open_questions = _ground_open_questions(raw_log.open_questions, index, policy)

    key = digest_key or compute_digest_key(
        thread.channel_id, thread.platform, [m.id for m in thread.messages]
    )

    return DecisionLog(
        channel_id=thread.channel_id,
        range_label=range_label,
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        participants=list(participants.values()),
        digest_key=key,
        truncated=thread.truncated,
    )
