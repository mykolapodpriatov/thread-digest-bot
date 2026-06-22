"""Rendering — ``DecisionLog`` -> Markdown log entry and -> chat reply.

Two surfaces:

* :func:`render_markdown_entry` produces the durable, append-only Markdown block
  committed to Git. Truncation is always noted so a partial digest is never presented
  as complete.
* :func:`render_chat_reply` produces a compact, platform-safe plain-text reply.

Both are pure functions of the log; no clock or network access.
"""

from __future__ import annotations

from thread_digest_bot.types import Citation, DecisionLog


def _format_citation(citation: Citation) -> str:
    """Render a single citation as a Markdown fragment with optional deep link."""
    label = citation.author.display
    ref = f"[{label}]({citation.permalink})" if citation.permalink else label
    if citation.quote:
        return f'{ref}: "{citation.quote}"'
    return ref


def _format_citations(citations: list[Citation]) -> str:
    """Render a citation list, or an explicit unsourced marker when empty."""
    if not citations:
        return "_unsourced_"
    return "; ".join(_format_citation(c) for c in citations)


def render_markdown_entry(log: DecisionLog) -> str:
    """Render a ``DecisionLog`` as an append-only Markdown entry.

    Args:
        log: The grounded decision log.

    Returns:
        A Markdown string beginning with an ``##`` header and ending with a single
        trailing newline, suitable for appending to a per-channel log file.
    """
    lines: list[str] = []
    lines.append(f"## {log.range_label}")
    lines.append("")
    lines.append(f"- **Channel:** `{log.channel_id}`")
    participants = ", ".join(p.display for p in log.participants) or "_none_"
    lines.append(f"- **Participants:** {participants}")
    lines.append(f"- **Digest key:** `{log.digest_key}`")
    if log.truncated:
        lines.append("- **Note:** thread was truncated; this digest may be incomplete.")
    lines.append("")

    lines.append("### Decisions")
    if log.decisions:
        for decision in log.decisions:
            lines.append(f"- {decision.statement}")
            if decision.rationale:
                lines.append(f"  - _Rationale:_ {decision.rationale}")
            lines.append(f"  - _Sources:_ {_format_citations(decision.citations)}")
    else:
        lines.append("- _none_")
    lines.append("")

    lines.append("### Action items")
    if log.action_items:
        for item in log.action_items:
            assignee = f" — **{item.assignee.display}**" if item.assignee else ""
            lines.append(f"- {item.task}{assignee}")
            lines.append(f"  - _Sources:_ {_format_citations(item.citations)}")
    else:
        lines.append("- _none_")
    lines.append("")

    lines.append("### Open questions")
    if log.open_questions:
        for question in log.open_questions:
            lines.append(f"- {question.question}")
            lines.append(f"  - _Sources:_ {_format_citations(question.citations)}")
    else:
        lines.append("- _none_")
    lines.append("")

    return "\n".join(lines) + "\n"


def render_chat_reply(log: DecisionLog) -> str:
    """Render a compact, platform-safe chat reply for a ``DecisionLog``.

    Args:
        log: The grounded decision log.

    Returns:
        A short plain-text summary safe to post to Telegram or Slack.
    """
    lines: list[str] = [f"Digest — {log.range_label}"]
    if log.truncated:
        lines.append("(thread truncated; digest may be incomplete)")

    if log.decisions:
        lines.append("")
        lines.append("Decisions:")
        for decision in log.decisions:
            lines.append(f"• {decision.statement}")
    if log.action_items:
        lines.append("")
        lines.append("Action items:")
        for item in log.action_items:
            who = f" ({item.assignee.display})" if item.assignee else ""
            lines.append(f"• {item.task}{who}")
    if log.open_questions:
        lines.append("")
        lines.append("Open questions:")
        for question in log.open_questions:
            lines.append(f"• {question.question}")

    if log.is_empty():
        lines.append("")
        lines.append("No decisions, action items, or open questions found.")

    return "\n".join(lines)
