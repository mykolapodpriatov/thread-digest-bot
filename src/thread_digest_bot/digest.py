"""Thread -> DecisionLog extraction.

:func:`digest` renders a prompt embedding the numbered messages, asks the
:class:`~thread_digest_bot.llm.LLMBackend` for structured JSON conforming to
:class:`~thread_digest_bot.llm.RawDecisionLog`, then hands the result to
:func:`~thread_digest_bot.grounding.ground` for hard grounding.

The prompt is explicit that citations must reference the shown message ids and must
**not** include author or permalink — those are derived during grounding.
"""

from __future__ import annotations

from thread_digest_bot.grounding import GroundingPolicy, GroundingReport, ground_with_report
from thread_digest_bot.llm import LLMBackend, RawDecisionLog
from thread_digest_bot.types import DecisionLog, Thread

_SYSTEM_INSTRUCTIONS = """\
You extract a structured decision log from a chat thread.

You are given numbered messages, each with an id, an author display name, and text.
Produce JSON with three lists: "decisions", "action_items", "open_questions".

Rules:
- Cite evidence: every item must include a "citations" list referencing the message
  ids that support it. Use ONLY ids that appear in the thread below.
- A citation has exactly: "message_id" (required) and optionally "quote" (a short
  verbatim excerpt copied from that message's text). Do NOT include author names,
  permalinks, or any other field in a citation — those are added automatically.
- For action_items you may set "assignee" to the display name of the responsible
  person if it is clear from the thread; otherwise omit it.
- Only include a "quote" if you copy it verbatim from the cited message.
- If nothing of a given kind is present, return an empty list for it.
"""


def build_prompt(thread: Thread) -> str:
    """Render the extraction prompt for a thread.

    Args:
        thread: The normalized thread to digest.

    Returns:
        A complete prompt string embedding the numbered messages.
    """
    lines = [_SYSTEM_INSTRUCTIONS, "", "Thread messages:"]
    for i, message in enumerate(thread.messages, start=1):
        text = message.text.replace("\n", " ").strip()
        lines.append(f"{i}. [id={message.id}] {message.author.display}: {text}")
    if not thread.messages:
        lines.append("(the thread is empty)")
    if thread.truncated:
        lines.append("")
        lines.append("(note: the thread was truncated; earlier messages are not shown)")
    lines.append("")
    lines.append("Return only the JSON object.")
    return "\n".join(lines)


def digest_with_report(
    thread: Thread,
    llm: LLMBackend,
    *,
    range_label: str | None = None,
    policy: GroundingPolicy | None = None,
    digest_key: str | None = None,
) -> tuple[DecisionLog, GroundingReport]:
    """Digest a thread and also return the grounding drop report.

    Identical to :func:`digest`, but additionally returns the
    :class:`~thread_digest_bot.grounding.GroundingReport` for the single grounding pass —
    letting callers (e.g. ``digest-file --stats``) surface what was dropped without a
    second LLM call.

    Args:
        thread: The normalized thread to digest.
        llm: A structured-output backend implementing
            :class:`~thread_digest_bot.llm.LLMBackend`.
        range_label: Human-readable label for the digested range; defaults to a label
            derived from the message count.
        policy: Grounding policy override.
        digest_key: Optional pre-computed idempotency key (see :func:`digest`).

    Returns:
        A ``(DecisionLog, GroundingReport)`` pair.

    Raises:
        thread_digest_bot.llm.LLMError: If the backend cannot return schema-valid
            output after one bounded retry.
    """
    label = range_label or _default_range_label(thread)

    if not thread.messages:
        raw_log = RawDecisionLog()
    else:
        prompt = build_prompt(thread)
        raw_log = llm.complete_json(prompt, RawDecisionLog)

    return ground_with_report(
        raw_log, thread, range_label=label, policy=policy, digest_key=digest_key
    )


def digest(
    thread: Thread,
    llm: LLMBackend,
    *,
    range_label: str | None = None,
    policy: GroundingPolicy | None = None,
    digest_key: str | None = None,
) -> DecisionLog:
    """Digest a thread into a grounded :class:`DecisionLog`.

    For an empty thread the LLM is not invoked; an empty (but well-formed) log is
    returned so callers and the store behave uniformly. A thin wrapper over
    :func:`digest_with_report` that discards the drop report.

    Args:
        thread: The normalized thread to digest.
        llm: A structured-output backend implementing
            :class:`~thread_digest_bot.llm.LLMBackend`.
        range_label: Human-readable label for the digested range; defaults to a label
            derived from the message count.
        policy: Grounding policy override.
        digest_key: Optional pre-computed idempotency key. Defaults to the message-set
            key; rollups supply a period-scoped key so distinct periods over the same
            messages are not treated as duplicates.

    Returns:
        A grounded decision log with a deterministic ``digest_key``.

    Raises:
        thread_digest_bot.llm.LLMError: If the backend cannot return schema-valid
            output after one bounded retry.
    """
    log, _report = digest_with_report(
        thread, llm, range_label=range_label, policy=policy, digest_key=digest_key
    )
    return log


def _default_range_label(thread: Thread) -> str:
    count = len(thread.messages)
    suffix = " (truncated)" if thread.truncated else ""
    if count == 0:
        return "empty thread"
    if count == 1:
        return f"1 message{suffix}"
    return f"last {count} messages{suffix}"
