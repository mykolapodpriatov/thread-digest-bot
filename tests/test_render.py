"""Render tests — Markdown entry, chat reply, and JSON shapes; deep links, truncation."""

from __future__ import annotations

import json

from thread_digest_bot.render import (
    render_chat_reply,
    render_json_entry,
    render_markdown_entry,
)
from thread_digest_bot.types import (
    ActionItem,
    Author,
    Citation,
    Decision,
    DecisionLog,
    OpenQuestion,
)

ADA = Author(id="u_ada", display="Ada")
BOB = Author(id="u_bob", display="Bob")


def _full_log(*, truncated: bool = False) -> DecisionLog:
    return DecisionLog(
        channel_id="team-eng",
        range_label="last 3 messages",
        decisions=[
            Decision(
                statement="Ship Friday",
                rationale="QA passed",
                citations=[
                    Citation(
                        message_id="m1",
                        author=ADA,
                        permalink="https://t.me/c/1/1",
                        quote="ship Friday",
                    )
                ],
            )
        ],
        action_items=[
            ActionItem(
                task="Write release notes",
                assignee=BOB,
                citations=[Citation(message_id="m2", author=BOB, permalink="https://t.me/c/1/2")],
            ),
            ActionItem(
                task="Unassigned task",
                assignee=None,
                citations=[Citation(message_id="m1", author=ADA)],
            ),
        ],
        open_questions=[
            OpenQuestion(
                question="Feature flag?",
                citations=[Citation(message_id="m3", author=ADA)],
            )
        ],
        participants=[ADA, BOB],
        digest_key="abc123",
        truncated=truncated,
    )


def test_markdown_entry_structure_and_links() -> None:
    md = render_markdown_entry(_full_log())
    assert md.startswith("## last 3 messages\n")
    assert md.endswith("\n")
    assert "- **Channel:** `team-eng`" in md
    assert "- **Participants:** Ada, Bob" in md
    assert "- **Digest key:** `abc123`" in md
    # Deep link rendered as a Markdown link with the quote.
    assert '[Ada](https://t.me/c/1/1): "ship Friday"' in md
    assert "_Rationale:_ QA passed" in md


def test_markdown_assignee_rendered_and_optional() -> None:
    md = render_markdown_entry(_full_log())
    assert "- Write release notes — **Bob**" in md
    assert "- Unassigned task\n" in md  # no assignee suffix


def test_markdown_notes_truncation() -> None:
    md = render_markdown_entry(_full_log(truncated=True))
    assert "truncated" in md.lower()
    assert "may be incomplete" in md


def test_markdown_unsourced_marker_for_empty_citations() -> None:
    log = DecisionLog(
        channel_id="c",
        range_label="r",
        decisions=[Decision(statement="Flagged", citations=[])],
        digest_key="k",
    )
    md = render_markdown_entry(log)
    assert "_unsourced_" in md


def test_markdown_none_sections() -> None:
    log = DecisionLog(channel_id="c", range_label="r", digest_key="k")
    md = render_markdown_entry(log)
    # Each empty section renders an explicit "_none_".
    assert md.count("- _none_") == 3
    assert "_none_" in md  # participants line too


def test_chat_reply_shape() -> None:
    reply = render_chat_reply(_full_log())
    assert reply.startswith("Digest — last 3 messages")
    assert "Decisions:" in reply
    assert "• Ship Friday" in reply
    assert "• Write release notes (Bob)" in reply
    assert "• Unassigned task" in reply
    assert "Open questions:" in reply


def test_chat_reply_truncation_note() -> None:
    reply = render_chat_reply(_full_log(truncated=True))
    assert "truncated" in reply.lower()


def test_chat_reply_empty_log() -> None:
    log = DecisionLog(channel_id="c", range_label="r", digest_key="k")
    reply = render_chat_reply(log)
    assert "No decisions, action items, or open questions found." in reply


def test_json_entry_serializes_full_log() -> None:
    payload = json.loads(render_json_entry(_full_log()))

    assert payload["channel_id"] == "team-eng"
    assert payload["digest_key"] == "abc123"
    assert payload["range_label"] == "last 3 messages"
    assert payload["truncated"] is False
    assert payload["participants"] == ["Ada", "Bob"]

    decision = payload["decisions"][0]
    assert decision["statement"] == "Ship Friday"
    assert decision["rationale"] == "QA passed"
    citation = decision["citations"][0]
    # Provenance fields, author as display only (no id leaked).
    assert citation == {
        "author": "Ada",
        "permalink": "https://t.me/c/1/1",
        "quote": "ship Friday",
    }

    action = payload["action_items"][0]
    assert action["task"] == "Write release notes"
    assert action["assignee"] == "Bob"
    # An unassigned item serializes assignee as null.
    assert payload["action_items"][1]["assignee"] is None

    question = payload["open_questions"][0]
    assert question["question"] == "Feature flag?"


def test_json_entry_truncation_flag() -> None:
    payload = json.loads(render_json_entry(_full_log(truncated=True)))
    assert payload["truncated"] is True


def test_json_entry_empty_log() -> None:
    log = DecisionLog(channel_id="c", range_label="r", digest_key="k")
    payload = json.loads(render_json_entry(log))
    assert payload["decisions"] == []
    assert payload["action_items"] == []
    assert payload["open_questions"] == []
    assert payload["participants"] == []
