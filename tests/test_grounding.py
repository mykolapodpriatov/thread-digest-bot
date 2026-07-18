"""Grounding tests — the correctness backbone.

Every citation must resolve to a real message, carry author/permalink provenance from
that message, and only keep a quote that actually appears in the text. Items left with
no valid citation are dropped (or flagged, configurably).
"""

from __future__ import annotations

import pytest

from thread_digest_bot.grounding import GroundingPolicy, ground, ground_with_report
from thread_digest_bot.llm import (
    RawActionItem,
    RawCitation,
    RawDecision,
    RawDecisionLog,
    RawOpenQuestion,
)
from thread_digest_bot.types import Author, Message, Thread


def _thread() -> Thread:
    ada = Author(id="u_ada", display="Ada")
    bob = Author(id="u_bob", display="Bob")
    return Thread(
        channel_id="c1",
        platform="telegram",
        messages=[
            Message(
                id="m1",
                author=ada,
                text="Let's ship the onboarding flow on Friday.",
                ts_label="t1",
                permalink="https://t.me/c/1/1",
            ),
            Message(
                id="m2",
                author=bob,
                text="I'll write the release notes.",
                ts_label="t2",
                permalink="https://t.me/c/1/2",
            ),
        ],
    )


def test_hallucinated_id_is_dropped() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship Friday",
                citations=[
                    RawCitation(message_id="m1"),
                    RawCitation(message_id="ghost-id"),
                ],
            )
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    assert len(log.decisions) == 1
    citation_ids = [c.message_id for c in log.decisions[0].citations]
    assert citation_ids == ["m1"]  # the hallucinated id is gone


def test_zero_citation_item_dropped_by_default() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(statement="Real", citations=[RawCitation(message_id="m1")]),
            RawDecision(statement="Unsourced", citations=[RawCitation(message_id="ghost")]),
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    statements = [d.statement for d in log.decisions]
    assert statements == ["Real"]


def test_zero_citation_item_kept_when_flagged() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(statement="Unsourced", citations=[RawCitation(message_id="ghost")]),
        ]
    )
    policy = GroundingPolicy(drop_zero_citation_items=False)
    log = ground(raw, _thread(), range_label="r", policy=policy)
    assert len(log.decisions) == 1
    assert log.decisions[0].citations == []  # kept, flagged as unsourced


def test_valid_citation_enriched_with_real_author_and_permalink() -> None:
    raw = RawDecisionLog(
        decisions=[RawDecision(statement="Ship", citations=[RawCitation(message_id="m1")])]
    )
    log = ground(raw, _thread(), range_label="r")
    citation = log.decisions[0].citations[0]
    assert citation.author == Author(id="u_ada", display="Ada")
    assert citation.permalink == "https://t.me/c/1/1"


def test_llm_supplied_assignee_resolved_to_real_participant() -> None:
    # The model's free-text assignee is matched to a real participant by display name;
    # an unknown name never becomes an authoritative Author.
    raw = RawDecisionLog(
        action_items=[
            RawActionItem(
                task="Write notes",
                assignee="bob",  # different case than the real display "Bob"
                citations=[RawCitation(message_id="m2")],
            )
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    item = log.action_items[0]
    assert item.assignee == Author(id="u_bob", display="Bob")


def test_unknown_assignee_falls_back_to_citation_author() -> None:
    raw = RawDecisionLog(
        action_items=[
            RawActionItem(
                task="Write notes",
                assignee="Somebody Not In The Thread",
                citations=[RawCitation(message_id="m2")],
            )
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    # Falls back to the real author of the first grounded citation, never the raw string.
    assert log.action_items[0].assignee == Author(id="u_bob", display="Bob")


def test_fabricated_quote_dropped_by_default() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship",
                citations=[
                    RawCitation(
                        message_id="m1",
                        quote="a sentence nobody actually wrote in the thread",
                    )
                ],
            )
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    assert log.decisions[0].citations[0].quote is None


def test_real_quote_is_kept_normalized() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship",
                # Extra/odd whitespace still matches after normalization.
                citations=[RawCitation(message_id="m1", quote="ship   the\nonboarding flow")],
            )
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    assert log.decisions[0].citations[0].quote == "ship the onboarding flow"


def test_fabricated_quote_replaced_with_leading_text_when_configured() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship",
                citations=[RawCitation(message_id="m1", quote="totally made up")],
            )
        ]
    )
    policy = GroundingPolicy(replace_invalid_quote_with_leading_text=True, leading_text_chars=20)
    log = ground(raw, _thread(), range_label="r", policy=policy)
    quote = log.decisions[0].citations[0].quote
    assert quote is not None
    assert "Let's ship the" in quote


def test_duplicate_citations_deduped() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship",
                citations=[
                    RawCitation(message_id="m1", quote="ship the onboarding flow"),
                    RawCitation(message_id="m1", quote="ship the onboarding flow"),
                    RawCitation(message_id="m1", quote="ship the onboarding flow"),
                ],
            )
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    assert len(log.decisions[0].citations) == 1


def test_open_questions_and_action_items_grounded() -> None:
    raw = RawDecisionLog(
        action_items=[
            RawActionItem(task="Notes", citations=[RawCitation(message_id="m2")]),
            RawActionItem(task="Ghost", citations=[RawCitation(message_id="nope")]),
        ],
        open_questions=[
            RawOpenQuestion(question="Flag?", citations=[RawCitation(message_id="m1")]),
            RawOpenQuestion(question="Ghost?", citations=[RawCitation(message_id="nope")]),
        ],
    )
    log = ground(raw, _thread(), range_label="r")
    assert [a.task for a in log.action_items] == ["Notes"]
    assert [q.question for q in log.open_questions] == ["Flag?"]


def test_digest_key_and_participants_populated() -> None:
    raw = RawDecisionLog()
    log = ground(raw, _thread(), range_label="r")
    assert log.digest_key  # non-empty deterministic key
    assert [p.display for p in log.participants] == ["Ada", "Bob"]
    assert log.channel_id == "c1"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_quote_becomes_none(blank: str) -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(statement="Ship", citations=[RawCitation(message_id="m1", quote=blank)])
        ]
    )
    log = ground(raw, _thread(), range_label="r")
    assert log.decisions[0].citations[0].quote is None


def test_grounding_report_counts_hallucination_invalid_quote_and_zero_item() -> None:
    # A raw log that trips every drop path: a hallucinated id (which also leaves its item
    # zero-citation), plus a valid id carrying a non-substring quote.
    raw = RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship",
                citations=[
                    RawCitation(message_id="m1", quote="a quote that was never written"),
                ],
            ),
            RawDecision(
                statement="Ghost decision",
                citations=[RawCitation(message_id="ghost-id")],
            ),
        ]
    )
    log, report = ground_with_report(raw, _thread(), range_label="r")

    assert report.dropped_hallucinated_citations == 1
    assert report.dropped_invalid_quotes == 1
    assert report.dropped_zero_citation_items == 1
    assert report.total_dropped == 3
    assert not report.is_clean()
    # The surviving decision kept its (valid) citation but nulled the invalid quote.
    assert [d.statement for d in log.decisions] == ["Ship"]
    assert log.decisions[0].citations[0].quote is None


def test_grounding_report_clean_for_fully_valid_log() -> None:
    raw = RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship",
                citations=[RawCitation(message_id="m1", quote="ship the onboarding flow")],
            )
        ]
    )
    _log, report = ground_with_report(raw, _thread(), range_label="r")
    assert report.is_clean()
    assert report.total_dropped == 0


def test_ground_matches_ground_with_report_log() -> None:
    # The backward-compatible ground() returns exactly the log ground_with_report() does.
    raw = RawDecisionLog(
        decisions=[RawDecision(statement="Ship", citations=[RawCitation(message_id="m1")])]
    )
    only_log = ground(raw, _thread(), range_label="r")
    log, _report = ground_with_report(raw, _thread(), range_label="r")
    assert only_log == log
