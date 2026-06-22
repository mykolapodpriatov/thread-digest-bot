"""Digest-key invariants: period-scoped rollup keys and the NUL-separator guard."""

from __future__ import annotations

import pytest

from thread_digest_bot.types import compute_digest_key, compute_rollup_key


def test_rollup_key_is_period_scoped_not_message_scoped() -> None:
    # A rollup's identity is (channel, period) — independent of any message set — so a
    # new period over the same messages is NOT deduped against a prior digest.
    week25 = compute_rollup_key("team-eng", "2026-W25")
    week26 = compute_rollup_key("team-eng", "2026-W26")
    assert week25 != week26  # distinct periods -> distinct keys (both committable)
    assert compute_rollup_key("team-eng", "2026-W25") == week25  # same period -> no-op


def test_rollup_key_distinct_from_message_set_digest_key() -> None:
    # The rollup namespace ("rollup") must never collide with a per-digest message-set
    # key for the same channel, even when the message ids would hash into the payload.
    msg_key = compute_digest_key("team-eng", "telegram", ["m1", "m2", "m3"])
    rollup = compute_rollup_key("team-eng", "2026-W25")
    assert msg_key != rollup


def test_rollup_key_scoped_per_channel() -> None:
    assert compute_rollup_key("team-eng", "2026-W25") != compute_rollup_key("team-ops", "2026-W25")


def test_compute_digest_key_rejects_nul_in_field() -> None:
    # The \x00 field separator is unambiguous only if no field value contains it;
    # a field carrying a literal NUL must fail loudly rather than alias inputs.
    with pytest.raises(ValueError, match="NUL"):
        compute_digest_key("team-eng", "telegram", ["m1", "bad\x00id"])
    with pytest.raises(ValueError, match="NUL"):
        compute_digest_key("chan\x00nel", "telegram", ["m1"])


def test_compute_rollup_key_rejects_nul_in_field() -> None:
    with pytest.raises(ValueError, match="NUL"):
        compute_rollup_key("team-eng", "2026\x00W25")
    with pytest.raises(ValueError, match="NUL"):
        compute_rollup_key("team\x00eng", "2026-W25")


def test_nul_guard_blocks_the_collision_it_documents() -> None:
    # Without the guard these two distinct inputs would build the same NUL-joined
    # payload ("a\x00b\x00c"); the guard makes both raise instead of silently aliasing.
    with pytest.raises(ValueError):
        compute_digest_key("a\x00b", "c", [])
    with pytest.raises(ValueError):
        compute_digest_key("a", "b\x00c", [])
