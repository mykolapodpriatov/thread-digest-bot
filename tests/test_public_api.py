"""Public API smoke test — ``import thread_digest_bot`` exposes the core surface."""

from __future__ import annotations

import thread_digest_bot as tdb


def test_version_exported() -> None:
    assert isinstance(tdb.__version__, str)
    assert tdb.__version__


def test_all_names_are_importable() -> None:
    # Every name in __all__ resolves to a real attribute.
    for name in tdb.__all__:
        assert hasattr(tdb, name), name


def test_end_to_end_via_public_api() -> None:
    thread = tdb.Thread(
        channel_id="c1",
        platform="telegram",
        messages=[
            tdb.Message(
                id="m1",
                author=tdb.Author(id="u1", display="Ada"),
                text="Let's ship the onboarding flow",
                ts_label="t1",
            )
        ],
    )
    log = tdb.digest(thread, tdb.FakeLLM("happy"))
    assert isinstance(log, tdb.DecisionLog)
    md = tdb.render_markdown_entry(log)
    assert md.startswith("## ")


def test_link_builders_exported() -> None:
    assert tdb.telegram_private_permalink(-1001234567890, 42) == "https://t.me/c/1234567890/42"
