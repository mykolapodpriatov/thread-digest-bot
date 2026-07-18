"""thread-digest-bot: turn sprawling chat threads into attributed decision logs.

The public API re-exported here covers the offline core: the domain types, the
digest/grounding pipeline, rendering, the append-only Git store, the deterministic
``FakeLLM`` and ``FakePlatform``, and the link builders. Optional provider/platform
adapters live behind extras and are imported from their own modules.

Example:
    >>> from thread_digest_bot import digest, ground, FakeLLM, Thread, Message, Author
    >>> thread = Thread(
    ...     channel_id="c1",
    ...     platform="telegram",
    ...     messages=[
    ...         Message(id="m1", author=Author(id="u1", display="Ada"),
    ...                 text="Let's ship Friday", ts_label="t0"),
    ...     ],
    ... )
    >>> log = digest(thread, FakeLLM("empty"))
    >>> log.is_empty()
    True
"""

from __future__ import annotations

from thread_digest_bot.digest import build_prompt, digest
from thread_digest_bot.grounding import GroundingPolicy, ground
from thread_digest_bot.ingest import (
    MessageInput,
    ThreadInput,
    thread_from_dict,
    thread_from_input,
    thread_from_json,
)
from thread_digest_bot.links import (
    discord_permalink,
    slack_archives_permalink,
    telegram_private_permalink,
    telegram_public_permalink,
)
from thread_digest_bot.llm import (
    LLMBackend,
    LLMError,
    RawActionItem,
    RawCitation,
    RawDecision,
    RawDecisionLog,
    RawOpenQuestion,
)
from thread_digest_bot.llm.fake import FakeLLM
from thread_digest_bot.platforms import (
    ChatPlatform,
    CommandContext,
    FakePlatform,
    FetchError,
    PlatformError,
    ThreadNotFoundError,
)
from thread_digest_bot.render import render_chat_reply, render_markdown_entry
from thread_digest_bot.rollup import build_rollup, rollup_label
from thread_digest_bot.schedule import FakeScheduler, IntervalScheduler, Scheduler
from thread_digest_bot.search import SearchHit, search_logs
from thread_digest_bot.service import DigestOutcome, DigestService
from thread_digest_bot.store import (
    AppendOnlyViolation,
    AppendResult,
    DecisionStore,
    OrphanStateError,
    StoreConfig,
    StoreError,
)
from thread_digest_bot.types import (
    ActionItem,
    Author,
    Citation,
    Decision,
    DecisionLog,
    Message,
    OpenQuestion,
    Platform,
    Thread,
    compute_digest_key,
    compute_rollup_key,
)

__version__ = "0.1.0"

__all__ = [
    "ActionItem",
    "AppendOnlyViolation",
    "AppendResult",
    "Author",
    "ChatPlatform",
    "Citation",
    "CommandContext",
    "Decision",
    "DecisionLog",
    "DecisionStore",
    "DigestOutcome",
    "DigestService",
    "FakeLLM",
    "FakePlatform",
    "FakeScheduler",
    "FetchError",
    "GroundingPolicy",
    "IntervalScheduler",
    "LLMBackend",
    "LLMError",
    "Message",
    "MessageInput",
    "OpenQuestion",
    "OrphanStateError",
    "Platform",
    "PlatformError",
    "RawActionItem",
    "RawCitation",
    "RawDecision",
    "RawDecisionLog",
    "RawOpenQuestion",
    "Scheduler",
    "SearchHit",
    "StoreConfig",
    "StoreError",
    "Thread",
    "ThreadInput",
    "ThreadNotFoundError",
    "__version__",
    "build_prompt",
    "build_rollup",
    "compute_digest_key",
    "compute_rollup_key",
    "digest",
    "discord_permalink",
    "ground",
    "render_chat_reply",
    "render_markdown_entry",
    "rollup_label",
    "search_logs",
    "slack_archives_permalink",
    "telegram_private_permalink",
    "telegram_public_permalink",
    "thread_from_dict",
    "thread_from_input",
    "thread_from_json",
]
