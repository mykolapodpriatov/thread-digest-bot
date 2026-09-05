# thread-digest-bot

> Turns sprawling Telegram/Slack threads into structured, attributed decision logs and action items — committed to Git as an append-only audit trail.

![status](https://img.shields.io/badge/status-active%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue) ![license](https://img.shields.io/badge/license-MIT-green)

A Telegram/Slack bot that, on `/digest` or a schedule, converts the last N messages or a replied-to thread into a structured decision log: who decided what, the rationale, open questions, and action items with assignees — each claim carrying speaker attribution and a deep-linked message backreference, and each digest appended to a per-channel Markdown log committed to Git.

## Why

Decisions made in chat evaporate. This captures them as a durable, attributed, searchable record your team actually keeps — and, crucially, **never invents an attribution**: every cited source is a real message, resolved by id, with its author and permalink taken from the message itself.

## Highlights

- **Grounded by construction.** Every extracted item must cite a real message id. Hallucinated ids are dropped, an item left with no valid citation is dropped, the author/permalink are populated from the real message (never trusted from the model), and a quote that is not actually present in the message text is rejected.
- **Append-only audit trail.** Each digest is appended to `docs/decisions/<channel>.md` and committed to Git. Appends are prefix-preserving (history can't be silently rewritten), idempotent on a deterministic `digest_key` (replays are no-ops), and orphan-aware (a write that died before its commit is recovered, never appended over).
- **Correct deep links or none.** Permalink builders are pure functions; an unknown shape yields `None` rather than a plausible-but-wrong link.
- **Pluggable LLM backend.** One `complete_json(prompt, schema)` interface over OpenAI, Anthropic, or local Ollama, with a single bounded retry on malformed output (never an infinite loop). A deterministic `FakeLLM` makes the whole pipeline testable offline with zero network.
- **Platform-agnostic core.** A `ChatPlatform` protocol keeps Telegram/Slack specifics at the edges; a `FakePlatform` proves the protocol and drives the end-to-end tests offline.

## Install

```bash
pip install thread-digest-bot                 # core (Fake + Ollama backends, CLI)
pip install "thread-digest-bot[openai]"       # OpenAI backend
pip install "thread-digest-bot[anthropic]"    # Anthropic backend
pip install "thread-digest-bot[telegram]"     # Telegram adapter
pip install "thread-digest-bot[slack]"        # Slack adapter
```

Requires Python 3.11+.

## Quick start (offline, no API key)

Digest the bundled example thread into a decision log:

```bash
thread-digest-bot digest-file examples/thread.json
```

Commit it into a Git repository as an append-only audit entry:

```bash
thread-digest-bot digest-file examples/thread.json --commit --repo-root .
# writes & commits docs/decisions/team-eng.md
```

Or run the self-contained demo, which digests `examples/thread.json` into a throwaway Git
repo with the deterministic `FakeLLM`, prints the commit log and the rendered Markdown,
and shows that a re-append is an idempotent no-op:

```bash
python examples/demo.py
```

### As a library

```python
from thread_digest_bot import digest, FakeLLM, thread_from_json, DecisionStore, StoreConfig

thread = thread_from_json(open("examples/thread.json").read())
log = digest(thread, FakeLLM("happy"))  # swap in an OpenAI/Anthropic/Ollama backend
DecisionStore(".", config=StoreConfig(commit=True)).append(log)
```

## CLI

| Command | Description |
| --- | --- |
| `digest-file THREAD.json [--out FILE] [--commit] [--repo-root DIR] [--config TOML] [--fixture NAME] [--range-label LABEL]` | Offline digest of a validated thread JSON into an attributed log. |
| `search QUERY [--repo-root DIR] [--channel ID] [--kind decision\|action_item\|open_question] [--format term\|json] [--config TOML]` | Case-insensitive substring search over the committed decision logs. |
| `rollup --channel ID --period weekly --period-key 2026-W25` | Build the label/identity for a periodic rollup. |
| `run --config config.toml` | Start the configured bot(s) (live adapters land in M3). |

### `thread.json` schema

Validated before digesting; a missing field or unknown shape fails with a clear message and a non-zero exit code.

```json
{
  "channel_id": "team-eng",
  "platform": "telegram",
  "messages": [
    {
      "id": "m1",
      "author": { "id": "u_ada", "display": "Ada" },
      "text": "Let's ship the onboarding flow on Friday once QA signs off.",
      "ts_label": "Mon 09:14",
      "permalink": "https://t.me/c/1234567890/1"
    }
  ]
}
```

Messages are taken in array order (caller-ordered).

## How it works

```
fetch ──▶ digest (LLM → structured JSON) ──▶ ground (drop/validate citations)
      ──▶ render (Markdown entry + chat reply) ──▶ store (append-only Git commit)
```

The bot reads the requested message range, asks the LLM for a structured decision log
whose citations contain only message ids (never author/permalink), grounds every
citation against the real thread, posts a compact reply to the channel, and appends the
same entry to a Markdown file committed to a configured Git repo. Scheduled rollups run
the identical pipeline on a cadence and dedup on their period.

## Architecture

| Module | Responsibility |
| --- | --- |
| `types` | Domain models (`Message`, `Thread`, `DecisionLog`, `Citation`, …); clock-free. |
| `ingest` | Normalize platform payloads / `thread.json` into a `Thread`. |
| `digest` | Build the prompt and call the LLM for structured JSON. |
| `grounding` | The correctness core: validate/enrich/drop citations and items. |
| `llm/` | `LLMBackend` protocol + `FakeLLM`, OpenAI, Anthropic, Ollama adapters. |
| `links` | Pure permalink builders (Telegram private/public, Slack archives). |
| `render` | `DecisionLog` → Markdown entry and → chat reply. |
| `store` | Append-only, idempotent, orphan-aware Git store (+ webhook export). |
| `search` | Read-side substring search over the committed per-channel logs. |
| `platforms/` | `ChatPlatform` protocol + `FakePlatform`; thin Telegram/Slack adapters. |
| `schedule` | `Scheduler` protocol + interval and fake schedulers. |
| `rollup` / `service` | Periodic rollups and the fetch→digest→post→store wiring. |
| `config` / `cli` | Validated TOML config and the Typer CLI. |

## Status & roadmap

**Active development.** Milestones M1–M2 are implemented and tested offline; M3–M4 are designed and proven by the `FakePlatform`.

- [x] Thread → grounded, attributed decision log (with `FakeLLM`, fully offline)
- [x] Append-only Markdown log committed to Git (idempotent, orphan-aware)
- [x] Telegram/Slack deep-link builders; `ChatPlatform` protocol + `/digest` flow
- [x] OpenAI / Anthropic / Ollama backends behind one interface
- [x] Search over the committed decision logs (`search` command)
- [ ] Live Telegram + Slack adapters; scheduled rollups in production (M3)
- [x] Webhook export (`HttpWebhookSink`)
- [ ] Discord adapter (M4)

## Webhook export

A digest lands as an append-only Markdown entry in Git. That is the right home
for it, and it used to be the only one, so nothing downstream could react to a
decision being recorded.

```python
from thread_digest_bot import DecisionStore, HttpWebhookSink

store = DecisionStore(
    repo_root,
    webhook=HttpWebhookSink(
        urls=["https://hooks.example/decisions", os.environ["SLACK_WEBHOOK_URL"]],
        secret=os.environ["DIGEST_WEBHOOK_SECRET"],
    ),
)
```

The payload is the structured facts plus the commit SHA and the path, not the
rendered Markdown: a consumer that wants Markdown can render it, one that wants
fields cannot un-render it.

Three rules, and they are why this is a module rather than a `post()` in the
store:

- **Delivery happens after the commit, never before.** A webhook that fires for
  a digest that then fails to commit reports a decision that is not recorded
  anywhere. The delivery carries the SHA of the commit it is reporting.
- **A failed delivery never fails the digest.** Git is the source of truth and a
  flaky endpoint cannot be allowed to take it down. Failures are retried with
  exponential backoff a bounded number of times, then logged and dropped. A sink
  that raises is caught too.
- **Redelivery is safe.** Every delivery carries a stable id derived from the
  channel and the digest key, so a receiver can deduplicate a retry rather than
  recording the same decision twice.

With `secret` set, each request carries an HMAC-SHA256 of the exact body it
sends in `X-Digest-Signature`, so a receiver can tell a real delivery from
anything else that finds the URL. The secret comes from the environment, never
the config file, matching how LLM keys are handled here. Neither the secret nor
a URL query string appears in the payload or in a log line.

Several endpoints can be fed from one pipeline; one failing never stops the
others. `CollectingWebhookSink` is the in-memory sink for tests and demos.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check && ruff format --check
mypy src
pytest -q --cov=thread_digest_bot --cov-report=term-missing
```

Everything is deterministic and offline: `FakeLLM` + `FakePlatform` + a temporary Git repo exercise the full flow with zero network. CI runs lint, format, strict mypy, and the test suite on Python 3.11–3.13.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
