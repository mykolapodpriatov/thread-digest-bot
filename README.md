# thread-digest-bot

> Turns sprawling Telegram/Slack threads into structured, attributed decision logs and action items — committed to Git as an append-only audit trail.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

A Telegram/Slack bot that, on `/digest` or a schedule, converts the last N messages or a replied-to thread into a structured decision log: who decided what, the rationale, open questions, and action items with assignees — each claim carrying speaker attribution and a deep-linked message backreference.

## Why

Decisions made in chat evaporate. This captures them as a durable, attributed, searchable record your team actually keeps.

## Features

- `/digest` summarizes into decisions, rationale, and action items with assignees
- Per-claim speaker attribution + deep-linked message backreferences
- Append-only per-channel Markdown decision log committed to a Git repo
- Scheduled daily/weekly rollups maintaining a running knowledge file
- Pluggable LLM backend (Claude/OpenAI/Gemini cloud or local Ollama); Telegram + Slack

## How it works

The bot reads the requested message range, extracts a structured decision record with attribution, posts it back to the channel, and appends the same entry to a Markdown file committed to a configured Git repo.

## Tech stack

- Python
- python-telegram-bot
- slack-bolt
- GitPython
- Claude / OpenAI / Gemini SDKs
- Ollama

## Status & roadmap

🚧 **Early development.** This repository is being built in the open; the scaffold and design are in place and the implementation is landing incrementally.

- [ ] Telegram `/digest` -> structured decision log with attribution
- [ ] Append-only Markdown log committed to Git
- [ ] Slack support + scheduled rollups
- [ ] Discord support; search over the committed knowledge file

## Installation

> Coming soon.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
