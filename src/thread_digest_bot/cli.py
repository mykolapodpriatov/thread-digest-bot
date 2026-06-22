"""Command-line interface (``thread-digest-bot``).

Three subcommands mirror the plan:

* ``digest-file`` — offline digest of a validated ``thread.json`` using the configured
  (or Fake) LLM; great for demos and CI. The thread file is validated against a strict
  Pydantic schema first, so a malformed file fails with a clear message and a non-zero
  exit code rather than a stack trace.
* ``rollup`` — build a periodic rollup for a channel.
* ``run`` — start the bot(s) from a config (platform adapters land in milestone M3).

All offline paths are deterministic: with the ``fake`` LLM provider, ``digest-file`` and
``rollup`` need no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from thread_digest_bot.config import AppConfig, LLMConfig, StorageConfig, load_config
from thread_digest_bot.digest import digest
from thread_digest_bot.ingest import ThreadInput, thread_from_input
from thread_digest_bot.llm import LLMBackend
from thread_digest_bot.llm.factory import build_llm
from thread_digest_bot.render import render_chat_reply, render_markdown_entry
from thread_digest_bot.rollup import rollup_label
from thread_digest_bot.store import DecisionStore, StoreConfig
from thread_digest_bot.types import DecisionLog

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn chat threads into attributed, append-only decision logs.",
)

#: Exit code used for user-facing validation / not-found failures.
EXIT_USAGE_ERROR = 2


def _err(message: str) -> None:
    """Print an error to stderr."""
    typer.echo(message, err=True)


def _load_thread_input(path: Path) -> ThreadInput:
    """Read and validate a ``thread.json`` file into a :class:`ThreadInput`.

    Exits with :data:`EXIT_USAGE_ERROR` and a clear message on a missing file, invalid
    JSON, or a schema mismatch (missing field / unknown shape).
    """
    if not path.exists():
        _err(f"Error: thread file not found: {path}")
        raise typer.Exit(code=EXIT_USAGE_ERROR)
    raw = path.read_text(encoding="utf-8")
    try:
        return ThreadInput.model_validate_json(raw)
    except ValidationError as exc:
        _err(f"Error: {path} is not a valid thread file.")
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "<root>"
            _err(f"  - {location}: {error['msg']}")
        raise typer.Exit(code=EXIT_USAGE_ERROR) from None
    except json.JSONDecodeError as exc:
        _err(f"Error: {path} is not valid JSON: {exc}")
        raise typer.Exit(code=EXIT_USAGE_ERROR) from None


def _build_llm_from_config(config_path: Path | None, fixture: str) -> LLMBackend:
    """Build an LLM backend from a config file or the default Fake provider."""
    if config_path is not None:
        config = load_config(config_path)
    else:
        config = AppConfig(llm=LLMConfig(provider="fake", fixture=fixture))
    return build_llm(config.llm)


@app.command("digest-file")
def digest_file(
    thread_file: Annotated[
        Path,
        typer.Argument(help="Path to a thread.json file matching the documented schema."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the rendered Markdown entry to this file."),
    ] = None,
    commit: Annotated[
        bool,
        typer.Option("--commit/--no-commit", help="Commit the entry into a Git repo."),
    ] = False,
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", help="Git working-tree root for --commit."),
    ] = Path(),
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Config TOML selecting the LLM backend."),
    ] = None,
    fixture: Annotated[
        str,
        typer.Option("--fixture", help="FakeLLM fixture when no --config is given."),
    ] = "happy",
    range_label: Annotated[
        str | None,
        typer.Option("--range-label", help="Override the digest range label."),
    ] = None,
) -> None:
    """Digest a thread JSON file offline into an attributed decision log."""
    thread_input = _load_thread_input(thread_file)
    thread = thread_from_input(thread_input)
    llm = _build_llm_from_config(config, fixture)

    log = digest(thread, llm, range_label=range_label)
    entry = render_markdown_entry(log)

    if commit:
        _commit_entry(log, repo_root, config)
        typer.echo(f"Committed digest for channel {log.channel_id} into {repo_root}.")
    elif out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(entry, encoding="utf-8")
        typer.echo(f"Wrote digest to {out}.")
    else:
        typer.echo(render_chat_reply(log))
        typer.echo("")
        typer.echo(entry, nl=False)


def _commit_entry(log: DecisionLog, repo_root: Path, config_path: Path | None) -> None:
    """Append ``log`` into the Git-backed store at ``repo_root``."""
    if config_path is not None:
        storage: StorageConfig = load_config(config_path).storage
        store_config = StoreConfig(
            decisions_dir=storage.decisions_dir,
            orphan_policy=storage.orphan_policy,
            commit=True,
        )
    else:
        store_config = StoreConfig(commit=True)
    store = DecisionStore(repo_root, config=store_config)
    store.append(log)


@app.command("rollup")
def rollup(
    channel: Annotated[str, typer.Option("--channel", help="Channel id to roll up.")],
    period: Annotated[
        str,
        typer.Option("--period", help="Cadence label, e.g. 'weekly' or 'daily'."),
    ] = "weekly",
    period_key: Annotated[
        str,
        typer.Option("--period-key", help="Stable period identifier, e.g. '2026-W25'."),
    ] = "current",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Config TOML selecting the LLM backend."),
    ] = None,
    fixture: Annotated[
        str,
        typer.Option("--fixture", help="FakeLLM fixture when no --config is given."),
    ] = "empty",
) -> None:
    """Print the range label a rollup would use for a channel/period.

    The full fetch-and-commit rollup runs through a live platform (milestone M3); this
    offline command surfaces the deterministic label so schedules can be inspected.
    """
    label = rollup_label(period, period_key)
    # ``config``/``fixture`` are accepted for parity with ``digest-file`` and validated
    # here so a bad config fails fast even on the offline path.
    if config is not None:
        load_config(config)
    typer.echo(f"Rollup for channel {channel}: {label}")


@app.command("run")
def run(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to the application config TOML."),
    ],
) -> None:
    """Start the configured bot(s).

    Live Telegram/Slack adapters land in milestone M3; this command validates the
    config and reports the configured platforms so deployments fail fast on a bad file.
    """
    app_config = load_config(config)
    if not app_config.platforms:
        _err("Error: no platforms configured; nothing to run.")
        raise typer.Exit(code=EXIT_USAGE_ERROR)
    names = ", ".join(p.name for p in app_config.platforms)
    typer.echo(f"Configured platforms: {names}.")
    typer.echo("Live platform adapters land in milestone M3; use 'digest-file' offline.")


def main() -> None:  # pragma: no cover - thin console-script shim
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
