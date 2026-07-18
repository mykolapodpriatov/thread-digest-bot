"""Command-line interface (``thread-digest-bot``).

Four subcommands mirror the plan:

* ``digest-file`` — offline digest of a validated ``thread.json`` using the configured
  (or Fake) LLM; great for demos and CI. The thread file is validated against a strict
  Pydantic schema first, so a malformed file fails with a clear message and a non-zero
  exit code rather than a stack trace.
* ``search`` — read-side substring search over the committed decision logs.
* ``rollup`` — build a periodic rollup for a channel.
* ``run`` — start the bot(s) from a config (platform adapters land in milestone M3).

All offline paths are deterministic: with the ``fake`` LLM provider, ``digest-file`` and
``rollup`` need no network.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from thread_digest_bot.config import AppConfig, LLMConfig, StorageConfig, load_config
from thread_digest_bot.digest import digest_with_report
from thread_digest_bot.grounding import GroundingReport
from thread_digest_bot.ingest import ThreadInput, thread_from_input
from thread_digest_bot.llm import LLMBackend
from thread_digest_bot.llm.factory import build_llm
from thread_digest_bot.render import render_chat_reply, render_json_entry, render_markdown_entry
from thread_digest_bot.rollup import rollup_label
from thread_digest_bot.search import search_logs
from thread_digest_bot.store import DecisionStore, StoreConfig
from thread_digest_bot.types import DecisionLog

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn chat threads into attributed, append-only decision logs.",
)

#: Exit code used for user-facing validation / not-found failures.
EXIT_USAGE_ERROR = 2

#: Valid ``digest-file --format`` values (``None`` keeps the default chat+md output).
_DIGEST_FORMATS = frozenset({"chat", "md", "json"})


def _err(message: str) -> None:
    """Print an error to stderr."""
    typer.echo(message, err=True)


def _render_for_format(log: DecisionLog, output_format: str) -> str:
    """Render ``log`` for a single explicit ``digest-file`` format."""
    if output_format == "json":
        return render_json_entry(log)
    if output_format == "chat":
        return render_chat_reply(log)
    return render_markdown_entry(log)  # "md"


def _print_grounding_report(report: GroundingReport) -> None:
    """Print the grounding drop report to stderr (keeps stdout machine-clean)."""
    _err("Grounding report:")
    _err(f"  dropped hallucinated citations: {report.dropped_hallucinated_citations}")
    _err(f"  dropped invalid quotes: {report.dropped_invalid_quotes}")
    _err(f"  dropped zero-citation items: {report.dropped_zero_citation_items}")


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
    output_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            help="Output format: 'chat', 'md', or 'json'. Default prints chat reply + md.",
        ),
    ] = None,
    stats: Annotated[
        bool,
        typer.Option("--stats", help="Print a grounding drop report to stderr."),
    ] = False,
) -> None:
    """Digest a thread JSON file offline into an attributed decision log."""
    if output_format is not None and output_format not in _DIGEST_FORMATS:
        _err(
            f"Error: unknown --format {output_format!r}; "
            f"expected one of {', '.join(sorted(_DIGEST_FORMATS))}."
        )
        raise typer.Exit(code=EXIT_USAGE_ERROR)

    thread_input = _load_thread_input(thread_file)
    thread = thread_from_input(thread_input)
    llm = _build_llm_from_config(config, fixture)

    log, report = digest_with_report(thread, llm, range_label=range_label)
    entry = render_markdown_entry(log)

    if commit:
        _commit_entry(log, repo_root, config)
        typer.echo(f"Committed digest for channel {log.channel_id} into {repo_root}.")
    elif out is not None:
        # Default to the Markdown entry (the committed audit format) unless overridden.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_for_format(log, output_format or "md"), encoding="utf-8")
        typer.echo(f"Wrote digest to {out}.")
    elif output_format is None:
        # Preserve the original stdout behavior: compact chat reply followed by the entry.
        typer.echo(render_chat_reply(log))
        typer.echo("")
        typer.echo(entry, nl=False)
    else:
        typer.echo(_render_for_format(log, output_format))

    if stats:
        _print_grounding_report(report)


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


@app.command("search")
def search(
    query: Annotated[
        str,
        typer.Argument(help="Case-insensitive substring to find across the decision logs."),
    ],
    repo_root: Annotated[
        Path,
        typer.Option("--repo-root", help="Repo root containing docs/decisions."),
    ] = Path(),
    channel: Annotated[
        str | None,
        typer.Option("--channel", help="Restrict the search to a single channel id."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'term' (default) or 'json'."),
    ] = "term",
) -> None:
    """Search the committed decision logs for a substring.

    Reads ``docs/decisions/<channel>.md`` back into entries and matches the query against
    each decision, action item, and open question. A repository with no ``docs/decisions``
    directory yields no matches rather than an error.
    """
    if output_format not in {"term", "json"}:
        _err(f"Error: unknown --format {output_format!r}; expected 'term' or 'json'.")
        raise typer.Exit(code=EXIT_USAGE_ERROR)

    hits = search_logs(repo_root, query, channel=channel)

    if output_format == "json":
        typer.echo(json.dumps([dataclasses.asdict(hit) for hit in hits], indent=2))
        return

    if not hits:
        typer.echo("No matches.")
        return
    for hit in hits:
        link = f" <{hit.permalink}>" if hit.permalink else ""
        typer.echo(f"[{hit.channel}] {hit.kind}: {hit.line}{link}")


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
