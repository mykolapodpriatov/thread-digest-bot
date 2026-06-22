"""Configuration models and loading.

Settings are validated Pydantic models so a malformed config fails with a clear error.
A config maps channels to a repo/path, selects the LLM backend, and sets the rollup
cadence. Secrets (bot tokens, API keys) are read from the environment by name, never
hard-coded in the file.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LLMProvider = Literal["fake", "openai", "anthropic", "ollama"]
PlatformName = Literal["telegram", "slack"]
RollupPeriod = Literal["daily", "weekly"]


class LLMConfig(BaseModel):
    """LLM backend selection.

    Attributes:
        provider: Which backend to construct.
        model: Provider model name (ignored by ``fake``).
        api_key_env: Environment variable holding the API key (cloud providers).
        base_url: Override base URL (e.g. a local Ollama endpoint).
        fixture: Named fixture for the ``fake`` provider.
    """

    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = "fake"
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    fixture: str = "happy"

    def api_key(self) -> str | None:
        """Resolve the API key from the environment, if configured."""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


class StorageConfig(BaseModel):
    """Where and how decision logs are persisted."""

    model_config = ConfigDict(extra="forbid")

    repo_root: str = "."
    decisions_dir: str = "docs/decisions"
    orphan_policy: Literal["auto-commit", "raise"] = "auto-commit"
    commit: bool = True


class ChannelConfig(BaseModel):
    """Per-channel settings."""

    model_config = ConfigDict(extra="forbid")

    channel_id: str
    platform: PlatformName = "telegram"
    default_last_n: int = 200
    rollup_period: RollupPeriod | None = None


class PlatformConfig(BaseModel):
    """A platform binding and its token environment variable."""

    model_config = ConfigDict(extra="forbid")

    name: PlatformName
    token_env: str | None = None

    def token(self) -> str | None:
        """Resolve the platform token from the environment, if configured."""
        if not self.token_env:
            return None
        return os.environ.get(self.token_env)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    platforms: list[PlatformConfig] = Field(default_factory=list)
    channels: list[ChannelConfig] = Field(default_factory=list)

    def channel(self, channel_id: str) -> ChannelConfig | None:
        """Return the configuration for a channel, if present."""
        for channel in self.channels:
            if channel.channel_id == channel_id:
                return channel
        return None


def load_config(path: str | os.PathLike[str]) -> AppConfig:
    """Load and validate an application config from a TOML file.

    Args:
        path: Path to a TOML config file.

    Returns:
        A validated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the config does not conform.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)
