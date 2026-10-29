"""Resolve the absolute path to the `claude` CLI.

Subprocess contexts (LaunchAgent, capture hooks invoked by Claude Code itself,
cron) frequently inherit a sanitised PATH that excludes the user's shells'
profile additions. `shutil.which("claude")` then returns None and distill.py /
judge.py silently fall back to truncation/passthrough — the same root cause
behind the v0.6 audit's "every chunk is exactly 2000 bytes" finding.

Resolution order:
  1. config.toml `[distill] claude_bin` (operator override)
  2. `BISS_CLAUDE_BIN` env var (test/CI override)
  3. `shutil.which("claude")` (PATH-based)
  4. Curated probe list of well-known install locations

Returns None only if every probe fails — callers should log loudly and
degrade, never silently truncate.
"""
from __future__ import annotations

import os
import shutil
import tomllib
from functools import lru_cache
from pathlib import Path

_PROBE_PATHS: tuple[str, ...] = (
    "~/.local/bin/claude",
    "~/.npm-global/bin/claude",
    "/usr/local/bin/claude",
    "/opt/homebrew/bin/claude",
    "/usr/bin/claude",
)


def _from_config() -> Path | None:
    cfg = Path(os.environ.get("BISS_BRAIN_ROOT", "~/.claude/brain")).expanduser() / "config.toml"
    if not cfg.exists():
        return None
    try:
        with cfg.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    raw = (data.get("distill") or {}).get("claude_bin")
    if not raw or not isinstance(raw, str):
        return None
    p = Path(raw).expanduser()
    return p if p.is_file() and os.access(p, os.X_OK) else None


def _from_env() -> Path | None:
    raw = os.environ.get("BISS_CLAUDE_BIN")
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_file() and os.access(p, os.X_OK) else None


def _from_path() -> Path | None:
    found = shutil.which("claude")
    return Path(found) if found else None


def _from_probes() -> Path | None:
    for raw in _PROBE_PATHS:
        p = Path(raw).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


@lru_cache(maxsize=1)
def find_claude_cli() -> Path | None:
    """Return absolute path to `claude` or None if unreachable.

    Does NOT call .resolve() — the canonical install at
    ``~/.local/bin/claude`` is a symlink to the versioned binary, but
    only the launcher symlink loads the credential-discovery wrapper
    that talks to the user's keychain. Following the symlink yields a
    binary that responds "Not logged in" to every prompt. Use the
    absolute (non-resolved) form so the launcher executes.
    """
    for source in (_from_env, _from_config, _from_path, _from_probes):
        result = source()
        if result is not None:
            return Path(os.path.abspath(result))
    return None


def reset_cache() -> None:
    """Drop the memoised result — tests use this between fixtures."""
    find_claude_cli.cache_clear()
