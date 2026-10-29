"""Tests for scripts.cli_resolver.find_claude_cli."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)


def _make_fake_cli(dirpath: Path, name: str = "claude") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text("#!/bin/sh\necho fake\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _reset() -> None:
    import cli_resolver
    cli_resolver.reset_cache()


def test_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    fake = _make_fake_cli(tmp_path / "bin")
    monkeypatch.setenv("BISS_CLAUDE_BIN", str(fake))
    monkeypatch.delenv("PATH", raising=False)
    _reset()
    from cli_resolver import find_claude_cli
    assert find_claude_cli() == fake.resolve()


def test_config_override_wins_over_path(tmp_path: Path, monkeypatch) -> None:
    fake = _make_fake_cli(tmp_path / "cfgbin")
    brain_root = tmp_path / "brain"
    brain_root.mkdir(parents=True)
    (brain_root / "config.toml").write_text(
        f'[distill]\nclaude_bin = "{fake}"\n'
    )
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(brain_root))
    monkeypatch.delenv("BISS_CLAUDE_BIN", raising=False)
    _reset()
    from cli_resolver import find_claude_cli
    assert find_claude_cli() == fake.resolve()


def test_path_lookup_fallback(tmp_path: Path, monkeypatch) -> None:
    bindir = tmp_path / "bin"
    fake = _make_fake_cli(bindir)
    monkeypatch.delenv("BISS_CLAUDE_BIN", raising=False)
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path / "noexist-brain"))
    monkeypatch.setenv("PATH", str(bindir))
    _reset()
    from cli_resolver import find_claude_cli
    assert find_claude_cli() == fake.resolve()


def test_returns_none_when_unreachable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BISS_CLAUDE_BIN", raising=False)
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path / "noexist-brain"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    _reset()
    from cli_resolver import find_claude_cli
    # All probe paths are under HOME or absolute system dirs; with HOME redirected
    # and PATH empty the only risk is /usr/local/bin / /opt/homebrew/bin existing
    # on the dev machine. Skip if a real claude is installed there.
    result = find_claude_cli()
    if result is not None:
        pytest.skip(f"system has claude at {result}; cannot test None case")
    assert result is None
