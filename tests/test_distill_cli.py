"""Tests for distill.py cli_resolver integration and truncation logic."""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_cli(dirpath: Path, name: str = "claude") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text("#!/bin/sh\necho fake\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _reset_distill_module() -> None:
    """Re-import distill so _FAILURE_COUNTER resets between tests."""
    import importlib
    import cli_resolver
    cli_resolver.reset_cache()
    if "distill" in sys.modules:
        del sys.modules["distill"]


@pytest.fixture(autouse=True)
def _clean_distill():
    _reset_distill_module()
    yield
    _reset_distill_module()


# ---------------------------------------------------------------------------
# find_claude_cli() returns None → distill returns None and counter increments
# ---------------------------------------------------------------------------

def test_distill_returns_none_when_cli_missing(monkeypatch, tmp_path):
    """When find_claude_cli() returns None, distill() must return None."""
    monkeypatch.setenv("BISS_CLAUDE_BIN", "")
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path / "no-brain"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    import cli_resolver
    cli_resolver.reset_cache()

    # Only run if we can force None (system may have claude in /usr/local/bin etc.)
    if cli_resolver.find_claude_cli() is not None:
        pytest.skip("system has claude installed; cannot force None path")

    cli_resolver.reset_cache()

    import distill as _distill_mod

    # Stub load_config to return a minimal claude_code config
    fake_cfg = {
        "distill": {"backend": "claude_code", "models": {"stop": "haiku"}},
        "embed": {"endpoint": "http://localhost:11434"},
    }
    with patch.object(_distill_mod, "load_config", return_value=fake_cfg):
        result = _distill_mod.distill("stop", "some session text")

    assert result is None, f"expected None, got {result!r}"
    assert _distill_mod._FAILURE_COUNTER["claude_cli_missing"] >= 1


def test_distill_cli_missing_counter_increments(monkeypatch, tmp_path):
    """Each call with missing CLI increments claude_cli_missing."""
    monkeypatch.setenv("BISS_CLAUDE_BIN", "")
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path / "no-brain"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    import cli_resolver
    cli_resolver.reset_cache()

    if cli_resolver.find_claude_cli() is not None:
        pytest.skip("system has claude installed; cannot force None path")

    cli_resolver.reset_cache()
    import distill as _distill_mod

    fake_cfg = {
        "distill": {"backend": "claude_code", "models": {"stop": "haiku"}},
        "embed": {"endpoint": "http://localhost:11434"},
    }

    before = _distill_mod._FAILURE_COUNTER["claude_cli_missing"]
    with patch.object(_distill_mod, "load_config", return_value=fake_cfg):
        _distill_mod.distill("stop", "text A")
        _distill_mod.distill("stop", "text B")

    assert _distill_mod._FAILURE_COUNTER["claude_cli_missing"] >= before + 2


# ---------------------------------------------------------------------------
# find_claude_cli() returns fake path → correct argv[0] is used
# ---------------------------------------------------------------------------

def test_distill_uses_resolved_cli_path(tmp_path):
    """When find_claude_cli() returns a path, that path must be argv[0]."""
    fake_cli = _make_fake_cli(tmp_path / "bin")

    import distill as _distill_mod

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "distilled summary"})
    mock_proc.stderr = ""

    captured_cmd: list[list[str]] = []

    def fake_subprocess_run(cmd, **_kw):
        captured_cmd.append(list(cmd))
        return mock_proc

    fake_cfg = {
        "distill": {"backend": "claude_code", "models": {"stop": "haiku"}},
        "embed": {"endpoint": "http://localhost:11434"},
    }

    # Patch the resolver where cli.run_claude looks it up (at call time).
    with patch.object(_distill_mod, "load_config", return_value=fake_cfg), \
         patch("cli_resolver.find_claude_cli", return_value=fake_cli), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        result = _distill_mod.distill("stop", "some session text")

    assert result == "distilled summary", f"unexpected result: {result!r}"
    assert len(captured_cmd) == 1, "subprocess.run not called exactly once"
    assert captured_cmd[0][0] == str(fake_cli), (
        f"expected argv[0]={fake_cli!s}, got {captured_cmd[0][0]!r}"
    )


# ---------------------------------------------------------------------------
# Truncation: 11 000-byte input → output ≤ 8000 and contains marker
# ---------------------------------------------------------------------------

def test_smart_truncate_large_input():
    """_smart_truncate must keep output ≤ 8000 bytes and embed the marker."""
    import distill as _distill_mod

    big_text = "x" * 11_000
    result = _distill_mod._smart_truncate(big_text)

    assert len(result) <= 8000, f"output too long: {len(result)} bytes"
    assert "[... truncated" in result, "truncation marker missing"


def test_smart_truncate_small_input():
    """Inputs within the cap must be returned unchanged."""
    import distill as _distill_mod

    text = "hello world"
    assert _distill_mod._smart_truncate(text) == text


def test_smart_truncate_exact_cap():
    """An input exactly at the cap must be returned unchanged."""
    import distill as _distill_mod

    text = "a" * 8000
    assert _distill_mod._smart_truncate(text) == text


def test_distill_passthrough_uses_smart_truncate(tmp_path):
    """The passthrough backend must apply _smart_truncate, not text[:2000]."""
    import distill as _distill_mod

    big_text = "z" * 11_000
    fake_cfg = {
        "distill": {"backend": "passthrough", "models": {}, "ollama": {"model": "q"}},
        "embed": {"endpoint": "http://localhost:11434"},
    }
    with patch.object(_distill_mod, "load_config", return_value=fake_cfg):
        result = _distill_mod.distill("stop", big_text)

    assert result is not None
    assert len(result) <= 8000, f"passthrough output too long: {len(result)}"
    assert "[... truncated" in result


# ---------------------------------------------------------------------------
# Timeout increments the right counter
# ---------------------------------------------------------------------------

def test_distill_timeout_increments_counter(tmp_path):
    """A TimeoutExpired from subprocess increments claude_cli_timeout."""
    import subprocess
    import distill as _distill_mod

    fake_cli = _make_fake_cli(tmp_path / "bin")

    fake_cfg = {
        "distill": {"backend": "claude_code", "models": {"stop": "haiku"}},
        "embed": {"endpoint": "http://localhost:11434"},
    }

    before = _distill_mod._FAILURE_COUNTER["claude_cli_timeout"]

    # Patch the resolver where cli.run_claude looks it up.
    with patch.object(_distill_mod, "load_config", return_value=fake_cfg), \
         patch("cli_resolver.find_claude_cli", return_value=fake_cli), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=1)):
        result = _distill_mod.distill("stop", "some text")

    assert result is None
    assert _distill_mod._FAILURE_COUNTER["claude_cli_timeout"] >= before + 1
