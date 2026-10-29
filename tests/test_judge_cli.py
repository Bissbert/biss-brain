"""Tests for judge.py cli_resolver integration."""
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


def _reset_judge_module() -> None:
    import cli_resolver
    cli_resolver.reset_cache()
    if "judge" in sys.modules:
        del sys.modules["judge"]


@pytest.fixture(autouse=True)
def _clean_judge():
    _reset_judge_module()
    yield
    _reset_judge_module()


# ---------------------------------------------------------------------------
# find_claude_cli() returns None → judge_conflict returns None
# ---------------------------------------------------------------------------

def test_judge_returns_none_when_cli_missing(monkeypatch, tmp_path):
    """When the CLI is unreachable, judge_conflict must return None."""
    monkeypatch.setenv("BISS_CLAUDE_BIN", "")
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path / "no-brain"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    import cli_resolver
    cli_resolver.reset_cache()

    if cli_resolver.find_claude_cli() is not None:
        pytest.skip("system has claude installed; cannot force None path")

    cli_resolver.reset_cache()
    import judge as _judge_mod

    # Provide a real-looking prompt file via mock so the OSError branch is skipped
    with patch.object(_judge_mod, "PROMPT_PATH") as mock_path:
        mock_path.read_text.return_value = (
            "New: {new_text}\nOld: {old_text}\n"
            "{old_project_label}{new_project_label}{same_project}"
        )
        result = _judge_mod.judge_conflict("new info", "old info")

    assert result is None, f"expected None, got {result!r}"


# ---------------------------------------------------------------------------
# find_claude_cli() returns fake path → correct argv[0] is used
# ---------------------------------------------------------------------------

def test_judge_uses_resolved_cli_path(tmp_path):
    """judge_conflict must pass the resolved CLI path as argv[0]."""
    import judge as _judge_mod
    fake_cli = _make_fake_cli(tmp_path / "bin")

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": "CONTRADICTS"})
    mock_proc.stderr = ""

    captured_cmd: list[list[str]] = []

    def fake_subprocess_run(cmd, **_kw):
        captured_cmd.append(list(cmd))
        return mock_proc

    # Patch the resolver where cli.run_claude looks it up.
    with patch.object(_judge_mod, "PROMPT_PATH") as mock_path, \
         patch("cli_resolver.find_claude_cli", return_value=fake_cli), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        mock_path.read_text.return_value = (
            "New: {new_text}\nOld: {old_text}\n"
            "{old_project_label}{new_project_label}{same_project}"
        )
        verdict = _judge_mod.judge_conflict("uses JWT", "uses sessions")

    assert verdict == "CONTRADICTS", f"unexpected verdict: {verdict!r}"
    assert len(captured_cmd) == 1, "subprocess.run not called exactly once"
    assert captured_cmd[0][0] == str(fake_cli), (
        f"expected argv[0]={fake_cli!s}, got {captured_cmd[0][0]!r}"
    )


# ---------------------------------------------------------------------------
# judge_with_project_context: CLI missing → verdict None
# ---------------------------------------------------------------------------

def test_judge_with_project_context_cli_missing(monkeypatch, tmp_path):
    """judge_with_project_context must return verdict None when the CLI is gone."""
    monkeypatch.setenv("BISS_CLAUDE_BIN", "")
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path / "no-brain"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    import cli_resolver
    cli_resolver.reset_cache()

    if cli_resolver.find_claude_cli() is not None:
        pytest.skip("system has claude installed; cannot force None path")

    cli_resolver.reset_cache()
    import judge as _judge_mod

    cfg = {"distill": {"models": {"conflict_judge": "haiku"}}}
    new_chunk = {"text": "Auth uses JWT", "project": "proj-a"}
    old_chunk = {"text": "Auth uses sessions", "project": "proj-a"}

    with patch.object(_judge_mod, "PROMPT_PATH") as mock_path:
        mock_path.read_text.return_value = (
            "New: {new_text}\nOld: {old_text}\n"
            "{old_project_label}{new_project_label}{same_project}"
        )
        result = _judge_mod.judge_with_project_context(new_chunk, old_chunk, True, cfg)

    assert isinstance(result, dict), "must return a dict"
    # A failure is "no verdict", never a verdict a caller could act on.
    assert result["verdict"] is None, f"unexpected verdict: {result['verdict']!r}"
    assert "reason" in result


# ---------------------------------------------------------------------------
# judge_with_project_context: resolved CLI path is used in argv
# ---------------------------------------------------------------------------

def test_judge_with_project_context_uses_resolved_cli(tmp_path):
    """judge_with_project_context must pass the resolved CLI as argv[0]."""
    import judge as _judge_mod
    fake_cli = _make_fake_cli(tmp_path / "bin")

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": json.dumps({
        "verdict": "UNRELATED", "action": "none", "reason": "no conflict",
    })})
    mock_proc.stderr = ""

    captured_cmd: list[list[str]] = []

    def fake_subprocess_run(cmd, **_kw):
        captured_cmd.append(list(cmd))
        return mock_proc

    cfg = {"distill": {"models": {"conflict_judge": "haiku"}}}
    new_chunk = {"text": "new info", "project": "p1"}
    old_chunk = {"text": "old info", "project": "p1"}

    # Patch the resolver where cli.run_claude looks it up.
    with patch.object(_judge_mod, "PROMPT_PATH") as mock_path, \
         patch("cli_resolver.find_claude_cli", return_value=fake_cli), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        mock_path.read_text.return_value = (
            "New: {new_text}\nOld: {old_text}\n"
            "{old_project_label}{new_project_label}{same_project}"
        )
        result = _judge_mod.judge_with_project_context(new_chunk, old_chunk, True, cfg)

    assert len(captured_cmd) == 1, "subprocess.run not called exactly once"
    assert captured_cmd[0][0] == str(fake_cli), (
        f"expected argv[0]={fake_cli!s}, got {captured_cmd[0][0]!r}"
    )
    assert result["verdict"] == "UNRELATED"


# ---------------------------------------------------------------------------
# R2 guard still applies when CLI is present
# ---------------------------------------------------------------------------

def test_r2_guard_rewrites_contradicts_cross_project(tmp_path):
    """Cross-project CONTRADICTS must be rewritten to DIVERGES by the R2 guard."""
    import judge as _judge_mod
    fake_cli = _make_fake_cli(tmp_path / "bin")

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"result": json.dumps({
        "verdict": "CONTRADICTS", "action": "supersede", "reason": "conflicts",
    })})
    mock_proc.stderr = ""

    cfg = {"distill": {"models": {"conflict_judge": "haiku"}}}
    new_chunk = {"text": "new info", "project": "project-x"}
    old_chunk = {"text": "old info", "project": "project-y"}

    # Patch the resolver where cli.run_claude looks it up.
    with patch.object(_judge_mod, "PROMPT_PATH") as mock_path, \
         patch("cli_resolver.find_claude_cli", return_value=fake_cli), \
         patch("subprocess.run", return_value=mock_proc):
        mock_path.read_text.return_value = (
            "New: {new_text}\nOld: {old_text}\n"
            "{old_project_label}{new_project_label}{same_project}"
        )
        result = _judge_mod.judge_with_project_context(new_chunk, old_chunk, False, cfg)

    assert result["verdict"] == "DIVERGES", (
        f"R2 guard failed: expected DIVERGES, got {result['verdict']!r}"
    )
