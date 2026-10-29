"""Wave 3: cli_failures_24h surfacing.

distill._record_failure appends to BRAIN_ROOT/cli_failures.jsonl; the status
panel reads the tail (last 24h) and renders a red-tinted nav pill when the
count is > 0. These tests verify the persistence + surfacing pipeline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)


def test_record_failure_persists_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path))
    # Force re-import so common.BRAIN_ROOT picks up the env var
    for mod in ("common", "distill", "ui.data"):
        sys.modules.pop(mod, None)

    import distill
    distill._record_failure("claude_cli_missing")
    distill._record_failure("claude_cli_timeout")
    distill._record_failure("claude_cli_missing")

    log = tmp_path / "cli_failures.jsonl"
    assert log.exists(), "_record_failure must create cli_failures.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 3
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["claude_cli_missing", "claude_cli_missing", "claude_cli_timeout"]
    assert all("ts" in r and r["ts"] > 0 for r in rows)


def test_count_cli_failures_24h_window(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path))
    for mod in ("common", "ui.data"):
        sys.modules.pop(mod, None)

    now = int(time.time())
    log = tmp_path / "cli_failures.jsonl"
    log.write_text(
        json.dumps({"ts": now - 90000, "kind": "claude_cli_missing"}) + "\n"  # > 24h: drop
        + json.dumps({"ts": now - 3600, "kind": "claude_cli_missing"}) + "\n"  # 1h ago: keep
        + json.dumps({"ts": now - 60, "kind": "claude_cli_error"}) + "\n"      # 1min ago: keep
        + "malformed line should not crash\n"
        + json.dumps({"ts": "not_an_int", "kind": "claude_cli_timeout"}) + "\n"  # malformed ts
    )

    from ui.data import _count_cli_failures_24h
    assert _count_cli_failures_24h() == 2


def test_count_cli_failures_missing_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path))
    for mod in ("common", "ui.data"):
        sys.modules.pop(mod, None)

    from ui.data import _count_cli_failures_24h
    assert _count_cli_failures_24h() == 0


def test_fetch_brain_status_includes_cli_failures(tmp_path: Path, monkeypatch) -> None:
    """fetch_brain_status returns cli_failures_24h derived from the JSONL log."""
    monkeypatch.setenv("BISS_BRAIN_ROOT", str(tmp_path))
    for mod in ("common", "ui.data"):
        sys.modules.pop(mod, None)

    now = int(time.time())
    (tmp_path / "cli_failures.jsonl").write_text(
        json.dumps({"ts": now - 60, "kind": "claude_cli_missing"}) + "\n"
        + json.dumps({"ts": now - 30, "kind": "claude_cli_error"}) + "\n"
    )

    # Build a project dir without a meta.sqlite so the function returns early
    # but still includes cli_failures_24h from the global counter.
    project = "abc123def456"
    (tmp_path / project).mkdir()

    from ui.data import fetch_brain_status
    status = fetch_brain_status(project)
    assert status["cli_failures_24h"] == 2
