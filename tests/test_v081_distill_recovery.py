"""v0.8.1: a NUL in a window no longer jams distill, a failed call says why,
and a newer plugin copy replaces a flush daemon started from an older one."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import capture  # noqa: E402
import cli  # noqa: E402
import doctor as D  # noqa: E402


def _cli(monkeypatch, tmp_path, run):
    monkeypatch.setattr(cli.cli_resolver, "find_claude_cli", lambda: tmp_path / "claude")
    monkeypatch.setattr(cli, "_workdir", lambda: str(tmp_path))
    monkeypatch.setattr(cli.subprocess, "run", run)
    failures = []
    monkeypatch.setattr(cli, "_record_failure", failures.append)
    return failures


def test_a_nul_in_the_prompt_is_dropped(monkeypatch, tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[2])
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": "ok"}), "")

    failures = _cli(monkeypatch, tmp_path, run)
    assert cli.run_claude("a\0b") == "ok"
    assert seen == ["ab"] and failures == []


def test_a_call_that_cannot_start_is_a_recorded_failure(monkeypatch, tmp_path):
    def run(cmd, **kw):
        raise OSError(7, "Argument list too long")

    failures = _cli(monkeypatch, tmp_path, run)
    assert cli.run_claude("x") is None
    assert failures == ["claude_cli_error"]


def test_a_failed_call_logs_the_reason_from_stdout(monkeypatch, tmp_path, caplog):
    def run(cmd, **kw):
        out = json.dumps({"is_error": True, "subtype": "error", "result": "Invalid API key"})
        return subprocess.CompletedProcess(cmd, 1, out, "")

    _cli(monkeypatch, tmp_path, run)
    with caplog.at_level(logging.WARNING, logger="biss-brain.cli"):
        assert cli.run_claude("x") is None
    assert "Invalid API key" in caplog.text


def _copy(base: Path, ver: str) -> Path:
    root = base / ver
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"version": ver}))
    return root


def test_an_older_daemon_is_replaced_and_a_newer_one_kept(monkeypatch, tmp_path):
    old, mine, newer = _copy(tmp_path, "0.7.0"), _copy(tmp_path, "0.8.1"), _copy(tmp_path, "0.10.0")
    killed = []
    monkeypatch.setattr(capture.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    monkeypatch.setattr(capture, "daemon_root", lambda pid: old)
    assert capture._replace_older(42, mine) is True
    assert killed == [(42, capture.signal.SIGTERM)]

    killed.clear()
    for running in (newer, mine, None):
        monkeypatch.setattr(capture, "daemon_root", lambda pid, r=running: r)
        assert capture._replace_older(42, mine) is False
    assert killed == []


def test_doctor_flags_a_daemon_from_an_older_copy(tmp_path):
    old, mine = _copy(tmp_path, "0.7.0"), _copy(tmp_path, "0.8.1")
    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / "flush.pid").write_text(f"{__import__('os').getpid()}\n")  # a live pid
    out = D.daemon_checks(brain, mine, root_of=lambda pid: old)
    assert len(out) == 1 and "0.7.0" in out[0][0]
    assert D.daemon_checks(brain, mine, root_of=lambda pid: mine) == []


def test_daemon_root_reads_the_flush_path(monkeypatch):
    import common

    cmd = "/c/biss-brain/0.7.0/.venv/bin/python /c/biss-brain/0.7.0/scripts/flush.py\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, cmd, ""))
    assert common.daemon_root(1) == Path("/c/biss-brain/0.7.0")


def test_the_dream_pass_records_its_state(tmp_path):
    import dream

    pdir = tmp_path / "abc123def456"
    pdir.mkdir()
    dream._record_state(pdir, "STAGEALL", error="RuntimeError: boom")
    dream._record_state(pdir, "COMPLETE")
    conn = dream._open_meta_direct(pdir)
    row = conn.execute("SELECT last_stage, last_complete_ts, last_error, last_error_ts FROM dream_state").fetchone()
    conn.close()
    assert row["last_stage"] == "COMPLETE" and row["last_complete_ts"] > 0
    assert row["last_error"] == "" and row["last_error_ts"] == 0  # a later success clears the error


def test_the_status_page_reads_the_keys_data_returns(monkeypatch):
    sys.path.insert(0, str(REPO))
    from ui import data
    from ui.server import templates

    status = data.fetch_brain_status("nope")  # defaults: the real key set
    status.update(dream_complete_ts=1_700_000_000, dream_error="boom", dream_error_ts=1_700_000_100,
                  dmn_plist_present=True, dream_plist_present=True)
    html = templates.get_template("status.html").render(
        status=status, project="x", project_label="x", request=None, projects=[])
    assert "2023-11-14 22:13 UTC" in html  # last complete
    assert "2023-11-14 22:15 UTC" in html  # error time
    assert "&#10007;" not in html  # both jobs shown present
