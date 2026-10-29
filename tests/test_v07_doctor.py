"""v0.7 phase 8.4: doctor checks, with Ollama, launchctl and the claude CLI mocked. Tmp roots only."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402
import doctor as D  # noqa: E402

CFG = {"embed": {"model": "nomic-embed-text", "dim": 768}, "distill": {"backend": "claude_code"},
       "scope": {"exclude": ["/tmp"], "include_only": []}}


def _meta(root: Path, name="abc123def456") -> None:
    (root / name).mkdir(parents=True)
    conn = sqlite3.connect(str(root / name / "meta.sqlite"))
    common.ensure_schema(conn)
    conn.close()


def test_resolver_missing_and_present(tmp_path):
    assert "not reachable" in D.resolver_checks(CFG, find=lambda: None)[0][0]
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert D.resolver_checks(CFG, find=lambda: exe) == []
    assert D.resolver_checks({**CFG, "distill": {"backend": "passthrough"}}, find=lambda: None) == []


def test_embed_dim_probe_and_failure(tmp_path):
    assert D.embed_checks(CFG, tmp_path, embed=lambda t: [0.0] * 768) == []
    assert "1024-dim" in D.embed_checks(CFG, tmp_path, embed=lambda t: [0.0] * 1024)[0][0]

    def down(t):
        raise ConnectionError

    assert "failed: ConnectionError" in D.embed_checks(CFG, tmp_path, embed=down)[0][0]
    assert D.embed_checks(CFG, tmp_path, embed=down, probe=False) == []


def test_store_marker_states(tmp_path):
    assert D.store_checks(CFG, tmp_path) == []  # empty brain, nothing to say
    _meta(tmp_path)
    assert "pre-v0.7" in D.store_checks(CFG, tmp_path)[0][0]
    common.write_store_marker(CFG, tmp_path, schema_only=True)
    assert "no embed fingerprint" in D.store_checks(CFG, tmp_path)[0][0]
    common.write_store_marker(CFG, tmp_path)
    assert D.store_checks(CFG, tmp_path) == []
    m = json.loads((tmp_path / "store.json").read_text())
    (tmp_path / "store.json").write_text(json.dumps({**m, "schema_version": 6}))
    assert "store schema 6" in D.store_checks(CFG, tmp_path)[0][0]


def _plist(agents: Path, label: str, script: Path) -> Path:
    agents.mkdir(exist_ok=True)
    p = agents / f"{label}.plist"
    p.write_text(json.dumps([str(script.parent / "_run.sh"), str(script)]))  # read by the fake plutil
    return p


def test_launchd_flags_other_versions_and_unloaded_jobs(tmp_path):
    cache = tmp_path / "cache" / "biss-brain"
    for v in ("0.5.4", "0.7.0"):
        (cache / v / "scripts").mkdir(parents=True)
        for f in ("_run.sh", "dream.py", "dmn.py"):
            (cache / v / "scripts" / f).write_text("")
    agents = tmp_path / "LaunchAgents"
    _plist(agents, "ch.bissbert.biss-brain.dream", cache / "0.5.4" / "scripts" / "dream.py")
    _plist(agents, "ch.bissbert.biss-brain.dmn", cache / "0.7.0" / "scripts" / "dmn.py")
    _plist(agents, "ch.bissbert.biss-brain.decay", tmp_path / "gone" / "scripts" / "decay.py")

    def run(cmd):
        if cmd[0] == "launchctl":
            return "PID\tStatus\tLabel\n-\t0\tch.bissbert.biss-brain.dream\n-\t0\tch.bissbert.biss-brain.decay\n"
        return Path(cmd[-1]).read_text()

    out = [w for w, _ in D.launchd_checks(cache / "0.7.0", agents=agents, run=run)]
    assert any("dream runs plugin 0.5.4" in w for w in out)
    assert any("decay runs a plugin copy that no longer exists" in w for w in out)
    assert any("dmn is installed but not loaded" in w for w in out)
    assert not any("dmn runs" in w for w in out)


def test_dream_age_and_error_after_completion(tmp_path):
    now = time.mktime(time.strptime("2026-10-26 12:00:00", "%Y-%m-%d %H:%M:%S"))
    log = tmp_path / "dream.log"
    assert D.dream_checks(tmp_path, now=now) == []
    log.write_text("2026-10-26 03:00:00,001 biss-brain.dream INFO dream pass complete: {\"secret\": 1}\n")
    assert D.dream_checks(tmp_path, now=now) == []
    out = D.dream_checks(tmp_path, now=now + 4 * 86400)
    assert "105 h ago" in out[0][0] and "secret" not in out[0][0]
    with log.open("a") as f:
        f.write("2026-10-26 04:00:00,000 biss-brain.dream ERROR stage 2 failed: token=abc\n")
    out = D.dream_checks(tmp_path, now=now)
    assert "ERROR at 2026-10-26 04:00" in out[0][0] and "token" not in out[0][0]


def test_pending_counts(tmp_path):
    import pending

    _meta(tmp_path)
    pdir = tmp_path / "abc123def456"
    assert D.pending_checks(tmp_path) == []
    pending.mark_embed(pdir, "/x")
    pending.skip_embed(pdir, "c1")
    pending.skip(pdir, "gk", "/t.jsonl", 0, 10, "boom")
    out = " | ".join(w for w, _ in D.pending_checks(tmp_path))
    assert "1 window(s) skipped" in out and "without a vector" in out and "1 memories the embedder rejects" in out


def test_scope_out_of_scope_and_missing_include(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "BRAIN_ROOT", tmp_path)
    cfg = {**CFG, "scope": {"exclude": [str(tmp_path / "skip")], "include_only": []}}
    (tmp_path / "skip").mkdir()
    (tmp_path / "work").mkdir()
    assert D.scope_checks(str(tmp_path / "work"), cfg) == []
    assert "out of scope" in D.scope_checks(str(tmp_path / "skip"), cfg)[0][0]
    cfg2 = {**CFG, "scope": {"exclude": [], "include_only": [str(tmp_path / "nope")]}}
    out = [w for w, _ in D.scope_checks(str(tmp_path / "work"), cfg2)]
    assert any("does not exist" in w for w in out) and any("out of scope" in w for w in out)
