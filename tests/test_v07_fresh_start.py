"""v0.7 phase 8.2: fresh start archives the old brain and carries config and core memory. Tmp roots only."""
from __future__ import annotations

import fcntl
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402
import fresh_start as F  # noqa: E402

CFG = {"embed": {"model": "nomic-embed-text", "dim": 768}}


@pytest.fixture
def brain(tmp_path):
    root = tmp_path / "brain"
    proj = root / "abc123def456"
    (proj / "queue").mkdir(parents=True)
    (proj / "queue" / "0001-stop-x.json").write_text("{}")
    conn = sqlite3.connect(str(proj / "meta.sqlite"))
    common.ensure_schema(conn)
    for i, pinned in enumerate((0, 1, 0)):
        conn.execute("INSERT INTO chunks (id, project, kind, ts, content_hash, pinned) VALUES (?, ?, 'stop', 1, ?, ?)",
                     (f"c{i}", proj.name, f"h{i}", pinned))
    conn.commit()
    conn.close()
    (root / "core" / ".git").mkdir(parents=True)
    (root / "core" / "_global.md").write_text("## Decisions\n- use ruff <!-- m:0123abcd -->\n")
    idx = sqlite3.connect(str(root / "core.sqlite"))
    idx.execute("CREATE TABLE core_items (mid TEXT)")
    idx.execute("INSERT INTO core_items VALUES ('0123abcd')")
    idx.commit()
    idx.close()
    (root / "config.toml").write_text('[embed]\nmodel = "nomic-embed-text"\n')
    (root / "flush.log").write_text("old log\n")
    (root / "scope.cache").write_text("e /\n")
    return root


def test_inventory_counts_what_stays_behind(brain):
    assert F.inventory(brain) == {"projects": 1, "chunks": 3, "pinned": 1, "queued": 1}


def test_apply_archives_everything_and_carries_config_and_core(brain):
    p = F.plan(brain, now=0)
    archive = F.apply(p, CFG, writers=[])
    assert archive == brain.parent / "brain-archive" / "brain-19700101T000000Z"
    assert (archive / "abc123def456" / "meta.sqlite").is_file() and (archive / "flush.log").exists()
    assert sorted(x.name for x in brain.iterdir()) == ["config.toml", "core", "core.sqlite", "store.json"]
    assert (brain / "core" / ".git").is_dir() and "m:0123abcd" in (brain / "core" / "_global.md").read_text()
    idx = sqlite3.connect(str(brain / "core.sqlite"))
    assert idx.execute("SELECT mid FROM core_items").fetchall() == [("0123abcd",)]
    idx.close()
    assert common.read_store_marker(brain)["embed"]["model"] == "nomic-embed-text"
    assert oct(brain.stat().st_mode & 0o777) == "0o700"
    assert not list(brain.parent.glob(".brain-fresh-*"))


def test_refuses_while_flush_holds_its_lock(brain):
    fd = os.open(str(brain / "flush.pid"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(F.Refused, match="flush daemon"):
            F.apply(F.plan(brain), CFG, writers=[])
    finally:
        os.close(fd)
    assert (brain / "abc123def456").is_dir()


def test_refuses_with_running_writers_or_an_existing_archive(brain):
    with pytest.raises(F.Refused, match="processes are running"):
        F.apply(F.plan(brain), CFG, writers=["123 python mcp/server.py"])
    p = F.plan(brain, now=0)
    p["archive"].mkdir(parents=True)
    with pytest.raises(F.Refused, match="already exists"):
        F.apply(p, CFG, writers=[])
    assert (brain / "abc123def456").is_dir()


def test_dry_run_changes_nothing(brain, monkeypatch, capsys):
    monkeypatch.setattr(common, "BRAIN_ROOT", brain)
    monkeypatch.setattr(F, "running_writers", lambda root=None: [])
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(brain / "config.toml"))
    before = sorted(str(x) for x in brain.rglob("*"))
    assert F.main([]) == 0
    assert sorted(str(x) for x in brain.rglob("*")) == before
    out = capsys.readouterr().out
    assert "3 memories (1 pinned)" in out and "dry run" in out
    assert not (brain.parent / "brain-archive").exists()


def test_process_probe_only_for_the_default_root(tmp_path, monkeypatch):
    monkeypatch.setattr(F.subprocess, "run", lambda *a, **k: pytest.fail("probed"))
    assert F.running_writers(tmp_path / "brain") == []


def test_a_failed_swap_restores_the_old_brain(brain, monkeypatch):
    real = os.rename
    calls = []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("boom")
        return real(src, dst)

    monkeypatch.setattr(F.os, "rename", flaky)
    with pytest.raises(OSError):
        F.apply(F.plan(brain), CFG, writers=[])
    assert (brain / "abc123def456" / "meta.sqlite").is_file()
