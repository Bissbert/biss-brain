"""v0.7 phase 8.3: in-place schema migration with a mandatory backup. Tmp roots only."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402
import migrate_v07 as M  # noqa: E402

RUNNER = REPO / "scripts" / "run_v07_migrations.sh"


def _old_store(root: Path) -> Path:
    """A project with the base chunks table and none of the additive columns (pre-v0.7)."""
    meta = root / "abc123def456" / "meta.sqlite"
    meta.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(meta))
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, project TEXT NOT NULL, session_id TEXT, "
                 "agent_id TEXT, kind TEXT NOT NULL, ts INTEGER NOT NULL, content_hash TEXT NOT NULL, "
                 "superseded_by TEXT NOT NULL DEFAULT '', pinned INTEGER DEFAULT 0, "
                 "retrieval_count INTEGER DEFAULT 0, last_accessed INTEGER, raw_ref TEXT)")
    conn.execute("INSERT INTO chunks (id, project, kind, ts, content_hash) "
                 "VALUES ('c1', 'abc123def456', 'stop', 1, 'h1')")
    conn.commit()
    conn.close()
    (root / "config.toml").write_text('[embed]\nmodel = "nomic-embed-text"\n')
    return meta


def test_runner_parses():
    assert subprocess.run(["bash", "-n", str(RUNNER)]).returncode == 0


def test_plan_reports_missing_columns_and_changes_nothing(tmp_path):
    meta = _old_store(tmp_path)
    before = meta.read_bytes()
    p = M.plan(tmp_path)
    assert p[0]["project"] == "abc123def456" and p[0]["columns"] > 0 and p[0]["fts"] == "no vectors"
    assert meta.read_bytes() == before


def test_backup_is_a_complete_tar(tmp_path):
    root = tmp_path / "brain"
    _old_store(root)
    dest = M.backup(root, now=0)
    assert dest == tmp_path / "brain-backups" / "brain-19700101T000000Z.tar.gz"
    with tarfile.open(dest) as t:
        assert "brain/abc123def456/meta.sqlite" in t.getnames()


def test_runner_yes_backs_up_migrates_and_marks_schema_only(tmp_path):
    root = tmp_path / "brain"
    meta = _old_store(root)
    env = {**os.environ, "BISS_BRAIN_ROOT": str(root), "BISS_BRAIN_CONFIG": str(root / "config.toml")}
    r = subprocess.run(["bash", str(RUNNER), "--yes"], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "launchctl bootstrap" in r.stdout
    assert len(list((tmp_path / "brain-backups").glob("brain-*.tar.gz"))) == 1
    conn = sqlite3.connect(str(meta))
    cols = {c[1] for c in conn.execute("PRAGMA table_info(chunks)")}
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    conn.close()
    assert M._missing_columns(meta) == 0 and "source" in cols
    marker = common.read_store_marker(root)
    assert marker["schema_version"] == common.SCHEMA_VERSION and "embed" not in marker


def test_lance_paths_fts_and_vector_dims(tmp_path):
    """The Lance branches of the migration plan, the FTS refresh and the doctor's dim and FTS checks."""
    import lancedb

    import doctor as D

    meta = _old_store(tmp_path)
    db = lancedb.connect(str(meta.parent / "lancedb"))
    tbl = db.create_table("chunks", schema=common.chunks_schema(8), mode="create")
    tbl.add([{"id": "c1", "project": "abc123def456", "session_id": "", "agent_id": "", "kind": "stop", "ts": 1,
              "text": "deploy with rsync", "content_hash": "h1", "superseded_by": "", "vector": [0.1] * 8}])
    assert M.plan(tmp_path)[0]["fts"] == "no FTS index"
    cfg = {"embed": {"model": "m", "dim": 8}}
    assert any("no keyword (FTS) index" in w for w, _ in D.store_checks(cfg, tmp_path))
    M.migrate(tmp_path, cfg)
    assert M.plan(tmp_path)[0]["fts"] == "ok"
    assert not any("FTS" in w for w, _ in D.store_checks(cfg, tmp_path))
    assert D.embed_checks(cfg, tmp_path, probe=False) == []
    out = D.embed_checks({"embed": {"model": "m", "dim": 768}}, tmp_path, probe=False)
    assert "stored vectors are 8-dim, config says 768" in out[0][0]
