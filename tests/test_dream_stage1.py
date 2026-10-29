#!/usr/bin/env python3
"""Tests for the nightly dream pass — Stage 1 (backlinks + conflict + decay).

Strategy mirrors test_backlinks.py: use BISS_BRAIN_FAKE_EMBED so we can mint
deterministic vectors, write directly into LanceDB+SQLite, then call the
relevant dream.py entry points and inspect SQLite afterwards.

External calls (judge_conflict) are monkey-patched so the tests run with no
network and no Claude CLI on PATH.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import lancedb
import pyarrow as pa

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ── helpers ───────────────────────────────────────────────────────────────


def _write_test_config(tmp: Path) -> Path:
    cfg_path = tmp / "config.toml"
    cfg_path.write_text(
        "[distill]\nbackend = \"passthrough\"\n"
        "[embed]\nbackend = \"ollama\"\nmodel = \"nomic-embed-text\"\ndim = 768\n"
        "endpoint = \"http://127.0.0.1:11434\"\n"
        "[retrieve]\nhybrid = false\n"
        "[dream]\nenabled = true\nsession_cooldown_hours = 0\nmin_chunks_per_arc = 2\n"
        "[dream.stage1]\nbacklink_refresh = true\nconflict_sweep = true\n"
        "conflict_sim_threshold = 0.85\ncandidate_window_hours = 24\n"
    )
    return cfg_path


def _insert_chunk_direct(
    cwd: str,
    chunk_id: str,
    text: str,
    vec: list[float],
    *,
    kind: str = "note",
    session_id: str = "",
    ts: int | None = None,
    dim: int = 768,
) -> None:
    """Insert a row into both LanceDB and SQLite."""
    from common import lancedb_dir, open_meta, project_key

    proj = project_key(cwd)
    h = hashlib.sha256(text.encode()).hexdigest()
    ts = ts or int(time.time())

    db = lancedb.connect(str(lancedb_dir(cwd)))
    if "chunks" in db.list_tables().tables:
        tbl = db.open_table("chunks")
    else:
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("project", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("agent_id", pa.string()),
                pa.field("kind", pa.string()),
                pa.field("ts", pa.int64()),
                pa.field("text", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("superseded_by", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        tbl = db.create_table("chunks", schema=schema, mode="create")

    tbl.add(
        [
            {
                "id": chunk_id,
                "project": proj,
                "session_id": session_id,
                "agent_id": "",
                "kind": kind,
                "ts": ts,
                "text": text,
                "content_hash": h,
                "superseded_by": "",
                "vector": [float(v) for v in vec],
            }
        ]
    )

    meta = open_meta(cwd)
    meta.execute(
        "INSERT INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, superseded_by, last_accessed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, '', ?)",
        (chunk_id, proj, session_id, "", kind, ts, h, ts),
    )
    meta.commit()
    meta.close()


def _reset_modules() -> None:
    for mod in ["common", "backlinks", "dream", "judge"]:
        if mod in sys.modules:
            del sys.modules[mod]


# ── tests ─────────────────────────────────────────────────────────────────


def test_stage1a_backlinks_refreshes_links() -> None:
    """Stage 1a re-runs the A-MEM-style linking on chunks not yet visited."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_s1a"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)  # isolate ~/.claude/brain

        _reset_modules()
        import common  # noqa: F401
        from common import project_key, project_dir
        import dream

        dim = 768
        base = [0.5] * dim
        near = [v + 0.001 for v in base]

        id_a = uuid.uuid4().hex
        id_b = uuid.uuid4().hex
        _insert_chunk_direct(str(fake_cwd), id_a, "near vector A", base)
        _insert_chunk_direct(str(fake_cwd), id_b, "near vector B", near)

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))

        # Sanity: no related_ids yet
        cfg = common.load_config()
        # Run only Stage 1a
        n = dream._stage1a_backlinks(pdir, proj, cfg, dry_run=False)
        assert n >= 2, f"expected to refresh at least 2 chunks, got {n}"

        from backlinks import fetch_related
        related_a = fetch_related(str(fake_cwd), id_a)
        related_b = fetch_related(str(fake_cwd), id_b)
        assert id_b in related_a, f"id_b not in id_a's related_ids: {related_a}"
        assert id_a in related_b, f"id_a not in id_b's related_ids: {related_b}"
        print(f"  PASS: Stage 1a wrote bidirectional backlinks for {n} chunks")


def test_stage1b_conflict_marks_superseded() -> None:
    """Stage 1b sets superseded_by on the OLDER chunk when judge says CONTRADICTS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_s1b"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream
        import judge as judge_mod

        # Monkey-patch judge_conflict so we don't shell out to claude
        def _fake_judge(new_text: str, old_text: str, model: str = "haiku") -> str:
            return "CONTRADICTS"
        judge_mod.judge_conflict = _fake_judge

        dim = 768
        base = [0.5] * dim
        near = [v + 0.001 for v in base]

        id_old = uuid.uuid4().hex
        id_new = uuid.uuid4().hex
        old_ts = int(time.time()) - 100
        new_ts = int(time.time()) - 50
        _insert_chunk_direct(str(fake_cwd), id_old, "old: uses JWT", base, ts=old_ts)
        _insert_chunk_direct(str(fake_cwd), id_new, "new: uses session cookies", near, ts=new_ts)

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        counters = dream._stage1b_conflicts(pdir, proj, cfg, dry_run=False)
        assert counters["contradicts"] >= 1, f"expected ≥1 contradict, got {counters}"

        meta = common.open_meta(str(fake_cwd))
        try:
            row = meta.execute(
                "SELECT superseded_by FROM chunks WHERE id = ?", (id_old,)
            ).fetchone()
        finally:
            meta.close()
        assert row[0] == id_new, (
            f"expected old chunk superseded_by={id_new[:8]}, got {row[0]!r}"
        )
        print(f"  PASS: Stage 1b marked {id_old[:8]} superseded by {id_new[:8]}")


def test_stage1b_extends_appends_related() -> None:
    """Stage 1b appends the neighbour to related_ids when judge says EXTENDS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_s1b2"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream
        import judge as judge_mod

        judge_mod.judge_conflict = lambda new_text, old_text, model="haiku": "EXTENDS"

        dim = 768
        base = [0.5] * dim
        near = [v + 0.001 for v in base]

        id_old = uuid.uuid4().hex
        id_new = uuid.uuid4().hex
        _insert_chunk_direct(str(fake_cwd), id_old, "old: chose LanceDB", base,
                             ts=int(time.time()) - 100)
        _insert_chunk_direct(str(fake_cwd), id_new, "new: LanceDB hybrid mode", near,
                             ts=int(time.time()) - 50)

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        counters = dream._stage1b_conflicts(pdir, proj, cfg, dry_run=False)
        assert counters["extends"] >= 1, f"expected ≥1 extend, got {counters}"

        from backlinks import fetch_related
        related = fetch_related(str(fake_cwd), id_new)
        assert id_old in related, f"id_old not in id_new related_ids: {related}"
        print(f"  PASS: Stage 1b appended {id_old[:8]} to {id_new[:8]} related_ids")


def test_stage1c_decay_recomputes() -> None:
    """Stage 1c writes a non-default decay_score to every chunk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_s1c"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_dir
        import dream

        dim = 768
        # Old chunk → low decay; recent chunk → near 1.0
        old_id = uuid.uuid4().hex
        new_id = uuid.uuid4().hex
        _insert_chunk_direct(
            str(fake_cwd), old_id, "ancient", [0.1] * dim, ts=int(time.time()) - 86400 * 365
        )
        _insert_chunk_direct(
            str(fake_cwd), new_id, "fresh", [0.1] * dim, ts=int(time.time()) - 60
        )

        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        n = dream._stage1c_decay(pdir, cfg, dry_run=False)
        assert n == 2, f"expected 2 decayed rows, got {n}"

        meta = common.open_meta(str(fake_cwd))
        try:
            old_score = meta.execute(
                "SELECT decay_score FROM chunks WHERE id = ?", (old_id,)
            ).fetchone()[0]
            new_score = meta.execute(
                "SELECT decay_score FROM chunks WHERE id = ?", (new_id,)
            ).fetchone()[0]
        finally:
            meta.close()
        assert new_score > old_score, f"expected new>old: {new_score} vs {old_score}"
        assert old_score < 0.5, f"expected old to decay below 0.5, got {old_score}"
        print(f"  PASS: Stage 1c decay recomputed: old={old_score:.4f} new={new_score:.4f}")


def test_stage1_dry_run_writes_nothing() -> None:
    """--dry-run does not modify SQLite or LanceDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_dry"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream
        import judge as judge_mod
        judge_mod.judge_conflict = lambda *a, **k: "CONTRADICTS"

        dim = 768
        id_a = uuid.uuid4().hex
        id_b = uuid.uuid4().hex
        _insert_chunk_direct(str(fake_cwd), id_a, "alpha", [0.5] * dim, ts=int(time.time()) - 100)
        _insert_chunk_direct(
            str(fake_cwd), id_b, "alpha v2", [0.5 + 0.001] * dim, ts=int(time.time()) - 50
        )

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        dream.run_stage1(pdir, cfg, dry_run=True)

        meta = common.open_meta(str(fake_cwd))
        try:
            row = meta.execute(
                "SELECT superseded_by FROM chunks WHERE id = ?", (id_a,)
            ).fetchone()
            last_dream = meta.execute(
                "SELECT last_dream_ts FROM chunks WHERE id = ?", (id_a,)
            ).fetchone()
        finally:
            meta.close()
        assert row[0] == "", f"dry-run should not set superseded_by, got {row[0]!r}"
        assert last_dream[0] == 0, f"dry-run should not bump last_dream_ts, got {last_dream[0]}"
        print("  PASS: Stage 1 --dry-run leaves all rows untouched")


if __name__ == "__main__":
    print("test_dream_stage1.py")
    test_stage1a_backlinks_refreshes_links()
    test_stage1b_conflict_marks_superseded()
    test_stage1b_extends_appends_related()
    test_stage1c_decay_recomputes()
    test_stage1_dry_run_writes_nothing()
    print("\nAll Stage 1 dream tests passed.")
