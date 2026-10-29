#!/usr/bin/env python3
"""Tests for the nightly dream pass — Stage 2 (session arc + daily summary).

Mocks _distill_arc to return a fixed string so we don't need a Claude CLI on
PATH; uses BISS_BRAIN_FAKE_EMBED for vectors. Asserts:

- a kind='arc' chunk is created per qualifying session
- evidence_ids contains the source chunk ids
- consolidation_round == 1 on the new arc
- parent_id is set on every source chunk to point at the new arc
- daily summary roll-up creates kind='daily_summary' at consolidation_round=2
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


def _write_test_config(tmp: Path, daily_enabled: bool = False) -> Path:
    cfg_path = tmp / "config.toml"
    cfg_path.write_text(
        "[distill]\nbackend = \"passthrough\"\n"
        "[embed]\nbackend = \"ollama\"\nmodel = \"nomic-embed-text\"\ndim = 768\n"
        "endpoint = \"http://127.0.0.1:11434\"\n"
        "[retrieve]\nhybrid = false\n"
        f"[dream]\nenabled = true\nsession_cooldown_hours = 0\nmin_chunks_per_arc = 2\n"
        f"daily_summary_enabled = {'true' if daily_enabled else 'false'}\n"
        "[dream.stage1]\nbacklink_refresh = false\nconflict_sweep = false\n"
    )
    return cfg_path


def _reset_modules() -> None:
    for mod in ["common", "backlinks", "dream", "judge"]:
        if mod in sys.modules:
            del sys.modules[mod]


def _insert_chunk_direct(
    cwd: str,
    chunk_id: str,
    text: str,
    vec: list[float],
    *,
    kind: str = "stop",
    session_id: str = "",
    ts: int | None = None,
    dim: int = 768,
    consolidation_round: int = 0,
) -> None:
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
        "(id, project, session_id, agent_id, kind, ts, content_hash, superseded_by, "
        " last_accessed, consolidation_round) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?)",
        (chunk_id, proj, session_id, "", kind, ts, h, ts, consolidation_round),
    )
    meta.commit()
    meta.close()


def test_stage2a_creates_arc_and_links_parents() -> None:
    """Three chunks in one session → one arc, parent_id set on all three, evidence_ids matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_arc"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_FAKE_DISTILL"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream

        dim = 768
        sid = uuid.uuid4().hex[:16]
        ids = [uuid.uuid4().hex for _ in range(3)]
        base_ts = int(time.time()) - 7200
        for i, cid in enumerate(ids):
            _insert_chunk_direct(
                str(fake_cwd),
                cid,
                f"session chunk {i}",
                [0.1 + i * 0.01] * dim,
                kind="stop",
                session_id=sid,
                ts=base_ts + i * 60,
            )

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        n = dream._stage2a_arcs(pdir, proj, cfg, dry_run=False)
        assert n == 1, f"expected 1 arc built, got {n}"

        meta = common.open_meta(str(fake_cwd))
        try:
            arc = meta.execute(
                "SELECT id, kind, consolidation_round, evidence_ids, session_id "
                "FROM chunks WHERE kind = 'arc'"
            ).fetchone()
            assert arc is not None, "no arc row found"
            arc_id = arc[0]
            assert arc[1] == "arc"
            assert arc[2] == 1, f"expected consolidation_round=1, got {arc[2]}"
            assert arc[4] == sid, f"arc.session_id mismatch: {arc[4]} vs {sid}"
            evidence = set(arc[3].split(","))
            assert evidence == set(ids), f"evidence_ids mismatch: {evidence} vs {set(ids)}"

            for cid in ids:
                row = meta.execute(
                    "SELECT parent_id FROM chunks WHERE id = ?", (cid,)
                ).fetchone()
                assert row[0] == arc_id, (
                    f"chunk {cid[:8]} parent_id={row[0]!r}, expected {arc_id[:8]}"
                )
        finally:
            meta.close()
        print(f"  PASS: Stage 2a built arc {arc_id[:8]} from 3 session chunks; parents linked")


def test_stage2a_skips_short_sessions() -> None:
    """Sessions below min_chunks_per_arc are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_short"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_FAKE_DISTILL"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream

        dim = 768
        sid = uuid.uuid4().hex[:16]
        cid = uuid.uuid4().hex
        _insert_chunk_direct(
            str(fake_cwd), cid, "lone chunk", [0.2] * dim,
            kind="stop", session_id=sid, ts=int(time.time()) - 7200,
        )

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        n = dream._stage2a_arcs(pdir, proj, cfg, dry_run=False)
        assert n == 0, f"expected 0 arcs for solo chunk, got {n}"
        print("  PASS: Stage 2a skipped session with < min_chunks_per_arc")


def test_stage2a_idempotent() -> None:
    """Running Stage 2a twice does not produce duplicate arcs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "proj_idem"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_FAKE_DISTILL"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream

        dim = 768
        sid = uuid.uuid4().hex[:16]
        for i in range(2):
            _insert_chunk_direct(
                str(fake_cwd), uuid.uuid4().hex, f"x{i}", [0.3 + 0.01 * i] * dim,
                kind="stop", session_id=sid, ts=int(time.time()) - 7200 + i * 30,
            )

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        first = dream._stage2a_arcs(pdir, proj, cfg, dry_run=False)
        second = dream._stage2a_arcs(pdir, proj, cfg, dry_run=False)
        assert first == 1 and second == 0, (
            f"first run should build, second should skip; got {first}, {second}"
        )

        meta = common.open_meta(str(fake_cwd))
        try:
            count = meta.execute(
                "SELECT COUNT(*) FROM chunks WHERE kind = 'arc'"
            ).fetchone()[0]
        finally:
            meta.close()
        assert count == 1, f"expected exactly 1 arc, got {count}"
        print("  PASS: Stage 2a is idempotent")


def _noon_yesterday() -> int:
    """Noon yesterday, so arcs a few minutes apart share a calendar date even
    near midnight or across a DST change (on a 25-hour day, now - 86400 can
    still be today). Calendar arithmetic, not seconds; mktime normalises day 0."""
    lt = time.localtime()
    return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday - 1, 12, 0, 0, 0, 0, -1)))


def test_stage2b_daily_rollup_when_enabled() -> None:
    """daily_summary_enabled=true and ≥2 arcs from the same prior day → 1 daily_summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp, daily_enabled=True)
        fake_cwd = tmp / "proj_daily"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_FAKE_DISTILL"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream

        dim = 768
        # Two arcs from yesterday (different sessions, same calendar date)
        yesterday = _noon_yesterday()
        arc_ids = []
        for i in range(2):
            cid = uuid.uuid4().hex
            arc_ids.append(cid)
            _insert_chunk_direct(
                str(fake_cwd), cid, f"arc {i} text", [0.4 + 0.01 * i] * dim,
                kind="arc", session_id=uuid.uuid4().hex[:16], ts=yesterday + i * 600,
                consolidation_round=1,
            )

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        n = dream._stage2b_daily(pdir, proj, cfg, dry_run=False)
        assert n == 1, f"expected 1 daily summary, got {n}"

        meta = common.open_meta(str(fake_cwd))
        try:
            row = meta.execute(
                "SELECT id, consolidation_round, evidence_ids FROM chunks "
                "WHERE kind = 'daily_summary'"
            ).fetchone()
            assert row is not None
            assert row[1] == 2
            assert set(row[2].split(",")) == set(arc_ids)
            for aid in arc_ids:
                parent = meta.execute(
                    "SELECT parent_id FROM chunks WHERE id = ?", (aid,)
                ).fetchone()[0]
                assert parent == row[0], f"arc {aid[:8]} parent_id={parent}"
        finally:
            meta.close()
        print("  PASS: Stage 2b rolled 2 arcs into 1 daily_summary")


def test_stage2b_disabled_by_default() -> None:
    """daily_summary_enabled=false → no daily_summary even with arcs present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp, daily_enabled=False)
        fake_cwd = tmp / "proj_no_daily"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_FAKE_DISTILL"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
        os.environ["HOME"] = str(tmp)

        _reset_modules()
        import common
        from common import project_key, project_dir
        import dream

        dim = 768
        yesterday = _noon_yesterday()
        for i in range(2):
            _insert_chunk_direct(
                str(fake_cwd), uuid.uuid4().hex, f"arc {i}", [0.5 + 0.01 * i] * dim,
                kind="arc", session_id=uuid.uuid4().hex[:16], ts=yesterday + i * 600,
                consolidation_round=1,
            )

        proj = project_key(str(fake_cwd))
        pdir = project_dir(str(fake_cwd))
        cfg = common.load_config()
        n = dream._stage2b_daily(pdir, proj, cfg, dry_run=False)
        assert n == 0, f"daily disabled, expected 0, got {n}"
        print("  PASS: Stage 2b skipped when daily_summary_enabled=false")


if __name__ == "__main__":
    print("test_dream_stage2.py")
    test_stage2a_creates_arc_and_links_parents()
    test_stage2a_skips_short_sessions()
    test_stage2a_idempotent()
    test_stage2b_daily_rollup_when_enabled()
    test_stage2b_disabled_by_default()
    print("\nAll Stage 2 dream tests passed.")
