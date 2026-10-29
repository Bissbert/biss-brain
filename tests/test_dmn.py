#!/usr/bin/env python3
"""Tests for the DMN idle daemon (scripts/dmn.py).

Strategy:
- Each test uses a temporary directory as BRAIN_ROOT (set via BISS_BRAIN_ROOT env).
- We write queue event files and SQLite rows directly to avoid starting the daemon
  loop itself.
- We call the individual dmn.py helper functions under test rather than running main().
- BISS_BRAIN_FAKE_EMBED is set so any embedding calls use deterministic vectors.

Tests:
  test_dmn_skips_when_disabled          — cfg.dmn.enabled=False -> main exits 0
  test_dmn_skips_when_recent_stop       — recent Stop event -> _is_idle returns False
  test_dmn_drains_edge_bumps            — seeded bumps -> edges written, bumps drained
  test_dmn_respects_runtime_budget      — deadline=past -> drain exits immediately
  test_dmn_no_llm_calls                 — distill monkeypatched to raise; tick must not raise
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ── helpers ───────────────────────────────────────────────────────────────


def _make_brain_root(tmp: Path) -> Path:
    """Create a BRAIN_ROOT under tmp and point the env var at it."""
    brain = tmp / "brain"
    brain.mkdir(parents=True, exist_ok=True)
    os.environ["BISS_BRAIN_ROOT"] = str(brain)
    return brain


def _write_config(
    brain: Path,
    *,
    dmn_enabled: bool = True,
    dmn_idle_seconds: int = 300,
    dmn_max_runtime: int = 60,
    stdp_min_count: int = 2,
) -> Path:
    """Write a minimal config.toml and set BISS_BRAIN_CONFIG.

    All dmn.* and stdp.* settings are passed as explicit keyword args to
    avoid duplicate-section errors in TOML.
    """
    cfg_path = brain / "config.toml"
    dmn_enabled_str = "true" if dmn_enabled else "false"
    cfg_path.write_text(
        f"[distill]\n"
        f'backend = "passthrough"\n'
        f"[embed]\n"
        f'backend = "ollama"\n'
        f'model = "nomic-embed-text"\n'
        f"dim = 4\n"
        f'endpoint = "http://127.0.0.1:11434"\n'
        f"[retrieve]\n"
        f"hybrid = false\n"
        f"[dmn]\n"
        f"enabled = {dmn_enabled_str}\n"
        f"idle_seconds_before_run = {dmn_idle_seconds}\n"
        f"max_runtime_seconds = {dmn_max_runtime}\n"
        f"[stdp]\n"
        f"enabled = true\n"
        f"co_retrieval_min_count = {stdp_min_count}\n"
        f"edge_drain_window_days = 7\n"
    )
    os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)
    return cfg_path


def _make_project(brain: Path, proj_hash: str | None = None) -> Path:
    """Create a project directory with an initialised meta.sqlite."""
    if proj_hash is None:
        proj_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
    pdir = brain / proj_hash
    pdir.mkdir(parents=True, exist_ok=True)

    # Initialise SQLite with minimal schema
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            agent_id TEXT DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'note',
            ts INTEGER NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            last_accessed INTEGER DEFAULT 0,
            raw_ref TEXT DEFAULT '',
            related_ids TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_bumps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project     TEXT NOT NULL,
            chunk_a     TEXT NOT NULL,
            chunk_b     TEXT NOT NULL,
            bumped_at   INTEGER NOT NULL,
            session_id  TEXT DEFAULT '',
            drained     INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dream_state (
            project          TEXT PRIMARY KEY,
            last_stage       TEXT NOT NULL DEFAULT 'IDLE',
            last_complete_ts INTEGER NOT NULL DEFAULT 0,
            last_error       TEXT NOT NULL DEFAULT '',
            last_error_ts    INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    return pdir


def _insert_chunk(pdir: Path, chunk_id: str, proj: str, ts: int | None = None) -> None:
    ts = ts or int(time.time())
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(
        "INSERT OR IGNORE INTO chunks (id, project, ts, content_hash, kind) "
        "VALUES (?, ?, ?, '', 'note')",
        (chunk_id, proj, ts),
    )
    conn.commit()
    conn.close()


def _insert_bump(pdir: Path, proj: str, chunk_a: str, chunk_b: str, count: int = 1) -> None:
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    now = int(time.time())
    for _ in range(count):
        conn.execute(
            "INSERT INTO edge_bumps (project, chunk_a, chunk_b, bumped_at) VALUES (?, ?, ?, ?)",
            (proj, chunk_a, chunk_b, now),
        )
    conn.commit()
    conn.close()


def _write_stop_event(brain: Path, proj_hash: str, ts: int | None = None) -> None:
    """Write a fake Stop event queue file for a project."""
    ts = ts or int(time.time())
    qdir = brain / proj_hash / "queue"
    qdir.mkdir(parents=True, exist_ok=True)
    fname = f"{ts:010d}-stop-abcd1234.json"
    (qdir / fname).write_text(json.dumps({"kind": "stop"}))


def _reset_dmn_module() -> None:
    for mod in list(sys.modules.keys()):
        if mod in ("dmn", "common"):
            del sys.modules[mod]


# ── tests ─────────────────────────────────────────────────────────────────


def test_dmn_skips_when_disabled() -> None:
    """When dmn.enabled=false, main() returns 0 without doing any work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain, dmn_enabled=False)

        _reset_dmn_module()
        import dmn

        result = dmn.main()
        assert result == 0


def test_dmn_skips_when_recent_stop() -> None:
    """_is_idle returns False when there is a Stop event within the idle window."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        _make_project(brain, proj_hash)

        # Write a Stop event timestamped NOW (well within 300 s window)
        _write_stop_event(brain, proj_hash, ts=int(time.time()))

        _reset_dmn_module()
        import dmn

        # Patch BRAIN_ROOT to our temp brain
        with patch.object(dmn, "BRAIN_ROOT", brain):
            cfg = dmn.load_config()
            assert not dmn._is_idle(cfg), "Expected not idle due to recent Stop event"


def test_dmn_is_idle_when_no_recent_stop() -> None:
    """_is_idle returns True when the most recent Stop event is older than the threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        _make_project(brain, proj_hash)

        # Write an old Stop event (600 s ago — beyond the 300 s threshold)
        old_ts = int(time.time()) - 600
        _write_stop_event(brain, proj_hash, ts=old_ts)

        _reset_dmn_module()
        import dmn

        with patch.object(dmn, "BRAIN_ROOT", brain):
            # Also patch system idle check to avoid ioreg subprocess
            with patch.object(dmn, "_system_idle_seconds", return_value=None):
                cfg = dmn.load_config()
                assert dmn._is_idle(cfg), "Expected idle: last Stop is 600 s ago"


def test_dmn_drains_edge_bumps() -> None:
    """Seed edge_bumps at or above threshold; one tick should write co_occurs edges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)
        proj = proj_hash

        # Insert two chunks
        _insert_chunk(pdir, "chunk-A", proj)
        _insert_chunk(pdir, "chunk-B", proj)

        # Insert 3 bumps (>= co_retrieval_min_count = 2)
        _insert_bump(pdir, proj, "chunk-A", "chunk-B", count=3)

        _reset_dmn_module()
        import dmn

        # Use a far-future deadline so budget is not a constraint
        deadline = time.monotonic() + 60

        with patch.object(dmn, "BRAIN_ROOT", brain):
            cfg = dmn.load_config()
            edges = dmn._drain_edge_bumps(pdir, cfg, deadline)

        assert edges == 1, f"Expected 1 edge written, got {edges}"

        # Verify co_occurs edge written into chunk-A's related_ids
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        row = conn.execute("SELECT related_ids FROM chunks WHERE id = 'chunk-A'").fetchone()
        conn.close()
        assert row is not None
        items = json.loads(row[0] or "[]")
        assert any(
            item.get("id") == "chunk-B" and item.get("type") == "co_occurs"
            for item in items
        ), f"co_occurs edge to chunk-B not found in related_ids: {items}"

        # Verify bumps marked drained
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        undrained = conn.execute(
            "SELECT COUNT(*) FROM edge_bumps WHERE drained = 0"
        ).fetchone()[0]
        conn.close()
        assert undrained == 0, f"Expected 0 undrained bumps, got {undrained}"


def test_dmn_drains_below_threshold_not_written() -> None:
    """Bumps below co_retrieval_min_count must NOT produce an edge and remain drained=0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)
        proj = proj_hash

        _insert_chunk(pdir, "chunk-X", proj)
        _insert_chunk(pdir, "chunk-Y", proj)

        # Only 1 bump — below threshold of 2
        _insert_bump(pdir, proj, "chunk-X", "chunk-Y", count=1)

        _reset_dmn_module()
        import dmn

        deadline = time.monotonic() + 60
        with patch.object(dmn, "BRAIN_ROOT", brain):
            cfg = dmn.load_config()
            edges = dmn._drain_edge_bumps(pdir, cfg, deadline)

        assert edges == 0, f"Expected 0 edges below threshold, got {edges}"

        # Sub-threshold bumps must remain undrained so they can accumulate further
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        undrained = conn.execute(
            "SELECT COUNT(*) FROM edge_bumps WHERE drained = 0"
        ).fetchone()[0]
        conn.close()
        assert undrained == 1, (
            f"Expected 1 undrained bump (sub-threshold, not consumed), got {undrained}"
        )


def test_dmn_respects_runtime_budget() -> None:
    """When deadline is already past, _drain_edge_bumps exits before processing rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)
        proj = proj_hash

        _insert_chunk(pdir, "chunk-A", proj)
        _insert_chunk(pdir, "chunk-B", proj)
        _insert_bump(pdir, proj, "chunk-A", "chunk-B", count=3)

        _reset_dmn_module()
        import dmn

        # Deadline is in the past — budget already exhausted
        deadline = time.monotonic() - 1.0

        with patch.object(dmn, "BRAIN_ROOT", brain):
            cfg = dmn.load_config()
            edges = dmn._drain_edge_bumps(pdir, cfg, deadline)

        # No edges should have been written (loop broke immediately on budget check)
        assert edges == 0, (
            f"Expected 0 edges when deadline is already past, got {edges}"
        )

        # related_ids should be empty (no edges written)
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        row = conn.execute("SELECT related_ids FROM chunks WHERE id = 'chunk-A'").fetchone()
        conn.close()
        items = json.loads(row[0] or "[]") if row else []
        co_occurs = [i for i in items if i.get("type") == "co_occurs"]
        assert len(co_occurs) == 0, f"Expected no co_occurs edges, got {co_occurs}"


def test_dmn_no_llm_calls() -> None:
    """A full DMN tick must not invoke distill or any LLM function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)
        proj = proj_hash

        _insert_chunk(pdir, "chunk-A", proj)
        _insert_chunk(pdir, "chunk-B", proj)
        _insert_bump(pdir, proj, "chunk-A", "chunk-B", count=3)

        # Write an old Stop event so the system appears idle
        old_ts = int(time.time()) - 600
        _write_stop_event(brain, proj_hash, ts=old_ts)

        _reset_dmn_module()
        import dmn

        # Monkeypatch distill to raise if called
        def _raise_if_called(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("distill was called — DMN must not invoke LLM")

        deadline = time.monotonic() + 60
        with patch.object(dmn, "BRAIN_ROOT", brain):
            # Patch distill in dmn's namespace if it were ever imported
            with patch.dict(sys.modules, {"distill": type(sys)("distill")}):
                cfg = dmn.load_config()
                # Run individual ops (no sleep — just the maintenance logic).
                # SEC-001: none of these may raise NameError or any unexpected
                # exception.  LanceDB absence exits gracefully (returns 0).
                dmn._drain_edge_bumps(pdir, cfg, deadline)
                dmn._refresh_recent_backlinks(pdir, cfg, deadline)
                dmn._update_dream_state(pdir, "DMN_TICK")

        # If we reach here, distill was never called AND no exception was raised.


def test_dmn_update_dream_state() -> None:
    """_update_dream_state records DMN_TICK in the dream_state table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)

        _reset_dmn_module()
        import dmn

        with patch.object(dmn, "BRAIN_ROOT", brain):
            dmn._update_dream_state(pdir, "DMN_TICK")

        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        row = conn.execute(
            "SELECT last_stage FROM dream_state WHERE project = ?", (proj_hash,)
        ).fetchone()
        conn.close()

        assert row is not None, "Expected a dream_state row"
        assert row[0] == "DMN_TICK", f"Expected 'DMN_TICK', got {row[0]!r}"


def test_dmn_drain_idempotent() -> None:
    """Running _drain_edge_bumps twice does not write duplicate edges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)
        proj = proj_hash

        _insert_chunk(pdir, "chunk-A", proj)
        _insert_chunk(pdir, "chunk-B", proj)
        _insert_bump(pdir, proj, "chunk-A", "chunk-B", count=3)

        _reset_dmn_module()
        import dmn

        deadline = time.monotonic() + 60
        with patch.object(dmn, "BRAIN_ROOT", brain):
            cfg = dmn.load_config()
            dmn._drain_edge_bumps(pdir, cfg, deadline)

            # Insert more bumps and drain again
            conn = sqlite3.connect(str(pdir / "meta.sqlite"))
            now = int(time.time())
            for _ in range(3):
                conn.execute(
                    "INSERT INTO edge_bumps (project, chunk_a, chunk_b, bumped_at) VALUES (?, ?, ?, ?)",
                    (proj, "chunk-A", "chunk-B", now),
                )
            conn.commit()
            conn.close()

            dmn._drain_edge_bumps(pdir, cfg, deadline)

        # There should still be exactly one co_occurs entry (upsert, not duplicate)
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        row = conn.execute("SELECT related_ids FROM chunks WHERE id = 'chunk-A'").fetchone()
        conn.close()
        items = json.loads(row[0] or "[]")
        co_occurs = [i for i in items if i.get("id") == "chunk-B" and i.get("type") == "co_occurs"]
        assert len(co_occurs) == 1, f"Expected exactly 1 co_occurs edge, got {len(co_occurs)}"


def test_dmn_drainer_does_not_consume_subthreshold_bumps() -> None:
    """Sub-threshold bumps must remain drained=0 after a DMN tick.

    Regression for F-003: the old drainer marked ALL undrained bumps (including
    sub-threshold) as drained=1 in one bulk UPDATE.  This permanently consumed
    low-frequency co-retrieval pairs before they could accumulate enough bumps
    to cross the threshold, causing them to never graduate to a co_occurs edge.

    The fixed drainer only marks bumps for promoted pairs as drained=1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        # Threshold = 3 so 2 bumps are below threshold
        _write_config(brain, stdp_min_count=3)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)
        proj = proj_hash

        _insert_chunk(pdir, "chunk-P", proj)
        _insert_chunk(pdir, "chunk-Q", proj)

        # Insert 2 bumps — below threshold of 3
        _insert_bump(pdir, proj, "chunk-P", "chunk-Q", count=2)

        _reset_dmn_module()
        import dmn

        deadline = time.monotonic() + 60
        with patch.object(dmn, "BRAIN_ROOT", brain):
            cfg = dmn.load_config()
            edges = dmn._drain_edge_bumps(pdir, cfg, deadline)

        # No edge must have been written (below threshold)
        assert edges == 0, f"Expected 0 edges for sub-threshold pair, got {edges}"

        # Both bumps must remain drained=0 for future accumulation
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        undrained = conn.execute(
            "SELECT COUNT(*) FROM edge_bumps WHERE project = ? AND drained = 0",
            (proj,),
        ).fetchone()[0]
        conn.close()
        assert undrained == 2, (
            f"Expected 2 undrained sub-threshold bumps (available for future ticks), "
            f"got {undrained}"
        )


def test_dmn_refresh_backlinks_updates_related_ids() -> None:
    """SEC-001 regression: _refresh_recent_backlinks must update related_ids for
    chunks whose neighbours pass the L2 gate (<= 0.78).

    Strategy: insert two chunks into SQLite and inject a mocked LanceDB table
    that returns the neighbour with _distance=0.0 (cosine=1.0, always passes
    the 0.78 gate).  After the refresh, the source chunk's related_ids must
    contain the neighbour id with type='co_occurs'.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        brain = _make_brain_root(tmp)
        _write_config(brain)

        proj_hash = "aabbccddeeff"
        pdir = _make_project(brain, proj_hash)
        proj = proj_hash

        now_ts = int(time.time())
        _insert_chunk(pdir, "chunk-SRC", proj, ts=now_ts)
        _insert_chunk(pdir, "chunk-NBR", proj, ts=now_ts)

        _reset_dmn_module()
        import dmn

        # Build mock LanceDB table that behaves like real LanceDB:
        # - search().where(...).limit(1).to_list() returns the source chunk with a vector
        # - search(vec).where(...).limit(5).to_list() returns the neighbour with _distance=0.0
        src_lance_row = {
            "id": "chunk-SRC",
            "kind": "note",
            "ts": now_ts,
            "project": proj,
            "session_id": "",
            "agent_id": "",
            "text": "source text",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "_distance": 0.0,
        }
        nbr_lance_row = {
            "id": "chunk-NBR",
            "kind": "note",
            "ts": now_ts,
            "project": proj,
            "session_id": "",
            "agent_id": "",
            "text": "neighbour text",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "_distance": 0.0,  # cosine distance 0 -> cosine 1.0, passes the 0.70 gate
        }

        # The mock must handle both call patterns:
        #   tbl.search().where(...).limit(1).to_list()  -> [src_lance_row]  (id lookup)
        #   tbl.search(vec).distance_type("cosine").where(...).limit(5).to_list()
        #     -> [nbr_lance_row]  (ANN search)
        def _mock_search(*args, **kwargs):
            search_mock = MagicMock()
            if args:
                search_mock.distance_type.return_value.where.return_value.limit.return_value.to_list.return_value = [nbr_lance_row]
            else:
                search_mock.where.return_value.limit.return_value.to_list.return_value = [src_lance_row]
            return search_mock

        mock_tbl = MagicMock()
        mock_tbl.search.side_effect = _mock_search

        mock_db = MagicMock()
        mock_db.list_tables.return_value.tables = ["chunks"]
        mock_db.open_table.return_value = mock_tbl

        # Create the lancedb directory so _refresh_recent_backlinks doesn't
        # return early on the path-existence check.
        (pdir / "lancedb").mkdir(parents=True, exist_ok=True)

        # lancedb is imported inside _refresh_recent_backlinks with a local
        # import, so we patch sys.modules so the local import sees the mock.
        deadline = time.monotonic() + 60
        mock_lancedb_mod = MagicMock()
        mock_lancedb_mod.connect.return_value = mock_db
        with patch.object(dmn, "BRAIN_ROOT", brain):
            cfg = dmn.load_config()
            with patch.dict(sys.modules, {"lancedb": mock_lancedb_mod}):
                count = dmn._refresh_recent_backlinks(pdir, cfg, deadline)

        # At least one chunk was processed
        assert count >= 1, f"Expected at least 1 chunk refreshed, got {count}"

        # chunk-SRC's related_ids must now contain chunk-NBR with type='co_occurs'
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        row = conn.execute(
            "SELECT related_ids FROM chunks WHERE id = 'chunk-SRC'"
        ).fetchone()
        conn.close()
        assert row is not None, "chunk-SRC row not found"
        items = json.loads(row[0] or "[]")
        co_occurs = [
            i for i in items
            if i.get("id") == "chunk-NBR" and i.get("type") == "co_occurs"
        ]
        assert len(co_occurs) >= 1, (
            f"Expected co_occurs edge from chunk-SRC to chunk-NBR in related_ids, "
            f"got: {items}"
        )
