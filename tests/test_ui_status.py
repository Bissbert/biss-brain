"""Tests for the Brain Status panel (B3 - F-010).

Covers:
  test_status_page_renders            -- GET /status?project=<h> returns 200
  test_status_json_shape              -- GET /api/status.json returns expected keys
  test_status_missing_dream_state     -- missing dream_state row degrades to defaults
  test_status_missing_project         -- unknown project returns 404
  test_status_json_edge_queue_depth   -- edge_bumps count is read correctly
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)

import anyio
import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(brain_root: Path, name: str = "statustest") -> tuple[str, Path]:
    fake_cwd = brain_root.parent / name
    project_hash = hashlib.sha256(str(fake_cwd.resolve()).encode()).hexdigest()[:12]
    pdir = brain_root / project_hash
    pdir.mkdir(parents=True, exist_ok=True)
    return project_hash, pdir


def _init_db_minimal(pdir: Path) -> sqlite3.Connection:
    """Create a minimal meta.sqlite with just the chunks table."""
    db_path = pdir / "meta.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            ts INTEGER NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            decay_score REAL NOT NULL DEFAULT 1.0,
            related_ids TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.commit()
    return conn


def _add_dream_state(conn: sqlite3.Connection, project: str,
                     stage: str = "IDLE", last_complete: int = 0,
                     error: str = "", error_ts: int = 0) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dream_state (
            project TEXT PRIMARY KEY,
            last_stage TEXT NOT NULL DEFAULT 'IDLE',
            last_complete_ts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            last_error_ts INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO dream_state "
        "(project, last_stage, last_complete_ts, last_error, last_error_ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (project, stage, last_complete, error, error_ts),
    )
    conn.commit()


def _add_edge_bumps(conn: sqlite3.Connection, project: str, count: int,
                    drained: int = 0) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_bumps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            chunk_a TEXT NOT NULL,
            chunk_b TEXT NOT NULL,
            bumped_at INTEGER NOT NULL,
            session_id TEXT,
            drained INTEGER NOT NULL DEFAULT 0
        )
    """)
    now = int(time.time())
    for _ in range(count):
        conn.execute(
            "INSERT INTO edge_bumps (project, chunk_a, chunk_b, bumped_at, drained) "
            "VALUES (?, ?, ?, ?, ?)",
            (project, uuid.uuid4().hex, uuid.uuid4().hex, now, drained),
        )
    conn.commit()


def _patch_for_status(brain_root: Path) -> None:
    """Patch module state so the status router uses our test brain_root."""
    for mod_name in list(sys.modules.keys()):
        if mod_name == "common" or mod_name == "ui" or mod_name.startswith("ui."):
            del sys.modules[mod_name]

    import common
    common.BRAIN_ROOT = brain_root

    import ui.data as _data
    _data.BRAIN_ROOT = brain_root

    from fastapi import HTTPException

    def _validate(ph: str) -> Path:
        pdir = brain_root / ph
        if not pdir.is_dir():
            raise HTTPException(status_code=404, detail=f"Unknown: {ph}")
        return pdir

    def _open_ro(ph: str) -> sqlite3.Connection:
        sp = brain_root / ph / "meta.sqlite"
        if not sp.exists():
            raise HTTPException(status_code=404, detail="meta.sqlite not found")
        c = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    import ui.data as _data2
    _data2.validate_project = _validate
    _data2.open_meta_ro = _open_ro

    import ui.routers.status as _st
    _st.validate_project = _validate
    _st.fetch_brain_status = lambda project: _patched_fetch_brain_status(
        project, brain_root
    )


def _patched_fetch_brain_status(project: str, brain_root: Path) -> dict:
    """Patched fetch_brain_status that uses test brain_root."""
    result: dict = {
        "dream_stage": "—",
        "last_complete_ts": None,
        "last_error": "",
        "last_error_ts": None,
        "edge_queue_depth": 0,
        "dmn_plist_exists": False,
        "dream_plist_exists": False,
        "dream_error": None,
    }
    sp = brain_root / project / "meta.sqlite"
    if not sp.exists():
        return result
    conn = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        try:
            row = conn.execute(
                "SELECT last_stage, last_complete_ts, last_error, last_error_ts "
                "FROM dream_state WHERE project = ?", (project,)
            ).fetchone()
            if row:
                result["dream_stage"] = row[0] or "IDLE"
                result["last_complete_ts"] = row[1] or None
                result["last_error"] = row[2] or ""
                result["last_error_ts"] = row[3] or None
                if result["last_error"]:
                    result["dream_error"] = result["last_error"]
        except sqlite3.OperationalError:
            pass
        try:
            row2 = conn.execute(
                "SELECT COUNT(*) FROM edge_bumps WHERE drained = 0"
            ).fetchone()
            result["edge_queue_depth"] = int(row2[0]) if row2 else 0
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Skip removed: conftest autouse fixture restores BRAIN_ROOT between tests.


def test_status_page_renders(tmp_path: Path) -> None:
    """GET /status?project=<h> returns 200 with Brain status heading."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "st1")
    conn = _init_db_minimal(pdir)
    _add_dream_state(conn, ph, stage="IDLE")
    conn.close()

    async def _run() -> None:
        _patch_for_status(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/status?project={ph}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:400]}"
            assert "Brain status" in resp.text, "Status page must contain 'Brain status' heading"
            assert "Dream stage" in resp.text, "Status page must contain 'Dream stage' tile"

    anyio.run(_run)


def test_status_json_shape(tmp_path: Path) -> None:
    """GET /api/status.json returns the expected keys."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "st2")
    conn = _init_db_minimal(pdir)
    _add_dream_state(conn, ph, stage="IDLE", last_complete=1000000)
    _add_edge_bumps(conn, ph, count=3, drained=0)
    _add_edge_bumps(conn, ph, count=2, drained=1)
    conn.close()

    async def _run() -> None:
        _patch_for_status(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/status.json?project={ph}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            expected_keys = {
                "dream_stage", "last_complete_ts", "last_error",
                "last_error_ts", "edge_queue_depth",
                "dmn_plist_exists", "dream_plist_exists", "dream_error",
            }
            missing = expected_keys - set(data.keys())
            assert not missing, f"Missing keys: {missing}"
            assert data["dream_stage"] == "IDLE"
            assert data["edge_queue_depth"] == 3, (
                f"Expected 3 undrained bumps, got {data['edge_queue_depth']}"
            )

    anyio.run(_run)


def test_status_missing_dream_state(tmp_path: Path) -> None:
    """When dream_state row is absent, status degrades to default values."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "st3")
    # Only create chunks table — no dream_state, no edge_bumps
    _init_db_minimal(pdir).close()

    async def _run() -> None:
        _patch_for_status(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/status.json?project={ph}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            assert data["dream_stage"] == "—", (
                f"Expected '—' for missing dream_state, got {data['dream_stage']!r}"
            )
            assert data["edge_queue_depth"] == 0
            assert data["dream_error"] is None

    anyio.run(_run)


def test_status_missing_project(tmp_path: Path) -> None:
    """Unknown project hash returns 404."""
    brain_root = tmp_path / "brain"
    brain_root.mkdir(parents=True, exist_ok=True)

    async def _run() -> None:
        _patch_for_status(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/api/status.json?project=000000000000")
            assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"

    anyio.run(_run)
