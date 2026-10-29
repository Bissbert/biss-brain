"""Tests for the Sessions panel (V5).

1. test_sessions_index_empty          – fresh project dir → 200 + empty-state text
2. test_sessions_aggregation          – seed 2 sessions, assert both listed
3. test_session_detail_orders_by_ts   – chunks rendered in ascending ts order
4. test_session_detail_supersession   – chunk with superseded_by renders strike-through class
5. test_recent_sessions_json          – /api/sessions/recent returns correct JSON list
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Ensure the project root precedes scripts/ so the ui package is importable.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)

import anyio
import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain_root(tmp_path: Path) -> Path:
    brain = tmp_path / "brain"
    brain.mkdir()
    return brain


def _project_hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()[:12]


def _create_project(brain_root: Path, label: str) -> tuple[str, Path]:
    """Create a project directory with meta.sqlite and return (hash, pdir)."""
    phash = _project_hash(label)
    pdir = brain_root / phash
    pdir.mkdir(parents=True)
    return phash, pdir


def _init_sqlite(pdir: Path) -> sqlite3.Connection:
    """Open (or create) meta.sqlite with the standard schema."""
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'note',
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


def _insert_chunk(
    conn: sqlite3.Connection,
    *,
    cid: str,
    session_id: str = "sess-a",
    agent_id: str = "",
    kind: str = "note",
    ts: int,
    superseded_by: str = "",
    pinned: int = 0,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO chunks
           (id, project, session_id, agent_id, kind, ts, content_hash,
            superseded_by, pinned, retrieval_count, decay_score, related_ids)
           VALUES (?,?,?,?,?,?,?,?,?,0,1.0,'')""",
        (cid, "proj", session_id, agent_id, kind, ts, cid, superseded_by, pinned),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixtures: BRAIN_ROOT override
# ---------------------------------------------------------------------------

def _app_with_brain(brain_root: Path):
    """Return the FastAPI app with BRAIN_ROOT patched to brain_root."""
    # We need to reload modules that cache BRAIN_ROOT at import time.
    import importlib

    for mod in list(sys.modules.keys()):
        if mod in ("common", "ui.data", "ui.server", "ui.routers.sessions"):
            del sys.modules[mod]

    with patch.dict("os.environ", {"BISS_BRAIN_ROOT": str(brain_root)}):
        # Patch common.BRAIN_ROOT before the server imports it
        import common as _common
        _common.BRAIN_ROOT = brain_root

        import ui.data as _data
        _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]
        # Patch the HTTPException-raising helpers to use new root
        from pathlib import Path as _Path

        def _validate_project(project_hash: str) -> _Path:
            pdir = brain_root / project_hash
            from fastapi import HTTPException
            if not pdir.is_dir():
                raise HTTPException(status_code=404, detail=f"Unknown project hash: {project_hash}")
            return pdir

        def _open_meta_ro(project_hash: str) -> sqlite3.Connection:
            pdir = brain_root / project_hash
            from fastapi import HTTPException
            sqlite_path = pdir / "meta.sqlite"
            if not sqlite_path.exists():
                raise HTTPException(status_code=404, detail="meta.sqlite not found")
            conn = sqlite3.connect(
                f"file:{sqlite_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            return conn

        _data.validate_project = _validate_project  # type: ignore[attr-defined]
        _data.open_meta_ro = _open_meta_ro  # type: ignore[attr-defined]

        # Patch open_lance to always return None (no LanceDB in tests)
        _data.open_lance = lambda _h: None  # type: ignore[attr-defined]
        # fetch_chunk_texts with None table returns {} — that is the default behaviour

        # Reload router so it picks up patched data helpers
        if "ui.routers.sessions" in sys.modules:
            del sys.modules["ui.routers.sessions"]
        import ui.routers.sessions as _sess_router
        # Re-register patched helpers in the router module namespace
        _sess_router.validate_project = _validate_project  # type: ignore[attr-defined]
        _sess_router.open_meta_ro = _open_meta_ro  # type: ignore[attr-defined]
        _sess_router.open_lance = lambda _h: None  # type: ignore[attr-defined]

        # Build a minimal app with just the sessions router
        from fastapi import FastAPI
        from fastapi.templating import Jinja2Templates

        mini = FastAPI()
        # Share the templates instance from the router (already configured with filters)
        mini.include_router(_sess_router.router)

        return mini


# ---------------------------------------------------------------------------
# Test 1: empty project → empty-state message
# ---------------------------------------------------------------------------

def test_sessions_index_empty(tmp_path: Path) -> None:
    brain = _make_brain_root(tmp_path)
    phash, pdir = _create_project(brain, "test-empty")
    # Create a meta.sqlite with no rows
    conn = _init_sqlite(pdir)
    conn.close()

    app = _app_with_brain(brain)

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/sessions?project={phash}")

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    assert "No sessions captured yet" in resp.text, (
        f"Empty-state text not found in response: {resp.text[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 2: two sessions are listed
# ---------------------------------------------------------------------------

def test_sessions_aggregation(tmp_path: Path) -> None:
    brain = _make_brain_root(tmp_path)
    phash, pdir = _create_project(brain, "test-agg")
    conn = _init_sqlite(pdir)

    now = int(time.time())
    # Session A — 3 chunks
    for i in range(3):
        _insert_chunk(conn, cid=f"aaaa{i:04d}", session_id="sess-alpha", ts=now - 200 + i)
    # Session B — 3 chunks
    for i in range(3):
        _insert_chunk(conn, cid=f"bbbb{i:04d}", session_id="sess-beta", ts=now - 100 + i)
    conn.close()

    app = _app_with_brain(brain)

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/sessions?project={phash}&limit=50")

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    body = resp.text
    assert "sess-alp" in body, "Session alpha short id not found"
    assert "sess-bet" in body, "Session beta short id not found"


# ---------------------------------------------------------------------------
# Test 3: session detail renders chunks in ascending ts order
# ---------------------------------------------------------------------------

def test_session_detail_orders_by_ts(tmp_path: Path) -> None:
    brain = _make_brain_root(tmp_path)
    phash, pdir = _create_project(brain, "test-order")
    conn = _init_sqlite(pdir)

    now = int(time.time())
    # Insert in reverse order to ensure the query re-orders them
    _insert_chunk(conn, cid="chunk-c", session_id="sess-ord", ts=now - 10)
    _insert_chunk(conn, cid="chunk-a", session_id="sess-ord", ts=now - 30)
    _insert_chunk(conn, cid="chunk-b", session_id="sess-ord", ts=now - 20)
    conn.close()

    app = _app_with_brain(brain)

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/session/sess-ord?project={phash}")

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    body = resp.text
    # Use the unique `id="c-<cid>"` anchor injected into each card to determine render order.
    pos_a = body.find('id="c-chunk-a"')
    pos_b = body.find('id="c-chunk-b"')
    pos_c = body.find('id="c-chunk-c"')
    assert pos_a != -1 and pos_b != -1 and pos_c != -1, (
        f"Not all chunk anchors found. a={pos_a} b={pos_b} c={pos_c}"
    )
    assert pos_a < pos_b < pos_c, (
        f"Chunks not in ascending ts order: a={pos_a} b={pos_b} c={pos_c}"
    )


# ---------------------------------------------------------------------------
# Test 4: superseded chunk renders with strike-through class
# ---------------------------------------------------------------------------

def test_session_detail_supersession_marker(tmp_path: Path) -> None:
    brain = _make_brain_root(tmp_path)
    phash, pdir = _create_project(brain, "test-super")
    conn = _init_sqlite(pdir)

    now = int(time.time())
    _insert_chunk(conn, cid="chunk-orig", session_id="sess-sup", ts=now - 20,
                  superseded_by="chunk-new")
    _insert_chunk(conn, cid="chunk-new",  session_id="sess-sup", ts=now - 10)
    conn.close()

    app = _app_with_brain(brain)

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/session/sess-sup?project={phash}")

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    body = resp.text
    assert "text-decoration:line-through" in body, (
        "Strike-through style not found for superseded chunk"
    )


# ---------------------------------------------------------------------------
# Test 5: /api/sessions/recent returns JSON list
# ---------------------------------------------------------------------------

def test_recent_sessions_json(tmp_path: Path) -> None:
    brain = _make_brain_root(tmp_path)
    phash, pdir = _create_project(brain, "test-recent")
    conn = _init_sqlite(pdir)

    now = int(time.time())
    for i in range(3):
        _insert_chunk(conn, cid=f"r{i:04d}sess1", session_id="sess-r1", ts=now - 300 + i)
    for i in range(2):
        _insert_chunk(conn, cid=f"r{i:04d}sess2", session_id="sess-r2", ts=now - 100 + i)
    conn.close()

    app = _app_with_brain(brain)

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/api/sessions/recent?project={phash}&limit=2")

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) == 2, f"Expected 2 items (limit=2), got {len(data)}"
    for item in data:
        assert "session_id" in item, f"Missing session_id key: {item}"
        assert "chunk_count" in item, f"Missing chunk_count key: {item}"
        assert "ended_ts" in item, f"Missing ended_ts key: {item}"
