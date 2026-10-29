"""Tests for the Decay Timeline panel (V2).

Covers:
  test_timeline_html_renders       – HTML page returns 200 with plot marker
  test_timeline_json_shape         – JSON endpoint returns correct shape
  test_timeline_filters_by_kind    – kind= filter restricts bins
  test_timeline_date_range_window  – days= filter excludes old chunks
  test_timeline_empty_project      – empty project returns zeros
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: project root must precede scripts/ so ui/ resolves correctly.
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
# Seeding helpers
# ---------------------------------------------------------------------------

def _make_project(brain_root: Path, cwd_name: str = "testproject") -> tuple[str, Path]:
    """Create a 12-hex project dir under brain_root and return (hash, dir)."""
    fake_cwd = brain_root.parent / cwd_name
    project_hash = hashlib.sha256(str(fake_cwd.resolve()).encode()).hexdigest()[:12]
    pdir = brain_root / project_hash
    pdir.mkdir(parents=True, exist_ok=True)
    return project_hash, pdir


def _init_db(pdir: Path) -> sqlite3.Connection:
    """Create meta.sqlite with correct schema in pdir."""
    db_path = pdir / "meta.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            session_id TEXT,
            agent_id TEXT,
            kind TEXT NOT NULL,
            ts INTEGER NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            last_accessed INTEGER,
            raw_ref TEXT,
            decay_score REAL NOT NULL DEFAULT 1.0,
            related_ids TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.commit()
    return conn


def _insert_chunk(
    conn: sqlite3.Connection,
    kind: str = "stop",
    ts: int | None = None,
    pinned: int = 0,
    superseded_by: str = "",
    decay_score: float = 1.0,
) -> str:
    cid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO chunks (id, project, kind, ts, content_hash, superseded_by, pinned, decay_score) "
        "VALUES (?, 'test', ?, ?, '', ?, ?, ?)",
        (cid, kind, ts or int(time.time()), superseded_by, pinned, decay_score),
    )
    conn.commit()
    return cid


# ---------------------------------------------------------------------------
# Shared async helper: build ASGI client with patched BRAIN_ROOT
# ---------------------------------------------------------------------------

def _make_client(brain_root: Path):
    """Return a context manager that yields an httpx.AsyncClient against the app."""
    import os

    async def _ctx():
        # Patch BRAIN_ROOT in every relevant module before importing the app
        import importlib
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("common", "ui.data", "ui.server", "ui.routers.timeline"):
                del sys.modules[mod_name]

        os.environ["_TEST_BRAIN_ROOT"] = str(brain_root)

        # Monkey-patch BRAIN_ROOT after import
        import common
        common.BRAIN_ROOT = brain_root

        import ui.data as _data
        _data.BRAIN_ROOT = brain_root

        # Re-import validate_project / open_meta_ro to use patched BRAIN_ROOT
        from ui.data import validate_project as _vp, open_meta_ro as _omr

        def _validate_project_patched(project_hash: str) -> Path:
            from fastapi import HTTPException
            pdir = brain_root / project_hash
            if not pdir.is_dir():
                raise HTTPException(status_code=404, detail=f"Unknown project hash: {project_hash}")
            return pdir

        def _open_meta_ro_patched(project_hash: str) -> sqlite3.Connection:
            from fastapi import HTTPException
            sqlite_path = brain_root / project_hash / "meta.sqlite"
            if not sqlite_path.exists():
                raise HTTPException(status_code=404, detail="meta.sqlite not found")
            conn = sqlite3.connect(
                f"file:{sqlite_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            return conn

        # Patch in timeline router module
        import ui.routers.timeline as _tl
        _tl.validate_project = _validate_project_patched
        _tl.open_meta_ro = _open_meta_ro_patched

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client

    return _ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_timeline_html_renders(tmp_path: Path) -> None:
    """GET /timeline returns 200, contains a Plot marker, contains project hash."""
    brain_root = tmp_path / "brain"
    project_hash, pdir = _make_project(brain_root)
    conn = _init_db(pdir)
    for _ in range(5):
        _insert_chunk(conn)
    conn.close()

    async def _run() -> None:
        import os
        import importlib
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("common", "ui.data", "ui.server", "ui.routers.timeline"):
                del sys.modules[mod_name]

        import common
        common.BRAIN_ROOT = brain_root
        import ui.data as _data
        _data.BRAIN_ROOT = brain_root

        def _validate(ph):
            from fastapi import HTTPException
            pdir = brain_root / ph
            if not pdir.is_dir():
                raise HTTPException(status_code=404, detail="Unknown project")
            return pdir

        def _open_ro(ph):
            from fastapi import HTTPException
            sp = brain_root / ph / "meta.sqlite"
            if not sp.exists():
                raise HTTPException(status_code=404, detail="meta.sqlite not found")
            c = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
            c.row_factory = sqlite3.Row
            return c

        import ui.routers.timeline as _tl
        _tl.validate_project = _validate
        _tl.open_meta_ro = _open_ro

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/timeline?project={project_hash}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
            body = resp.text
            assert project_hash in body, "Response should contain project hash"
            # Should contain Plot CDN script or chart container
            assert "Plot" in body or "chart-bars" in body, "Response should reference Observable Plot"
            assert "Traceback" not in body and "traceback" not in body, "No traceback in response"

    anyio.run(_run)


def test_timeline_json_shape(tmp_path: Path) -> None:
    """GET /api/timeline.json returns correct JSON shape."""
    brain_root = tmp_path / "brain"
    project_hash, pdir = _make_project(brain_root, "testproject2")
    conn = _init_db(pdir)
    now = int(time.time())
    for i in range(4):
        _insert_chunk(conn, ts=now - i * 86400)
    conn.close()

    async def _run() -> None:
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("common", "ui.data", "ui.server", "ui.routers.timeline"):
                del sys.modules[mod_name]

        import common
        common.BRAIN_ROOT = brain_root
        import ui.data as _data
        _data.BRAIN_ROOT = brain_root

        def _validate(ph):
            from fastapi import HTTPException
            d = brain_root / ph
            if not d.is_dir():
                raise HTTPException(status_code=404, detail="Unknown project")
            return d

        def _open_ro(ph):
            from fastapi import HTTPException
            sp = brain_root / ph / "meta.sqlite"
            if not sp.exists():
                raise HTTPException(status_code=404, detail="meta.sqlite not found")
            c = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
            c.row_factory = sqlite3.Row
            return c

        import ui.routers.timeline as _tl
        _tl.validate_project = _validate
        _tl.open_meta_ro = _open_ro

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/timeline.json?project={project_hash}&days=30")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
            data = resp.json()
            assert "bins" in data, f"Missing 'bins' key: {data}"
            assert "summary" in data, f"Missing 'summary' key: {data}"
            assert "kinds_present" in data, f"Missing 'kinds_present' key: {data}"
            assert isinstance(data["summary"]["total_chunks"], int), (
                f"total_chunks must be int, got {type(data['summary']['total_chunks'])}"
            )

    anyio.run(_run)


def test_timeline_filters_by_kind(tmp_path: Path) -> None:
    """?kind=decision returns only decision bins."""
    brain_root = tmp_path / "brain"
    project_hash, pdir = _make_project(brain_root, "testproject3")
    conn = _init_db(pdir)
    now = int(time.time())
    for _ in range(5):
        _insert_chunk(conn, kind="stop", ts=now)
    for _ in range(3):
        _insert_chunk(conn, kind="decision", ts=now)
    conn.close()

    async def _run() -> None:
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("common", "ui.data", "ui.server", "ui.routers.timeline"):
                del sys.modules[mod_name]

        import common
        common.BRAIN_ROOT = brain_root
        import ui.data as _data
        _data.BRAIN_ROOT = brain_root

        def _validate(ph):
            from fastapi import HTTPException
            d = brain_root / ph
            if not d.is_dir():
                raise HTTPException(status_code=404, detail="Unknown project")
            return d

        def _open_ro(ph):
            from fastapi import HTTPException
            sp = brain_root / ph / "meta.sqlite"
            if not sp.exists():
                raise HTTPException(status_code=404, detail="meta.sqlite not found")
            c = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
            c.row_factory = sqlite3.Row
            return c

        import ui.routers.timeline as _tl
        _tl.validate_project = _validate
        _tl.open_meta_ro = _open_ro

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/timeline.json?project={project_hash}&kind=decision")
            assert resp.status_code == 200
            data = resp.json()
            kinds_in_bins = {b["kind"] for b in data["bins"]}
            assert kinds_in_bins <= {"decision"}, (
                f"bins should only contain 'decision', got {kinds_in_bins}"
            )

    anyio.run(_run)


def test_timeline_date_range_window(tmp_path: Path) -> None:
    """?days=7 excludes chunks older than 7 days."""
    brain_root = tmp_path / "brain"
    project_hash, pdir = _make_project(brain_root, "testproject4")
    conn = _init_db(pdir)
    now = int(time.time())
    # Recent: within 7 days
    for i in range(3):
        _insert_chunk(conn, kind="stop", ts=now - i * 86400)
    # Old: 30 days ago
    for _ in range(4):
        _insert_chunk(conn, kind="stop", ts=now - 30 * 86400)
    conn.close()

    async def _run() -> None:
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("common", "ui.data", "ui.server", "ui.routers.timeline"):
                del sys.modules[mod_name]

        import common
        common.BRAIN_ROOT = brain_root
        import ui.data as _data
        _data.BRAIN_ROOT = brain_root

        def _validate(ph):
            from fastapi import HTTPException
            d = brain_root / ph
            if not d.is_dir():
                raise HTTPException(status_code=404, detail="Unknown project")
            return d

        def _open_ro(ph):
            from fastapi import HTTPException
            sp = brain_root / ph / "meta.sqlite"
            if not sp.exists():
                raise HTTPException(status_code=404, detail="meta.sqlite not found")
            c = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
            c.row_factory = sqlite3.Row
            return c

        import ui.routers.timeline as _tl
        _tl.validate_project = _validate
        _tl.open_meta_ro = _open_ro

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # days=7: should see only ~3 recent chunks
            resp7 = await client.get(f"/api/timeline.json?project={project_hash}&days=7")
            assert resp7.status_code == 200
            data7 = resp7.json()
            count7 = sum(b["count"] for b in data7["bins"])
            assert count7 == 3, f"Expected 3 recent chunks in 7d window, got {count7}"

            # days=90: should see all 7 chunks
            resp90 = await client.get(f"/api/timeline.json?project={project_hash}&days=90")
            assert resp90.status_code == 200
            data90 = resp90.json()
            count90 = sum(b["count"] for b in data90["bins"])
            assert count90 == 7, f"Expected 7 total chunks in 90d window, got {count90}"

    anyio.run(_run)


def test_timeline_empty_project(tmp_path: Path) -> None:
    """Empty project returns bins=[] and total_chunks=0."""
    brain_root = tmp_path / "brain"
    project_hash, pdir = _make_project(brain_root, "testproject5")
    _init_db(pdir).close()  # create schema but insert nothing

    async def _run() -> None:
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("common", "ui.data", "ui.server", "ui.routers.timeline"):
                del sys.modules[mod_name]

        import common
        common.BRAIN_ROOT = brain_root
        import ui.data as _data
        _data.BRAIN_ROOT = brain_root

        def _validate(ph):
            from fastapi import HTTPException
            d = brain_root / ph
            if not d.is_dir():
                raise HTTPException(status_code=404, detail="Unknown project")
            return d

        def _open_ro(ph):
            from fastapi import HTTPException
            sp = brain_root / ph / "meta.sqlite"
            if not sp.exists():
                raise HTTPException(status_code=404, detail="meta.sqlite not found")
            c = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
            c.row_factory = sqlite3.Row
            return c

        import ui.routers.timeline as _tl
        _tl.validate_project = _validate
        _tl.open_meta_ro = _open_ro

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/timeline.json?project={project_hash}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["bins"] == [], f"Expected empty bins, got {data['bins']}"
            assert data["summary"]["total_chunks"] == 0, (
                f"Expected total_chunks=0, got {data['summary']['total_chunks']}"
            )

    anyio.run(_run)
