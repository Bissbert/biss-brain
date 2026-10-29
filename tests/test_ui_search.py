"""Tests for the search panel (V4): GET /search, POST /api/pin, /api/unpin, /api/forget.

Follows the pattern from test_ui_foundation.py:
  - Uses httpx.ASGITransport + anyio.run for async client calls
  - Overrides BRAIN_ROOT via a temp directory
  - Creates minimal SQLite + LanceDB fixtures for write-endpoint tests
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path

# Ensure project root on path (ui/ package takes precedence over scripts/ui.py)
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

def _make_fake_project(brain_root: Path) -> tuple[str, Path]:
    """Create a minimal project dir with meta.sqlite. Returns (hash, project_dir)."""
    fake_hash = "aabb11223344"
    pdir = brain_root / fake_hash
    pdir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            session_id TEXT DEFAULT '',
            agent_id TEXT DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'note',
            ts INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT NOT NULL DEFAULT '',
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            last_accessed INTEGER,
            decay_score REAL NOT NULL DEFAULT 1.0,
            related_ids TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.commit()
    conn.close()
    return fake_hash, pdir


def _insert_chunk(pdir: Path, chunk_id: str | None = None) -> str:
    """Insert one test chunk into meta.sqlite and return its id."""
    cid = chunk_id or uuid.uuid4().hex
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(
        "INSERT INTO chunks (id, project, kind, ts, content_hash) VALUES (?,?,?,?,?)",
        (cid, pdir.name, "note", 1700000000, "deadbeef"),
    )
    conn.commit()
    conn.close()
    return cid


# ---------------------------------------------------------------------------
# Test 1: empty query renders the form
# ---------------------------------------------------------------------------

def test_search_empty_query(tmp_path: Path) -> None:
    """GET /search?project=<hash> with no q returns 200 with the search form."""
    brain_root = tmp_path / "brain"
    fake_hash, _ = _make_fake_project(brain_root)

    os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"

    import common
    import ui.data as _data
    from ui.server import app

    _orig_common = common.BRAIN_ROOT
    _orig_data = _data.BRAIN_ROOT
    common.BRAIN_ROOT = brain_root
    _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]
    try:
        async def _run() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(f"/search?project={fake_hash}")

        resp = anyio.run(_run)
    finally:
        common.BRAIN_ROOT = _orig_common
        _data.BRAIN_ROOT = _orig_data  # type: ignore[attr-defined]

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    body = resp.text
    assert "<form" in body, "Response should contain a <form>"
    assert 'name="q"' in body, "Form should contain query input"


# ---------------------------------------------------------------------------
# Test 2: unknown project hash returns 404
# ---------------------------------------------------------------------------

def test_search_validates_project(tmp_path: Path) -> None:
    """GET /search?project=does-not-exist&q=foo returns 404."""
    brain_root = tmp_path / "brain"
    brain_root.mkdir(parents=True, exist_ok=True)

    os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"

    import common
    import ui.data as _data
    from ui.server import app

    _orig_common = common.BRAIN_ROOT
    _orig_data = _data.BRAIN_ROOT
    common.BRAIN_ROOT = brain_root
    _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]
    try:
        async def _run() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/search?project=doesnotexist&q=foo")

        resp = anyio.run(_run)
    finally:
        common.BRAIN_ROOT = _orig_common
        _data.BRAIN_ROOT = _orig_data  # type: ignore[attr-defined]

    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"


# ---------------------------------------------------------------------------
# Test 3: pin / unpin cycle flips the pinned flag
# ---------------------------------------------------------------------------

def test_pin_unpin_cycle(tmp_path: Path) -> None:
    """POST /api/pin then /api/unpin toggles the pinned flag in SQLite."""
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_fake_project(brain_root)
    chunk_id = _insert_chunk(pdir)

    os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"

    import common
    import ui.data as _data
    from ui.server import app

    _orig_common = common.BRAIN_ROOT
    _orig_data = _data.BRAIN_ROOT
    common.BRAIN_ROOT = brain_root
    _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]

    def _read_pinned() -> int:
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        row = conn.execute("SELECT pinned FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        conn.close()
        return row[0] if row else -1

    assert _read_pinned() == 0, "Chunk should start unpinned"

    try:
        async def _run_pin() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(f"/api/pin?project={fake_hash}&id={chunk_id}")

        resp = anyio.run(_run_pin)
        assert resp.status_code == 200, f"pin returned {resp.status_code}: {resp.text[:300]}"
        assert _read_pinned() == 1, "Chunk should be pinned after /api/pin"

        async def _run_unpin() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(f"/api/unpin?project={fake_hash}&id={chunk_id}")

        resp2 = anyio.run(_run_unpin)
        assert resp2.status_code == 200, f"unpin returned {resp2.status_code}: {resp2.text[:300]}"
        assert _read_pinned() == 0, "Chunk should be unpinned after /api/unpin"
    finally:
        common.BRAIN_ROOT = _orig_common
        _data.BRAIN_ROOT = _orig_data  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test 4: forget marks superseded_by as non-empty
# ---------------------------------------------------------------------------

def test_forget_marks_superseded(tmp_path: Path) -> None:
    """POST /api/forget sets a tombstone in superseded_by."""
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_fake_project(brain_root)
    chunk_id = _insert_chunk(pdir)

    os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"

    import common
    import ui.data as _data
    from ui.server import app

    _orig_common = common.BRAIN_ROOT
    _orig_data = _data.BRAIN_ROOT
    common.BRAIN_ROOT = brain_root
    _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]

    def _read_superseded() -> str:
        conn = sqlite3.connect(str(pdir / "meta.sqlite"))
        row = conn.execute("SELECT superseded_by FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        conn.close()
        return row[0] if row else ""

    assert _read_superseded() == "", "Chunk should start with empty superseded_by"

    try:
        async def _run_forget() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(f"/api/forget?project={fake_hash}&id={chunk_id}")

        resp = anyio.run(_run_forget)
        assert resp.status_code == 200, f"forget returned {resp.status_code}: {resp.text[:300]}"
        superseded = _read_superseded()
        assert superseded != "", f"superseded_by should be set, got: {superseded!r}"
    finally:
        common.BRAIN_ROOT = _orig_common
        _data.BRAIN_ROOT = _orig_data  # type: ignore[attr-defined]
