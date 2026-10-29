"""Tests for GET /api/neighbours endpoint.

Verifies:
- 404 for unknown project
- 200 with empty list for project with no matching neighbours
- Response HTML shape matches _neighbours.html expectations
- fetch_neighbours can be mocked to return specific shapes
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

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

os.environ.setdefault("BISS_BRAIN_FAKE_EMBED", "1")


def _make_fake_project(brain_root: Path) -> tuple[str, Path]:
    """Create a minimal project dir with meta.sqlite. Returns (hash, project_dir)."""
    fake_hash = "cc1122334455"
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
    conn.execute(
        "INSERT INTO chunks (id, project, kind, ts, content_hash) VALUES (?,?,?,?,?)",
        ("testchunk001", fake_hash, "note", 1700000000, "deadbeef"),
    )
    conn.commit()
    conn.close()
    return fake_hash, pdir


def _patch_modules(brain_root: Path) -> None:
    """Re-import core modules with brain_root patched to avoid cross-test pollution."""
    for mod_name in list(sys.modules.keys()):
        if mod_name == "common" or mod_name == "ui" or mod_name.startswith("ui."):
            del sys.modules[mod_name]

    import common
    common.BRAIN_ROOT = brain_root

    import ui.data as _data
    _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test 1: unknown project returns 404
# ---------------------------------------------------------------------------

def test_neighbours_unknown_project(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    brain_root.mkdir(parents=True, exist_ok=True)

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/api/neighbours?project=doesnotexist&id=abc")

    resp = anyio.run(_run)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 2: valid project, no neighbours (mocked fetch_neighbours returns [])
# ---------------------------------------------------------------------------

def test_neighbours_empty(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_fake_project(brain_root)

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        import ui.routers.search as _search
        with patch.object(_search, "fetch_neighbours", return_value=[]):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(
                    f"/api/neighbours?project={fake_hash}&id=testchunk001"
                )

    resp = anyio.run(_run)
    assert resp.status_code == 200
    assert "no neighbours found" in resp.text


# ---------------------------------------------------------------------------
# Test 3: valid project, mocked fetch_neighbours returns 2 neighbours
# ---------------------------------------------------------------------------

def test_neighbours_with_results(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_fake_project(brain_root)

    fake_neighbours = [
        {
            "id": "nbr00000001",
            "kind": "decision",
            "text": "A related decision chunk with some content here.",
            "ts": 1700001000,
            "score": 0.12,
        },
        {
            "id": "nbr00000002",
            "kind": "note",
            "text": "Another related note chunk.",
            "ts": 1700002000,
            "score": None,
        },
    ]

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        import ui.routers.search as _search
        with patch.object(_search, "fetch_neighbours", return_value=fake_neighbours):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(
                    f"/api/neighbours?project={fake_hash}&id=testchunk001&k=2"
                )

    resp = anyio.run(_run)
    assert resp.status_code == 200
    html = resp.text
    # Should contain neighbour ids (first 12 chars)
    assert "nbr00000001" in html, "First neighbour id should appear"
    assert "nbr00000002" in html, "Second neighbour id should appear"
    # Should contain kind badges
    assert "decision" in html
    assert "note" in html
    # Score for first neighbour
    assert "0.120" in html or "0.12" in html
    # Text preview (80 char truncation)
    assert "A related decision chunk" in html


# ---------------------------------------------------------------------------
# Test 4: k parameter is respected (mock verifies call arg)
# ---------------------------------------------------------------------------

def test_neighbours_k_parameter(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_fake_project(brain_root)

    calls = []

    def _mock_fetch(project, chunk_id, k=5):
        calls.append({"project": project, "chunk_id": chunk_id, "k": k})
        return []

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        import ui.routers.search as _search
        with patch.object(_search, "fetch_neighbours", side_effect=_mock_fetch):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(
                    f"/api/neighbours?project={fake_hash}&id=testchunk001&k=10"
                )

    resp = anyio.run(_run)
    assert resp.status_code == 200
    assert calls, "fetch_neighbours should have been called"
    assert calls[0]["k"] == 10, f"Expected k=10, got {calls[0]['k']}"
