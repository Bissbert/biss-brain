"""Tests for POST /api/compose endpoint.

Verifies:
- 404 for unknown project
- 400 for empty or too-large ids list
- 200 with markdown content using _format_chunks
- Compose drawer HTML shape matches _compose_drawer.html
- Markdown contains chunk content for both submitted ids
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


def _make_project_with_chunks(brain_root: Path) -> tuple[str, Path, list[str]]:
    """Create a project with 2 test chunks. Returns (hash, pdir, [id1, id2])."""
    fake_hash = "dd2233445566"
    pdir = brain_root / fake_hash
    pdir.mkdir(parents=True, exist_ok=True)

    id1 = "compose0001111"
    id2 = "compose0002222"

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
        "INSERT INTO chunks (id, project, kind, ts, content_hash, session_id) VALUES (?,?,?,?,?,?)",
        (id1, fake_hash, "decision", 1700000001, "hash001", "sess0001"),
    )
    conn.execute(
        "INSERT INTO chunks (id, project, kind, ts, content_hash, session_id) VALUES (?,?,?,?,?,?)",
        (id2, fake_hash, "note", 1700000002, "hash002", "sess0002"),
    )
    conn.commit()
    conn.close()
    return fake_hash, pdir, [id1, id2]


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

def test_compose_unknown_project(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    brain_root.mkdir(parents=True, exist_ok=True)

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/api/compose?project=doesnotexist",
                json={"ids": ["abc"]},
            )

    resp = anyio.run(_run)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 2: empty ids list returns 400
# ---------------------------------------------------------------------------

def test_compose_empty_ids(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    fake_hash, pdir, ids = _make_project_with_chunks(brain_root)

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                f"/api/compose?project={fake_hash}",
                json={"ids": []},
            )

    resp = anyio.run(_run)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test 3: too-large ids list returns 400
# ---------------------------------------------------------------------------

def test_compose_too_many_ids(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    fake_hash, pdir, ids = _make_project_with_chunks(brain_root)

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                f"/api/compose?project={fake_hash}",
                json={"ids": [f"id{i:04d}" for i in range(51)]},
            )

    resp = anyio.run(_run)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test 4: valid compose with 2 ids returns drawer HTML with markdown
# ---------------------------------------------------------------------------

def test_compose_two_chunks(tmp_path: Path) -> None:
    brain_root = tmp_path / "brain"
    fake_hash, pdir, ids = _make_project_with_chunks(brain_root)

    id1, id2 = ids
    mock_texts = {
        id1: "This is the text of chunk one.",
        id2: "This is the text of chunk two.",
    }

    async def _run() -> httpx.Response:
        _patch_modules(brain_root)
        from ui.server import app
        import ui.routers.search as _search
        with patch.object(_search, "fetch_chunk_texts", return_value=mock_texts):
            with patch.object(_search, "open_lance", return_value=MagicMock()):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    return await client.post(
                        f"/api/compose?project={fake_hash}",
                        json={"ids": [id1, id2]},
                    )

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    html = resp.text

    # Should have compose drawer structure
    assert "compose-drawer" in html, "Response should contain compose drawer element"
    assert "compose-text" in html, "Response should contain compose-text textarea"

    # Should contain chunk count
    assert "2 chunk" in html, "Response should show chunk count"

    # Should contain chunk text in the textarea (markdown formatted)
    assert "chunk one" in html, "Markdown should contain text from chunk one"
    assert "chunk two" in html, "Markdown should contain text from chunk two"

    # Should contain copy and save buttons
    assert "copy" in html.lower(), "Compose drawer should have copy button"
    assert "save" in html.lower(), "Compose drawer should have save button"


# ---------------------------------------------------------------------------
# Test 5: _format_chunks reuse — verify markdown structure
# ---------------------------------------------------------------------------

def test_format_chunks_reuse() -> None:
    """Test that _format_chunks from inject_context is reused correctly."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "inject_context", _SCRIPTS / "inject_context.py"
    )
    ic = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ic)  # type: ignore[union-attr]

    chunks = [
        {"id": "aaa", "kind": "decision", "session_id": "sess0001", "text": "First chunk text."},
        {"id": "bbb", "kind": "note", "session_id": "sess0002", "text": "Second chunk text."},
    ]
    md = ic._format_chunks(chunks, header="Composed context (2 chunks)")

    assert "## Composed context (2 chunks)" in md
    assert "### [decision]" in md
    assert "### [note]" in md
    assert "First chunk text." in md
    assert "Second chunk text." in md
