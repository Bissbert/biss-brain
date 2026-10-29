"""Tests for the _global cross-project pool display in index.html (B3 - F-013).

Covers:
  test_global_row_appears         -- _global project shown pinned at top
  test_global_empty_state_copy    -- empty _global shows the expected empty-state message
  test_global_with_chunks         -- non-zero _global shows chunk count
  test_regular_projects_unchanged -- non-global projects still appear in the table
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
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

GLOBAL_HASH = "5da0e6b1ad8a"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(brain_root: Path, name: str) -> tuple[str, Path]:
    fake_cwd = brain_root.parent / name
    project_hash = hashlib.sha256(str(fake_cwd.resolve()).encode()).hexdigest()[:12]
    pdir = brain_root / project_hash
    pdir.mkdir(parents=True, exist_ok=True)
    return project_hash, pdir


def _init_db(pdir: Path, chunk_count: int = 0) -> None:
    db_path = pdir / "meta.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'stop',
            ts INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT NOT NULL DEFAULT '',
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            decay_score REAL NOT NULL DEFAULT 1.0,
            related_ids TEXT NOT NULL DEFAULT ''
        )
    """)
    import time, uuid
    for _ in range(chunk_count):
        conn.execute(
            "INSERT INTO chunks (id, project, kind, ts, content_hash) VALUES (?,?,?,?,?)",
            (uuid.uuid4().hex, "global", "lesson", int(time.time()), "abc"),
        )
    conn.commit()
    conn.close()


def _make_global_project(brain_root: Path, chunk_count: int = 0) -> Path:
    pdir = brain_root / GLOBAL_HASH
    pdir.mkdir(parents=True, exist_ok=True)
    _init_db(pdir, chunk_count=chunk_count)
    # Write a project_path.txt so discover_projects() doesn't skip it
    # (it skips dirs with chunk_count==0 AND no path label)
    (pdir / "project_path.txt").write_text("__global__")
    return pdir


def _patch_modules(brain_root: Path) -> None:
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "common", "ui.data", "ui.server",
            "ui.routers.status", "ui.routers.search",
            "ui.routers.sessions", "ui.routers.timeline",
        ):
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

    import ui.data as _data2
    _data2.validate_project = _validate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_global_empty_state_copy(tmp_path: Path) -> None:
    """When _global has 0 chunks, the empty-state copy appears."""
    brain_root = tmp_path / "brain"
    _make_global_project(brain_root, chunk_count=0)
    # Also add a regular project so page renders non-empty
    ph, pdir = _make_project(brain_root, "regproject")
    pdir.joinpath("project_path.txt").write_text("/some/path")
    _init_db(pdir, chunk_count=2)

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:400]}"
            assert "cross-project lessons" in resp.text, (
                "Index page must show 'cross-project pool' label for _global"
            )
            assert "No cross-project lessons yet" in resp.text, (
                "Empty _global must show empty-state copy"
            )
            assert "cross_project.enabled" in resp.text, (
                "Empty-state copy must mention cross_project.enabled"
            )

    anyio.run(_run)


def test_global_with_chunks(tmp_path: Path) -> None:
    """When _global has chunks, the count is shown (not the empty-state)."""
    brain_root = tmp_path / "brain"
    _make_global_project(brain_root, chunk_count=4)

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "cross-project lessons" in resp.text
            # Should NOT show the empty-state copy
            assert "No cross-project lessons yet" not in resp.text, (
                "Non-empty _global must not show empty-state copy"
            )
            # Should show the chunk count (4 lessons)
            assert "4 cross-project lesson" in resp.text, (
                f"Expected '4 lesson' count in page; got excerpt: {resp.text[resp.text.find('global'):resp.text.find('global')+300]}"
            )

    anyio.run(_run)


def test_global_row_appears(tmp_path: Path) -> None:
    """_global project appears with teal cross-project pool styling."""
    brain_root = tmp_path / "brain"
    _make_global_project(brain_root, chunk_count=0)

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            # Global hash is referenced in the open link
            assert GLOBAL_HASH in resp.text, (
                f"Global hash {GLOBAL_HASH} must appear in index page"
            )
            # The global block has teal colouring in the inline style
            assert "var(--teal)" in resp.text, (
                "Global section must use teal styling"
            )

    anyio.run(_run)


def test_regular_projects_unchanged(tmp_path: Path) -> None:
    """Regular projects still appear in the table even when _global is present."""
    brain_root = tmp_path / "brain"
    _make_global_project(brain_root, chunk_count=1)
    ph, pdir = _make_project(brain_root, "myproject")
    pdir.joinpath("project_path.txt").write_text("/my/project")
    _init_db(pdir, chunk_count=5)

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            # Regular project hash must appear in the table
            assert ph in resp.text, (
                f"Regular project hash {ph} must appear in index page"
            )
            # Regular project path
            assert "/my/project" in resp.text

    anyio.run(_run)
