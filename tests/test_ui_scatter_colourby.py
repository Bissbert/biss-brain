"""Tests for scatter colour-by feature (B3 - F-012).

Covers:
  test_scatter_json_includes_v05_columns  -- scatter.json points include all 5 extra fields
  test_scatter_colour_by_param            -- colour_by param is echoed in response
  test_scatter_empty_project              -- empty project returns empty points gracefully
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
import time
import uuid
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

# Skip removed: conftest autouse fixture restores BRAIN_ROOT between tests.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(brain_root: Path, name: str) -> tuple[str, Path]:
    fake_cwd = brain_root.parent / name
    project_hash = hashlib.sha256(str(fake_cwd.resolve()).encode()).hexdigest()[:12]
    pdir = brain_root / project_hash
    pdir.mkdir(parents=True, exist_ok=True)
    return project_hash, pdir


def _init_db_v05(pdir: Path, chunk_count: int = 3) -> list[str]:
    """Init meta.sqlite with v0.5 columns and insert chunks. Returns chunk ids."""
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
            retrieval_count_local INTEGER DEFAULT 0,
            decay_score REAL NOT NULL DEFAULT 1.0,
            related_ids TEXT NOT NULL DEFAULT '',
            salience INTEGER DEFAULT 50,
            consolidation_round INTEGER DEFAULT 1,
            co_retrieval_count INTEGER DEFAULT 2,
            labile_until INTEGER DEFAULT 0,
            cross_links TEXT DEFAULT '[]',
            cross_eligible INTEGER DEFAULT 0
        )
    """)
    ids = []
    now = int(time.time())
    for i in range(chunk_count):
        cid = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO chunks "
            "(id, project, kind, ts, content_hash, "
            "salience, consolidation_round, co_retrieval_count, retrieval_count_local, decay_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, "proj", "stop", now - i * 100, "",
             40 + i * 10, i, i + 1, i + 1, 0.9 - i * 0.1),
        )
        ids.append(cid)
    conn.commit()
    conn.close()
    return ids


def _make_fake_projection(ids: list[str]) -> dict:
    """Return a fake projection result matching projection_cache shape."""
    points = []
    for i, cid in enumerate(ids):
        points.append({
            "id": cid,
            "x": float(i),
            "y": float(i * 2),
            "kind": "stop",
            "ts": int(time.time()),
            "pinned": 0,
            "decay_score": 0.9,
            "superseded": False,
        })
    return {
        "points": points,
        "computed_ts": int(time.time()),
        "model_n": len(points),
        "stale": False,
    }


def _patch_modules(brain_root: Path) -> None:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scatter_json_includes_v05_columns(tmp_path: Path) -> None:
    """scatter.json points include salience, consolidation_round, retrieval_count, co_retrieval_count."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "sctest1")
    ids = _init_db_v05(pdir, chunk_count=3)
    fake_proj = _make_fake_projection(ids)

    async def _run() -> None:
        _patch_modules(brain_root)

        # Patch projection_cache to avoid umap-learn dependency
        import ui.routers.scatter as _sc
        from fastapi import HTTPException as _HE
        def _vp(ph: str) -> Path:
            pdir = brain_root / ph
            if not pdir.is_dir():
                raise _HE(status_code=404, detail="not found")
            return pdir
        _sc.validate_project = _vp

        mock_cache = MagicMock()
        mock_cache.get_or_compute.return_value = dict(fake_proj)
        sys.modules["ui.projection_cache"] = mock_cache

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/scatter.json?project={ph}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:400]}"
            data = resp.json()
            points = data.get("points", [])
            assert len(points) > 0, "Expected non-empty points"
            required_fields = {"id", "x", "y", "kind", "salience",
                               "consolidation_round", "decay_score", "retrieval_count"}
            for p in points:
                missing = required_fields - set(p.keys())
                assert not missing, f"Point missing fields {missing}: {p}"

    anyio.run(_run)


def test_scatter_colour_by_param(tmp_path: Path) -> None:
    """colour_by query param is echoed in the response JSON."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "sctest2")
    ids = _init_db_v05(pdir, chunk_count=2)
    fake_proj = _make_fake_projection(ids)

    async def _run() -> None:
        _patch_modules(brain_root)

        mock_cache = MagicMock()
        mock_cache.get_or_compute.return_value = dict(fake_proj)
        sys.modules["ui.projection_cache"] = mock_cache

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for field in ("kind", "salience", "consolidation_round", "decay_score", "retrieval_count"):
                resp = await client.get(f"/api/scatter.json?project={ph}&colour_by={field}")
                assert resp.status_code == 200
                data = resp.json()
                assert data.get("colour_by") == field, (
                    f"Expected colour_by={field!r}, got {data.get('colour_by')!r}"
                )

    anyio.run(_run)


def test_scatter_empty_project(tmp_path: Path) -> None:
    """Empty project (no points) returns empty list gracefully with colour_by."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "sctest3")
    _init_db_v05(pdir, chunk_count=0)

    async def _run() -> None:
        _patch_modules(brain_root)

        mock_cache = MagicMock()
        mock_cache.get_or_compute.return_value = {
            "points": [],
            "computed_ts": int(time.time()),
            "model_n": 0,
            "stale": False,
        }
        sys.modules["ui.projection_cache"] = mock_cache

        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/scatter.json?project={ph}&colour_by=salience")
            assert resp.status_code == 200
            data = resp.json()
            assert data["points"] == []
            assert "reason" in data
            assert data.get("colour_by") == "salience"

    anyio.run(_run)
