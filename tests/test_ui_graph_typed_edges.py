"""Tests for typed graph edges (B3 - F-009).

Covers:
  test_graph_edge_has_type_field      -- every edge in graph.json has a 'type' field
  test_graph_typed_related_ids        -- typed related_ids produce correct edge type
  test_graph_legacy_related_ids       -- legacy comma-separated ids default to co_occurs
  test_graph_cross_link_edges         -- cross_links column produces cross_link typed edges
  test_graph_reply_to_edge            -- reply_to typed dict produces reply_to edge
"""
from __future__ import annotations

import hashlib
import json
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

def _make_project(brain_root: Path, name: str) -> tuple[str, Path]:
    fake_cwd = brain_root.parent / name
    project_hash = hashlib.sha256(str(fake_cwd.resolve()).encode()).hexdigest()[:12]
    pdir = brain_root / project_hash
    pdir.mkdir(parents=True, exist_ok=True)
    return project_hash, pdir


def _init_db(pdir: Path) -> sqlite3.Connection:
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
            decay_score REAL NOT NULL DEFAULT 1.0,
            related_ids TEXT NOT NULL DEFAULT '',
            cross_links TEXT NOT NULL DEFAULT '[]'
        )
    """)
    conn.commit()
    return conn


def _insert_chunk(
    conn: sqlite3.Connection,
    chunk_id: str | None = None,
    kind: str = "stop",
    superseded_by: str = "",
    related_ids: str = "",
    cross_links: str = "[]",
) -> str:
    cid = chunk_id or uuid.uuid4().hex
    conn.execute(
        "INSERT INTO chunks (id, project, kind, ts, content_hash, superseded_by, related_ids, cross_links) "
        "VALUES (?, 'test', ?, ?, '', ?, ?, ?)",
        (cid, kind, int(time.time()), superseded_by, related_ids, cross_links),
    )
    conn.commit()
    return cid


def _patch_modules(brain_root: Path) -> None:
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "common", "ui.data", "ui.server",
            "ui.routers.graph", "ui.routers.status",
            "ui.routers.search", "ui.routers.sessions",
            "ui.routers.timeline", "ui.routers.scatter",
            "backlinks",
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

    def _open_ro(ph: str) -> sqlite3.Connection:
        sp = brain_root / ph / "meta.sqlite"
        if not sp.exists():
            raise HTTPException(status_code=404, detail="meta.sqlite not found")
        c = sqlite3.connect(f"file:{sp}?mode=ro", uri=True, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _open_lance(_ph: str):
        return None

    import ui.routers.graph as _gr
    _gr.validate_project = _validate
    _gr.open_meta_ro = _open_ro
    _gr.open_lance = _open_lance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_graph_edge_has_type_field(tmp_path: Path) -> None:
    """Every edge in graph.json must have a 'type' field."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtype1")
    conn = _init_db(pdir)
    a_id = uuid.uuid4().hex
    b_id = uuid.uuid4().hex
    # Legacy comma-separated related_ids
    _insert_chunk(conn, chunk_id=a_id, related_ids=b_id)
    _insert_chunk(conn, chunk_id=b_id, related_ids=a_id)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["edges"]) > 0, "Expected at least one edge"
            for edge in data["edges"]:
                assert "type" in edge, f"Edge missing 'type' field: {edge}"

    anyio.run(_run)


def test_graph_legacy_related_ids(tmp_path: Path) -> None:
    """Legacy comma-separated related_ids default to co_occurs edge type."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtype2")
    conn = _init_db(pdir)
    a_id = uuid.uuid4().hex
    b_id = uuid.uuid4().hex
    _insert_chunk(conn, chunk_id=a_id, related_ids=b_id)
    _insert_chunk(conn, chunk_id=b_id, related_ids=a_id)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200
            data = resp.json()
            edges = data["edges"]
            assert len(edges) == 1
            assert edges[0]["type"] == "co_occurs", (
                f"Legacy ids should produce co_occurs edge, got {edges[0]['type']!r}"
            )

    anyio.run(_run)


def test_graph_typed_related_ids(tmp_path: Path) -> None:
    """Typed JSON related_ids use the embedded type field."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtype3")
    conn = _init_db(pdir)
    a_id = uuid.uuid4().hex
    b_id = uuid.uuid4().hex
    # v0.5 typed format
    typed_a = json.dumps([{"id": b_id, "type": "co_occurs", "weight": 0.9, "ts": 0}])
    typed_b = json.dumps([{"id": a_id, "type": "co_occurs", "weight": 0.9, "ts": 0}])
    _insert_chunk(conn, chunk_id=a_id, related_ids=typed_a)
    _insert_chunk(conn, chunk_id=b_id, related_ids=typed_b)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200
            data = resp.json()
            edges = data["edges"]
            assert len(edges) == 1
            assert edges[0]["type"] == "co_occurs", (
                f"Typed co_occurs should produce co_occurs edge, got {edges[0]['type']!r}"
            )

    anyio.run(_run)


def test_graph_reply_to_edge(tmp_path: Path) -> None:
    """reply_to typed dict in related_ids produces reply_to edge type."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtype4")
    conn = _init_db(pdir)
    a_id = uuid.uuid4().hex
    b_id = uuid.uuid4().hex
    typed_a = json.dumps([{"id": b_id, "type": "reply_to", "weight": 1.0, "ts": 0}])
    _insert_chunk(conn, chunk_id=a_id, related_ids=typed_a)
    _insert_chunk(conn, chunk_id=b_id)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200
            data = resp.json()
            edges = data["edges"]
            assert len(edges) == 1, f"Expected 1 edge, got {len(edges)}"
            assert edges[0]["type"] == "reply_to", (
                f"Expected reply_to edge, got {edges[0]['type']!r}"
            )

    anyio.run(_run)


def test_graph_cross_link_edges(tmp_path: Path) -> None:
    """cross_links column produces edges with type='cross_link'."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtype5")
    conn = _init_db(pdir)
    a_id = uuid.uuid4().hex
    global_id = "global" + uuid.uuid4().hex[:26]
    cross = json.dumps([{"global_id": global_id, "weight": 0.92, "source_project": "abc123"}])
    _insert_chunk(conn, chunk_id=a_id, cross_links=cross)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200
            data = resp.json()
            cross_edges = [e for e in data["edges"] if e.get("type") == "cross_link"]
            assert len(cross_edges) >= 1, (
                f"Expected at least one cross_link edge, edges: {data['edges']}"
            )
            assert cross_edges[0]["type"] == "cross_link"

    anyio.run(_run)
