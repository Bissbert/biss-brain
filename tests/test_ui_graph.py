"""Tests for the Backlink Graph panel (V3).

Covers:
  test_graph_route_renders         -- HTML page returns 200, refs cytoscape.min.js
  test_graph_json_empty            -- empty project returns empty graph shape
  test_graph_edge_dedup            -- bidirectional related_ids produce one edge
  test_graph_filters_superseded    -- default omits superseded; include_superseded=1 includes them
  test_graph_min_degree_filter     -- star graph: min_degree=2 keeps only center
  test_graph_hub_detection         -- high-degree node gets is_hub=true
  test_graph_expand_neighborhood   -- BFS neighborhood expansion at depth 1 and 2
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


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def _make_project(brain_root: Path, cwd_name: str = "graphtest") -> tuple[str, Path]:
    fake_cwd = brain_root.parent / cwd_name
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
    chunk_id: str | None = None,
    kind: str = "stop",
    ts: int | None = None,
    pinned: int = 0,
    superseded_by: str = "",
    decay_score: float = 1.0,
    related_ids: str = "",
) -> str:
    cid = chunk_id or uuid.uuid4().hex
    conn.execute(
        "INSERT INTO chunks "
        "(id, project, kind, ts, content_hash, superseded_by, pinned, decay_score, related_ids) "
        "VALUES (?, 'test', ?, ?, '', ?, ?, ?, ?)",
        (cid, kind, ts or int(time.time()), superseded_by, pinned, decay_score, related_ids),
    )
    conn.commit()
    return cid


# ---------------------------------------------------------------------------
# Shared patch helper – called inside each async _run() function
# ---------------------------------------------------------------------------

def _patch_modules(brain_root: Path) -> None:
    """Remove cached modules and re-import with brain_root patched."""
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "common", "ui.data", "ui.server",
            "ui.routers.graph", "ui.routers.timeline",
            "ui.routers.search", "ui.routers.sessions",
            "ui.routers.scatter",
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
        return None  # no LanceDB needed for pure-SQL tests

    import ui.routers.graph as _gr
    _gr.validate_project = _validate
    _gr.open_meta_ro = _open_ro
    _gr.open_lance = _open_lance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_graph_route_renders(tmp_path: Path) -> None:
    """GET /graph?project=<h> returns 200 and contains cytoscape.min.js reference."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtest1")
    _init_db(pdir).close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/graph?project={ph}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
            assert "cytoscape.min.js" in resp.text, "Response must reference cytoscape.min.js"
            assert ph in resp.text, "Response must contain project hash"

    anyio.run(_run)


def test_graph_json_empty(tmp_path: Path) -> None:
    """Empty project returns node_count=0, edge_count=0."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtest2")
    _init_db(pdir).close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            data = resp.json()
            assert data["nodes"] == [], f"Expected empty nodes, got {data['nodes']}"
            assert data["edges"] == [], f"Expected empty edges, got {data['edges']}"
            assert data["summary"]["node_count"] == 0
            assert data["summary"]["edge_count"] == 0

    anyio.run(_run)


def test_graph_edge_dedup(tmp_path: Path) -> None:
    """Bidirectional related_ids (A->B and B->A) produce exactly one edge."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtest3")
    conn = _init_db(pdir)
    a_id = "aaaa" + uuid.uuid4().hex[:28]
    b_id = "bbbb" + uuid.uuid4().hex[:28]
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
            assert len(data["edges"]) == 1, (
                f"Expected exactly 1 edge (deduped), got {len(data['edges'])}: {data['edges']}"
            )
            assert len(data["nodes"]) == 2

    anyio.run(_run)


def test_graph_filters_superseded(tmp_path: Path) -> None:
    """Default omits superseded chunks; include_superseded=1 includes them."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtest4")
    conn = _init_db(pdir)
    a_id = "aaaa" + uuid.uuid4().hex[:28]
    b_id = "bbbb" + uuid.uuid4().hex[:28]
    c_id = "cccc" + uuid.uuid4().hex[:28]
    _insert_chunk(conn, chunk_id=a_id, related_ids=f"{b_id},{c_id}")
    _insert_chunk(conn, chunk_id=b_id, related_ids=f"{a_id},{c_id}")
    # c_id is superseded
    _insert_chunk(conn, chunk_id=c_id, related_ids=f"{a_id},{b_id}",
                  superseded_by=a_id)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Default: superseded excluded
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200
            data = resp.json()
            node_ids = {n["id"] for n in data["nodes"]}
            assert c_id not in node_ids, "Superseded node must not appear by default"
            assert a_id in node_ids
            assert b_id in node_ids

            # With include_superseded=1: all three appear
            resp2 = await client.get(f"/api/graph.json?project={ph}&include_superseded=1")
            assert resp2.status_code == 200
            data2 = resp2.json()
            node_ids2 = {n["id"] for n in data2["nodes"]}
            assert c_id in node_ids2, "Superseded node must appear with include_superseded=1"
            assert len(node_ids2) == 3

    anyio.run(_run)


def test_graph_min_degree_filter(tmp_path: Path) -> None:
    """Star graph: center+5 leaves. min_degree=2 keeps only the center."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtest5")
    conn = _init_db(pdir)
    center_id = "center" + uuid.uuid4().hex[:26]
    leaf_ids = ["leaf" + uuid.uuid4().hex[:28] for _ in range(5)]
    # Center points to all leaves
    _insert_chunk(conn, chunk_id=center_id, related_ids=",".join(leaf_ids))
    # Each leaf points back to center (bidirectional)
    for lid in leaf_ids:
        _insert_chunk(conn, chunk_id=lid, related_ids=center_id)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}&min_degree=2")
            assert resp.status_code == 200
            data = resp.json()
            node_ids = {n["id"] for n in data["nodes"]}
            assert center_id in node_ids, "Center node must survive min_degree=2"
            for lid in leaf_ids:
                assert lid not in node_ids, f"Leaf {lid} should be filtered out at min_degree=2"

    anyio.run(_run)


def test_graph_hub_detection(tmp_path: Path) -> None:
    """Node with high degree gets is_hub=true; low-degree nodes do not."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtest6")
    conn = _init_db(pdir)

    # Hub: connected to 6 others (degree=6)
    hub_id = "hub0" + uuid.uuid4().hex[:28]
    spokes = ["sp" + str(i) + uuid.uuid4().hex[:28] for i in range(6)]

    # 4 extra low-degree nodes connected only to each other in pairs
    extra_a = "exa" + uuid.uuid4().hex[:29]
    extra_b = "exb" + uuid.uuid4().hex[:29]
    extra_c = "exc" + uuid.uuid4().hex[:29]
    extra_d = "exd" + uuid.uuid4().hex[:29]

    _insert_chunk(conn, chunk_id=hub_id, related_ids=",".join(spokes))
    for sp in spokes:
        _insert_chunk(conn, chunk_id=sp, related_ids=hub_id)
    _insert_chunk(conn, chunk_id=extra_a, related_ids=extra_b)
    _insert_chunk(conn, chunk_id=extra_b, related_ids=extra_a)
    _insert_chunk(conn, chunk_id=extra_c, related_ids=extra_d)
    _insert_chunk(conn, chunk_id=extra_d, related_ids=extra_c)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/api/graph.json?project={ph}")
            assert resp.status_code == 200
            data = resp.json()
            nodes_by_id = {n["id"]: n for n in data["nodes"]}

            assert hub_id in nodes_by_id, "Hub node must appear in graph"
            assert nodes_by_id[hub_id]["is_hub"] is True, (
                f"Hub node must have is_hub=true, got {nodes_by_id[hub_id]}"
            )
            # Low-degree nodes should not be hubs
            for lid in [extra_a, extra_b, extra_c, extra_d]:
                if lid in nodes_by_id:
                    assert nodes_by_id[lid]["is_hub"] is False, (
                        f"Low-degree node {lid} must not be is_hub"
                    )

    anyio.run(_run)


def test_graph_expand_neighborhood(tmp_path: Path) -> None:
    """Chain A-B-C-D-E: expand from B depth=1 returns A,B,C; depth=2 returns A,B,C,D."""
    brain_root = tmp_path / "brain"
    ph, pdir = _make_project(brain_root, "gtest7")
    conn = _init_db(pdir)

    # Chain: A-B-C-D-E with bidirectional related_ids
    a_id = "chainA" + uuid.uuid4().hex[:26]
    b_id = "chainB" + uuid.uuid4().hex[:26]
    c_id = "chainC" + uuid.uuid4().hex[:26]
    d_id = "chainD" + uuid.uuid4().hex[:26]
    e_id = "chainE" + uuid.uuid4().hex[:26]

    _insert_chunk(conn, chunk_id=a_id, related_ids=b_id)
    _insert_chunk(conn, chunk_id=b_id, related_ids=f"{a_id},{c_id}")
    _insert_chunk(conn, chunk_id=c_id, related_ids=f"{b_id},{d_id}")
    _insert_chunk(conn, chunk_id=d_id, related_ids=f"{c_id},{e_id}")
    _insert_chunk(conn, chunk_id=e_id, related_ids=d_id)
    conn.close()

    async def _run() -> None:
        _patch_modules(brain_root)
        from ui.server import app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # depth=1: B plus immediate neighbors A, C
            resp1 = await client.post(
                f"/api/graph/expand?project={ph}&id={b_id}&depth=1"
            )
            assert resp1.status_code == 200, f"Expected 200, got {resp1.status_code}: {resp1.text}"
            d1 = resp1.json()
            ids1 = {n["id"] for n in d1["nodes"]}
            assert b_id in ids1, "B must be in depth=1 result"
            assert a_id in ids1, "A must be in depth=1 result"
            assert c_id in ids1, "C must be in depth=1 result"
            assert d_id not in ids1, f"D must NOT be in depth=1 result, got {ids1}"
            assert e_id not in ids1, f"E must NOT be in depth=1 result, got {ids1}"

            # depth=2: A, B, C plus next ring D
            resp2 = await client.post(
                f"/api/graph/expand?project={ph}&id={b_id}&depth=2"
            )
            assert resp2.status_code == 200, f"Expected 200, got {resp2.status_code}: {resp2.text}"
            d2 = resp2.json()
            ids2 = {n["id"] for n in d2["nodes"]}
            assert a_id in ids2, "A must be in depth=2 result"
            assert b_id in ids2, "B must be in depth=2 result"
            assert c_id in ids2, "C must be in depth=2 result"
            assert d_id in ids2, "D must be in depth=2 result"
            assert e_id not in ids2, f"E must NOT be in depth=2 result, got {ids2}"

    anyio.run(_run)
