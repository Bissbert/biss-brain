"""Tests for the UMAP Scatter panel (V1).

Test cases:
  1. test_scatter_route_renders              GET /scatter?project=<hash> returns 200 HTML
  2. test_scatter_json_empty_project         empty project returns {"points": [], "reason": "..."}
  3. test_scatter_json_with_vectors          30 fake 768-dim vectors -> len(points)==30 with x,y,kind
  4. test_scatter_cache_hit                  second call within TTL returns same computed_ts
  5. test_scatter_cache_recompute_force      force=True returns newer (or equal) computed_ts with fresh model
  6. test_scatter_excludes_superseded_unless_pinned  pinned-superseded survive, plain superseded filtered
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is on sys.path so `ui` package beats scripts/ui.py
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)

import anyio
import httpx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

os.environ.setdefault("BISS_BRAIN_FAKE_EMBED", "1")


def _purge_modules(*names: str) -> None:
    """Remove named modules (and ui.*) from sys.modules to allow re-patching."""
    prefixes = set(names) | {"ui.", "common"}
    for key in list(sys.modules.keys()):
        if any(key == p or key.startswith(p) for p in prefixes):
            del sys.modules[key]


def _make_project(brain_root: Path, project_hash: str | None = None) -> tuple[str, Path]:
    """Create a minimal project dir with meta.sqlite.  Returns (hash, pdir)."""
    h = project_hash or ("a" * 12)
    pdir = brain_root / h
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
    return h, pdir


def _insert_sqlite_row(
    pdir: Path,
    chunk_id: str,
    kind: str = "note",
    superseded_by: str = "",
    pinned: int = 0,
    decay_score: float = 1.0,
    ts: int | None = None,
) -> None:
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(
        "INSERT OR REPLACE INTO chunks "
        "(id, project, kind, ts, content_hash, superseded_by, pinned, decay_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            pdir.name,
            kind,
            ts or int(time.time()),
            uuid.uuid4().hex,
            superseded_by,
            pinned,
            decay_score,
        ),
    )
    conn.commit()
    conn.close()


def _insert_lance_row(pdir: Path, chunk_id: str, vec: list[float]) -> None:
    """Insert a single vector row into the LanceDB chunks table."""
    import lancedb
    import pyarrow as pa

    lance_path = pdir / "lancedb"
    db = lancedb.connect(str(lance_path))

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("project", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("agent_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("ts", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("superseded_by", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
    ])

    tables = db.list_tables()
    table_names = tables.tables if hasattr(tables, "tables") else list(tables)
    if "chunks" in table_names:
        tbl = db.open_table("chunks")
    else:
        tbl = db.create_table("chunks", schema=schema, mode="create")

    tbl.add([{
        "id": chunk_id,
        "project": pdir.name,
        "session_id": "",
        "agent_id": "",
        "kind": "note",
        "ts": int(time.time()),
        "text": "test",
        "content_hash": uuid.uuid4().hex,
        "superseded_by": "",
        "vector": [float(v) for v in vec],
    }])


def _seed_project(
    pdir: Path,
    n: int = 30,
    rng_seed: int = 42,
    kinds: list[str] | None = None,
) -> list[str]:
    """Seed a project with n random 768-dim vectors in both stores.
    Returns the list of chunk ids.
    """
    rng = np.random.default_rng(rng_seed)
    if kinds is None:
        all_kinds = ["note", "stop", "subagent_stop", "pre_compact", "decision"]
        kinds = [all_kinds[i % len(all_kinds)] for i in range(n)]

    ids: list[str] = []
    for i in range(n):
        cid = uuid.uuid4().hex
        vec = rng.standard_normal(768).astype(np.float32)
        _insert_sqlite_row(pdir, cid, kind=kinds[i])
        _insert_lance_row(pdir, cid, vec.tolist())
        ids.append(cid)
    return ids


def _patch_brain_root(brain_root: Path) -> None:
    """Patch BRAIN_ROOT in already-loaded modules."""
    import common
    common.BRAIN_ROOT = brain_root
    try:
        import ui.data as _data
        _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]
        # Also clear the LanceDB connection cache so the new project is found
        _data._lance_cache.clear()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Test 1: scatter route renders HTML
# ---------------------------------------------------------------------------

def test_scatter_route_renders(tmp_path: Path) -> None:
    """GET /scatter?project=<hash> returns 200 HTML containing expected elements."""
    brain_root = tmp_path / "brain"
    fake_hash, _ = _make_project(brain_root)

    _purge_modules("common", "ui.data", "ui.server", "ui.routers.scatter")
    _patch_brain_root(brain_root)

    from ui.server import app  # noqa: PLC0415

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/scatter?project={fake_hash}")

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    body = resp.text
    assert "Embedding map" in body, "Page should contain 'Embedding map' heading"
    assert "scatter-canvas" in body, "Page should contain scatter-canvas div"
    assert "api/scatter.json" in body, "Page should reference /api/scatter.json"


# ---------------------------------------------------------------------------
# Test 2: empty project returns points=[] with reason
# ---------------------------------------------------------------------------

def test_scatter_json_empty_project(tmp_path: Path) -> None:
    """Empty project (no SQLite rows) returns {points: [], reason: '...'}."""
    brain_root = tmp_path / "brain"
    fake_hash, _ = _make_project(brain_root)

    _purge_modules("common", "ui.data", "ui.server", "ui.routers.scatter", "ui.projection_cache")
    _patch_brain_root(brain_root)

    from ui.server import app  # noqa: PLC0415

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/api/scatter.json?project={fake_hash}")

    resp = anyio.run(_run)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    assert data["points"] == [], f"Expected empty points, got {len(data.get('points', []))} points"
    assert "reason" in data, f"Expected 'reason' key, got: {list(data.keys())}"
    assert data["reason"], "reason should be a non-empty string"


# ---------------------------------------------------------------------------
# Test 3: 30 fake vectors produce 30 projected points
# ---------------------------------------------------------------------------

def test_scatter_json_with_vectors(tmp_path: Path) -> None:
    """Seed 30 fake vectors; assert points==30 and each has x, y, kind."""
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_project(brain_root)
    _seed_project(pdir, n=30)

    _purge_modules("common", "ui.data", "ui.server", "ui.routers.scatter", "ui.projection_cache")
    _patch_brain_root(brain_root)

    # Override projection cache dir to use tmp
    os.environ["_BISS_PROJ_DIR_OVERRIDE"] = str(tmp_path / "projections")

    from ui.projection_cache import get_or_compute, PROJECTIONS_DIR  # noqa: PLC0415
    import ui.projection_cache as _pc

    orig_dir = _pc.PROJECTIONS_DIR
    override_dir = tmp_path / "projections"
    override_dir.mkdir(parents=True, exist_ok=True)
    _pc.PROJECTIONS_DIR = override_dir

    try:
        result = get_or_compute(fake_hash)
    finally:
        _pc.PROJECTIONS_DIR = orig_dir
        os.environ.pop("_BISS_PROJ_DIR_OVERRIDE", None)

    points = result["points"]
    assert len(points) == 30, f"Expected 30 points, got {len(points)}"
    for p in points:
        assert "x" in p and "y" in p, f"Point missing x/y: {p}"
        assert "kind" in p, f"Point missing kind: {p}"
        assert isinstance(p["x"], float), f"x should be float: {p['x']}"
        assert isinstance(p["y"], float), f"y should be float: {p['y']}"


# ---------------------------------------------------------------------------
# Test 4: cache hit — second call returns same computed_ts
# ---------------------------------------------------------------------------

def test_scatter_cache_hit(tmp_path: Path) -> None:
    """Second call within TTL returns the same computed_ts (cache hit)."""
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_project(brain_root)
    _seed_project(pdir, n=30)

    _purge_modules("common", "ui.data", "ui.routers.scatter", "ui.projection_cache")
    _patch_brain_root(brain_root)

    import ui.projection_cache as _pc  # noqa: PLC0415

    orig_dir = _pc.PROJECTIONS_DIR
    override_dir = tmp_path / "projections"
    override_dir.mkdir(parents=True, exist_ok=True)
    _pc.PROJECTIONS_DIR = override_dir

    try:
        r1 = _pc.get_or_compute(fake_hash)
        r2 = _pc.get_or_compute(fake_hash)
    finally:
        _pc.PROJECTIONS_DIR = orig_dir

    assert r1["computed_ts"] == r2["computed_ts"], (
        f"Cache hit should return same computed_ts: {r1['computed_ts']} != {r2['computed_ts']}"
    )


# ---------------------------------------------------------------------------
# Test 5: force=True produces a fresh (or equal-ts) projection
# ---------------------------------------------------------------------------

def test_scatter_cache_recompute_force(tmp_path: Path) -> None:
    """force=True should produce a fresh projection (computed_ts >= first call)."""
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_project(brain_root)
    _seed_project(pdir, n=30)

    _purge_modules("common", "ui.data", "ui.routers.scatter", "ui.projection_cache")
    _patch_brain_root(brain_root)

    import ui.projection_cache as _pc  # noqa: PLC0415

    orig_dir = _pc.PROJECTIONS_DIR
    override_dir = tmp_path / "projections"
    override_dir.mkdir(parents=True, exist_ok=True)
    _pc.PROJECTIONS_DIR = override_dir

    try:
        r1 = _pc.get_or_compute(fake_hash, force=False)
        # Small sleep to ensure clock advances; on fast machines both calls
        # may land in the same second — we only assert >=, not >.
        time.sleep(0.05)
        r2 = _pc.get_or_compute(fake_hash, force=True)
    finally:
        _pc.PROJECTIONS_DIR = orig_dir

    assert r2["computed_ts"] >= r1["computed_ts"], (
        f"force=True should produce ts >= first call: {r2['computed_ts']} < {r1['computed_ts']}"
    )
    # Both should have valid points
    assert len(r2["points"]) == 30, f"Expected 30 points after force, got {len(r2['points'])}"


# ---------------------------------------------------------------------------
# Test 6: superseded chunks are excluded unless pinned
# ---------------------------------------------------------------------------

def test_scatter_excludes_superseded_unless_pinned(tmp_path: Path) -> None:
    """Plain superseded chunks are filtered; superseded+pinned chunks survive."""
    brain_root = tmp_path / "brain"
    fake_hash, pdir = _make_project(brain_root)

    rng = np.random.default_rng(99)

    # 20 normal chunks
    normal_ids: list[str] = []
    for _ in range(20):
        cid = uuid.uuid4().hex
        vec = rng.standard_normal(768).astype(np.float32)
        _insert_sqlite_row(pdir, cid, kind="note")
        _insert_lance_row(pdir, cid, vec.tolist())
        normal_ids.append(cid)

    # 5 superseded (plain) — should be excluded
    superseded_ids: list[str] = []
    for _ in range(5):
        cid = uuid.uuid4().hex
        vec = rng.standard_normal(768).astype(np.float32)
        _insert_sqlite_row(pdir, cid, kind="note", superseded_by="someotherid")
        _insert_lance_row(pdir, cid, vec.tolist())
        superseded_ids.append(cid)

    # 5 superseded + pinned — should be included
    pinned_superseded_ids: list[str] = []
    for _ in range(5):
        cid = uuid.uuid4().hex
        vec = rng.standard_normal(768).astype(np.float32)
        _insert_sqlite_row(pdir, cid, kind="note", superseded_by="someotherid", pinned=1)
        _insert_lance_row(pdir, cid, vec.tolist())
        pinned_superseded_ids.append(cid)

    _purge_modules("common", "ui.data", "ui.routers.scatter", "ui.projection_cache")
    _patch_brain_root(brain_root)

    import ui.projection_cache as _pc  # noqa: PLC0415

    orig_dir = _pc.PROJECTIONS_DIR
    override_dir = tmp_path / "projections"
    override_dir.mkdir(parents=True, exist_ok=True)
    _pc.PROJECTIONS_DIR = override_dir

    try:
        result = _pc.get_or_compute(fake_hash)
    finally:
        _pc.PROJECTIONS_DIR = orig_dir

    returned_ids = {p["id"] for p in result["points"]}

    # All normal chunks should be present
    for cid in normal_ids:
        assert cid in returned_ids, f"Normal chunk {cid} missing from projection"

    # Pinned+superseded should be present
    for cid in pinned_superseded_ids:
        assert cid in returned_ids, f"Pinned-superseded chunk {cid} missing from projection"

    # Plain superseded should NOT be present
    for cid in superseded_ids:
        assert cid not in returned_ids, f"Plain superseded chunk {cid} should be excluded"

    expected_count = len(normal_ids) + len(pinned_superseded_ids)
    assert len(result["points"]) == expected_count, (
        f"Expected {expected_count} points, got {len(result['points'])}"
    )
