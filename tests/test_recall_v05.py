#!/usr/bin/env python3
"""Tests for B3 recall v0.5 features:
  - STDP co-retrieval pair canonicalisation
  - RIF decay on shared-neighbour siblings
  - Labile-window stamping
  - retrieval_count_local increment
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402  — ensure BRAIN_ROOT env is already set by conftest

from common import open_meta, project_key  # noqa: E402
from effects import (  # noqa: E402
    apply_rif as _apply_rif,
    bump_retrieval_count_local as _bump_retrieval_count_local,
    enqueue_co_retrievals as _enqueue_co_retrievals,
    stamp_labile as _stamp_labile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_defaults() -> dict:
    return {
        "stdp": {"enabled": True, "co_retrieval_min_count": 3, "edge_drain_window_days": 7},
        "rif": {"enabled": True, "shared_neighbour_threshold": 3, "decay_multiplier": 0.98},
        "reconsolidation": {"enabled": True, "labile_window_seconds": 300, "merge_cosine_threshold": 0.82},
        "forget": {"decay_floor": 0.05},
    }


def _make_meta(tmp_path: Path, proj: str) -> Any:
    """Open a meta.sqlite for an artificial project hash under BISS_BRAIN_ROOT."""
    brain_root = Path(os.environ["BISS_BRAIN_ROOT"])
    proj_dir = brain_root / proj
    proj_dir.mkdir(parents=True, exist_ok=True)
    # Monkey-patch project_dir to point here so open_meta works
    cwd = str(proj_dir)
    # We open directly via the path trick: override BISS_BRAIN_ROOT is already done
    # by conftest. Just open the meta under the test brain root.
    conn = open_meta(str(proj_dir))
    return conn, str(proj_dir)


def _insert_chunk(
    conn,
    proj: str,
    chunk_id: str | None = None,
    decay_score: float = 1.0,
    pinned: int = 0,
    retrieval_count: int = 0,
    retrieval_count_local: int = 0,
    labile_until: int = 0,
    kind: str = "stop",
) -> str:
    if chunk_id is None:
        chunk_id = uuid.uuid4().hex
    now = int(time.time())
    conn.execute(
        "INSERT OR REPLACE INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, "
        "superseded_by, pinned, retrieval_count, last_accessed, "
        "decay_score, retrieval_count_local, labile_until) "
        "VALUES (?, ?, '', '', ?, ?, ?, '', ?, ?, ?, ?, ?, ?)",
        (
            chunk_id, proj, kind, now,
            hashlib.sha256(chunk_id.encode()).hexdigest(),
            pinned, retrieval_count, now,
            decay_score, retrieval_count_local, labile_until,
        ),
    )
    conn.commit()
    return chunk_id


# ---------------------------------------------------------------------------
# STDP pair tests
# ---------------------------------------------------------------------------

def test_co_retrieval_pairs_canonicalised(tmp_path):
    """recall returns ids [c, a, b]; edge_bumps must have chunk_a < chunk_b for every pair."""
    proj = "testprojstdp1"
    conn, cwd = _make_meta(tmp_path, proj)

    # Use fixed IDs so we know the lex ordering
    id_a = "aaaa000000000000"
    id_b = "bbbb000000000000"
    id_c = "cccc000000000000"
    for cid in [id_a, id_b, id_c]:
        _insert_chunk(conn, proj, chunk_id=cid)

    retrieved = [id_c, id_a, id_b]
    _enqueue_co_retrievals(conn, proj, retrieved, session_id="sess-1")
    conn.commit()

    rows = conn.execute(
        "SELECT chunk_a, chunk_b FROM edge_bumps WHERE project = ?", (proj,)
    ).fetchall()
    conn.close()

    assert len(rows) == 3, f"Expected 3 pairs, got {len(rows)}"
    for a, b in rows:
        assert a < b, f"Pair not canonicalised: ({a}, {b})"

    pair_set = {(a, b) for a, b in rows}
    assert (id_a, id_b) in pair_set
    assert (id_a, id_c) in pair_set
    assert (id_b, id_c) in pair_set


def test_co_retrieval_does_not_self_pair(tmp_path):
    """A single retrieved id must produce no edge_bumps rows."""
    proj = "testprojstdp2"
    conn, cwd = _make_meta(tmp_path, proj)

    id_a = uuid.uuid4().hex
    _insert_chunk(conn, proj, chunk_id=id_a)

    _enqueue_co_retrievals(conn, proj, [id_a], session_id="sess-solo")
    conn.commit()

    rows = conn.execute(
        "SELECT COUNT(*) FROM edge_bumps WHERE project = ?", (proj,)
    ).fetchone()
    conn.close()

    assert rows[0] == 0, f"Expected 0 rows for single-id recall, got {rows[0]}"


def test_co_retrieval_empty_list(tmp_path):
    """An empty retrieved list must produce no edge_bumps rows and not raise."""
    proj = "testprojstdp3"
    conn, _ = _make_meta(tmp_path, proj)

    _enqueue_co_retrievals(conn, proj, [], session_id="sess-empty")
    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM edge_bumps WHERE project = ?", (proj,)
    ).fetchone()[0]
    conn.close()
    assert count == 0


# ---------------------------------------------------------------------------
# RIF decay tests
# ---------------------------------------------------------------------------

def _make_mock_tbl_with_neighbours(
    chunk_id: str,
    neighbour_ids: list[str],
    project: str,
) -> MagicMock:
    """Build a mock LanceDB table whose .search().where().limit().to_list() returns
    synthetic neighbour rows.

    The mock distinguishes two call patterns:
      - search called with the chunk_id's embedding → returns neighbour_ids rows
      - search called with a neighbour's embedding → returns rows including
        enough of neighbour_ids to hit the overlap threshold.
    """
    # Give each chunk a fake embedding (just a list for mock purposes)
    embeddings: dict[str, list[float]] = {
        chunk_id: [1.0] * 10,
    }
    for nid in neighbour_ids:
        embeddings[nid] = [0.5] * 10

    def _make_nbr_row(cid: str) -> dict:
        return {"id": cid, "vector": embeddings.get(cid, [0.0] * 10), "project": project}

    class _SearchChain:
        def __init__(self, vec: list[float]):
            self._vec = vec
            self._where_str = ""

        def distance_type(self, _metric: str) -> "_SearchChain":
            return self

        def where(self, sql: str) -> "_SearchChain":
            self._where_str = sql
            return self

        def limit(self, n: int) -> "_SearchChain":
            self._n = n
            return self

        def to_list(self) -> list[dict]:
            # When searching with the retrieved chunk's embedding, return neighbours
            if self._vec == embeddings.get(chunk_id):
                return [_make_nbr_row(nid) for nid in neighbour_ids[:self._n]]
            # When searching with a neighbour's embedding, return enough overlap
            # (simulate each neighbour sharing all neighbour_ids as its own neighbours)
            return [_make_nbr_row(nid) for nid in neighbour_ids[:self._n]]

    mock_tbl = MagicMock()
    mock_tbl.search.side_effect = lambda vec: _SearchChain(list(vec) if hasattr(vec, "__iter__") else vec)
    return mock_tbl


def test_rif_decreases_decay_for_shared_neighbours(tmp_path):
    """Chunk B that shares >= 3 of A's top-10 neighbours gets decay_score *= 0.98."""
    proj = "testprojrif1"
    conn, cwd = _make_meta(tmp_path, proj)

    id_a = "aaaa-rif-001"
    # Neighbours that B also has
    neighbour_ids = [f"nbr-{i:03d}" for i in range(4)]  # 4 shared neighbours >= threshold 3
    id_b = neighbour_ids[0]  # B is one of A's neighbours

    _insert_chunk(conn, proj, chunk_id=id_a, decay_score=1.0)
    for nid in neighbour_ids:
        _insert_chunk(conn, proj, chunk_id=nid, decay_score=1.0)

    cfg = _cfg_defaults()
    mock_tbl = _make_mock_tbl_with_neighbours(id_a, neighbour_ids, proj)

    _apply_rif(
        conn, mock_tbl, proj,
        retrieved_ids=[id_a],
        retrieved_embeddings={id_a: [1.0] * 10},
        cfg=cfg,
    )
    conn.commit()

    # B is a neighbour and should have been decayed
    row = conn.execute(
        "SELECT decay_score FROM chunks WHERE id = ?", (id_b,)
    ).fetchone()
    conn.close()
    assert row is not None, "chunk B not found"
    assert row[0] < 1.0, f"Expected B.decay_score < 1.0, got {row[0]}"
    assert abs(row[0] - 0.98) < 1e-6, f"Expected B.decay_score ≈ 0.98, got {row[0]}"


def test_rif_does_not_apply_cross_project(tmp_path):
    """RIF in project P1 must not modify decay_score of chunks in project P2."""
    proj1 = "testprojrif-p1"
    proj2 = "testprojrif-p2"
    conn1, _ = _make_meta(tmp_path, proj1)
    conn2, _ = _make_meta(tmp_path, proj2)

    id_a = "cross-proj-chunk-a"
    id_b_p2 = "cross-proj-chunk-b-in-p2"

    _insert_chunk(conn1, proj1, chunk_id=id_a, decay_score=1.0)
    _insert_chunk(conn2, proj2, chunk_id=id_b_p2, decay_score=1.0)

    cfg = _cfg_defaults()
    # The mock tbl only queries within proj1 (enforced by WHERE project = proj1 in _apply_rif)
    mock_tbl = _make_mock_tbl_with_neighbours(id_a, ["nbr-x1", "nbr-x2", "nbr-x3", "nbr-x4"], proj1)

    _apply_rif(
        conn1, mock_tbl, proj1,
        retrieved_ids=[id_a],
        retrieved_embeddings={id_a: [1.0] * 10},
        cfg=cfg,
    )
    conn1.commit()

    # B in P2 must be untouched (conn2 is separate; RIF only calls conn1)
    row = conn2.execute(
        "SELECT decay_score FROM chunks WHERE id = ?", (id_b_p2,)
    ).fetchone()
    conn1.close()
    conn2.close()
    assert row is not None, "chunk B_P2 not found"
    assert row[0] == 1.0, f"P2 chunk was modified: decay_score = {row[0]}"


def test_rif_skips_pinned(tmp_path):
    """Pinned chunks must not have their decay_score modified by RIF."""
    proj = "testprojrif-pin"
    conn, _ = _make_meta(tmp_path, proj)

    id_a = "pin-test-a"
    pinned_id = "pin-test-pinned"
    neighbour_ids = [pinned_id, "nbr-001", "nbr-002", "nbr-003"]

    _insert_chunk(conn, proj, chunk_id=id_a, decay_score=1.0, pinned=0)
    _insert_chunk(conn, proj, chunk_id=pinned_id, decay_score=0.9, pinned=1)
    for nid in neighbour_ids[1:]:
        _insert_chunk(conn, proj, chunk_id=nid, decay_score=1.0, pinned=0)

    cfg = _cfg_defaults()
    mock_tbl = _make_mock_tbl_with_neighbours(id_a, neighbour_ids, proj)

    _apply_rif(
        conn, mock_tbl, proj,
        retrieved_ids=[id_a],
        retrieved_embeddings={id_a: [1.0] * 10},
        cfg=cfg,
    )
    conn.commit()

    row = conn.execute(
        "SELECT decay_score FROM chunks WHERE id = ?", (pinned_id,)
    ).fetchone()
    conn.close()
    assert row is not None, "pinned chunk not found"
    assert row[0] == pytest.approx(0.9, abs=1e-6), (
        f"Pinned chunk decay_score should be unchanged (0.9), got {row[0]}"
    )


def test_rif_disabled_by_config(tmp_path):
    """When rif.enabled=False, no decay should be applied."""
    proj = "testprojrif-off"
    conn, _ = _make_meta(tmp_path, proj)

    id_a = "rif-off-a"
    nbr_id = "rif-off-nbr"
    _insert_chunk(conn, proj, chunk_id=id_a, decay_score=1.0)
    _insert_chunk(conn, proj, chunk_id=nbr_id, decay_score=1.0)

    cfg = _cfg_defaults()
    cfg["rif"]["enabled"] = False

    mock_tbl = _make_mock_tbl_with_neighbours(id_a, [nbr_id, "n2", "n3", "n4"], proj)
    _apply_rif(conn, mock_tbl, proj, [id_a], {id_a: [1.0] * 10}, cfg)
    conn.commit()

    row = conn.execute("SELECT decay_score FROM chunks WHERE id = ?", (nbr_id,)).fetchone()
    conn.close()
    assert row[0] == 1.0, f"Expected decay_score unchanged when RIF disabled, got {row[0]}"


# ---------------------------------------------------------------------------
# Labile-window stamping tests
# ---------------------------------------------------------------------------

def test_labile_until_set_on_recall(tmp_path):
    """_stamp_labile must set labile_until > now on every returned chunk."""
    proj = "testprojlabile1"
    conn, _ = _make_meta(tmp_path, proj)

    ids = [uuid.uuid4().hex for _ in range(3)]
    for cid in ids:
        _insert_chunk(conn, proj, chunk_id=cid, labile_until=0)

    cfg = _cfg_defaults()
    before = int(time.time())
    _stamp_labile(conn, proj, ids, cfg)
    conn.commit()

    for cid in ids:
        row = conn.execute(
            "SELECT labile_until FROM chunks WHERE id = ?", (cid,)
        ).fetchone()
        assert row is not None, f"chunk {cid} not found"
        assert row[0] > before, (
            f"labile_until should be in the future (> {before}), got {row[0]}"
        )
        assert row[0] >= before + 300, (
            f"labile_until should be >= now + 300s, got {row[0]}"
        )
    conn.close()


def test_labile_until_not_set_when_empty(tmp_path):
    """_stamp_labile with empty list must not raise and must not update any row."""
    proj = "testprojlabile2"
    conn, _ = _make_meta(tmp_path, proj)
    cid = _insert_chunk(conn, proj, labile_until=0)

    _stamp_labile(conn, proj, [], _cfg_defaults())
    conn.commit()

    row = conn.execute("SELECT labile_until FROM chunks WHERE id = ?", (cid,)).fetchone()
    conn.close()
    assert row[0] == 0, "labile_until should not change for empty id list"


# ---------------------------------------------------------------------------
# retrieval_count_local tests
# ---------------------------------------------------------------------------

def test_retrieval_count_local_increments(tmp_path):
    """_bump_retrieval_count_local must increment retrieval_count_local and NOT touch retrieval_count."""
    proj = "testprojrcl1"
    conn, _ = _make_meta(tmp_path, proj)

    cid = _insert_chunk(conn, proj, retrieval_count=7, retrieval_count_local=2)

    _bump_retrieval_count_local(conn, proj, [cid])
    conn.commit()

    row = conn.execute(
        "SELECT retrieval_count, retrieval_count_local FROM chunks WHERE id = ?", (cid,)
    ).fetchone()
    conn.close()

    retrieval_count, retrieval_count_local = row
    assert retrieval_count == 7, (
        f"Legacy retrieval_count must remain 7, got {retrieval_count}"
    )
    assert retrieval_count_local == 3, (
        f"retrieval_count_local should be 3 (was 2 + 1), got {retrieval_count_local}"
    )


def test_retrieval_count_local_only_local_project(tmp_path):
    """_bump_retrieval_count_local must not touch chunks in a different project."""
    proj1 = "testprojrcl-p1"
    proj2 = "testprojrcl-p2"
    conn1, _ = _make_meta(tmp_path, proj1)
    conn2, _ = _make_meta(tmp_path, proj2)

    id_p1 = _insert_chunk(conn1, proj1, retrieval_count_local=0)
    id_p2 = _insert_chunk(conn2, proj2, retrieval_count_local=0)

    # Bump only proj1 — conn1 only has proj1 rows
    _bump_retrieval_count_local(conn1, proj1, [id_p1, id_p2])
    conn1.commit()

    row_p2 = conn2.execute(
        "SELECT retrieval_count_local FROM chunks WHERE id = ?", (id_p2,)
    ).fetchone()
    conn1.close()
    conn2.close()

    assert row_p2[0] == 0, (
        f"P2 chunk retrieval_count_local should be unchanged, got {row_p2[0]}"
    )


# ---------------------------------------------------------------------------
# Integration: all v0.5 operations via mocked path
# ---------------------------------------------------------------------------

def test_all_v05_ops_on_single_retrieval(tmp_path):
    """Smoke-test that all four v0.5 operations run without error for 1 result."""
    proj = "testprojsmoke"
    conn, _ = _make_meta(tmp_path, proj)

    cid = _insert_chunk(conn, proj, decay_score=1.0, retrieval_count=0,
                        retrieval_count_local=0, labile_until=0)
    cfg = _cfg_defaults()

    mock_tbl = MagicMock()
    mock_tbl.search.return_value.where.return_value.limit.return_value.to_list.return_value = []

    before = int(time.time())
    _enqueue_co_retrievals(conn, proj, [cid], "sess-x")
    _apply_rif(conn, mock_tbl, proj, [cid], {cid: [1.0] * 10}, cfg)
    _stamp_labile(conn, proj, [cid], cfg)
    _bump_retrieval_count_local(conn, proj, [cid])
    conn.commit()

    row = conn.execute(
        "SELECT labile_until, retrieval_count_local FROM chunks WHERE id = ?", (cid,)
    ).fetchone()
    conn.close()

    assert row[0] > before, "labile_until not set"
    assert row[1] == 1, "retrieval_count_local not incremented"

    # No edge_bumps for single id
    # (already covered by test_co_retrieval_does_not_self_pair)
