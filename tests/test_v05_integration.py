#!/usr/bin/env python3
"""Cross-stream integration tests (F-006).

Three tests exercising end-to-end paths that cross builder boundaries:

  1. test_flow_capture_through_recall
     Flush a chunk via the capture path; run recall helpers; assert:
       - salience > 0
       - labile_until > now
       - retrieval_count_local == 1
       - edge_bumps row enqueued for any pair it was returned with

  2. test_flow_recall_to_reconsolidation
     Flush chunk A, recall it (stamping labile_until), flush a near-duplicate
     chunk A' within the labile window; assert A' merged into A (no new row,
     A.reconsolidation_count == 1).

  3. test_flow_global_admission
     With cross_project.enabled=True: create a chunk with kind='lesson',
     consolidation_round=2, pinned=1; run _stage_promote_to_global; assert
     cross_eligible == 1 and a row exists in _global meta.sqlite.
     With cross_project.enabled=False: second eligible chunk must NOT promote.

All tests use BISS_BRAIN_FAKE_EMBED=1 and stub distill to avoid network calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import lancedb
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Force fake embeddings before any import that might call _embed
os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"

from common import (  # noqa: E402
    DEFAULT_CONFIG,
    GLOBAL_PROJECT_HASH,
    _deep_merge,
    _fake_embed,
    content_hash,
    lancedb_dir,
    open_meta,
    project_key,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 4  # tiny dimension for test speed


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    cross_project_enabled: bool = False,
    min_retrievals: int = 5,
) -> dict:
    """Build a minimal config dict for integration tests."""
    return _deep_merge(DEFAULT_CONFIG, {
        "embed": {"backend": "ollama", "model": "nomic-embed-text", "dim": DIM,
                  "endpoint": "http://127.0.0.1:11434"},
        "distill": {
            "backend": "passthrough",
            "models": {"stop": "haiku", "merge": "haiku", "conflict_judge": "haiku"},
            "ollama": {"model": "qwen2.5-coder:14b-instruct"},
        },
        "reconsolidation": {
            "enabled": True,
            "labile_window_seconds": 300,
            "merge_cosine_threshold": 0.82,
            "promote_to_lesson_after": 5,
        },
        "stdp": {"enabled": True, "co_retrieval_min_count": 3, "edge_drain_window_days": 7},
        "rif": {"enabled": False},  # disabled to keep tests simple
        "retrieve": {"hybrid": False, "rerank": False, "session_start_top_k": 3},
        "cross_project": {
            "enabled": cross_project_enabled,
            "admission": {
                "min_consolidation_round": 1,
                "eligible_kinds": ["schema", "lesson", "hypothesis"],
                "frequency": {
                    "min_retrievals": min_retrievals,
                    "distinct_projects": 3,
                    "window_days": 30,
                },
            },
            "recall": {"cosine_gate": 0.85, "label_with_source_project": True},
            "backlinks": {"cosine_gate": 0.85},
        },
    })


def _fake_vec(text: str, dim: int = DIM) -> list[float]:
    """Deterministic unit-normalised fake embedding."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    v = [rng.random() for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        norm = 1.0
    return [x / norm for x in v]


def _near_vec(base: list[float], cosine: float = 0.92) -> list[float]:
    """Return a vector with approximately `cosine` similarity to base."""
    rng = random.Random(99)
    perp = [rng.gauss(0, 1) for _ in range(len(base))]
    dot = sum(p * b for p, b in zip(perp, base))
    perp = [p - dot * b for p, b in zip(perp, base)]
    pn_norm = math.sqrt(sum(x * x for x in perp))
    if pn_norm < 1e-9:
        pn_norm = 1.0
    pn = [x / pn_norm for x in perp]
    sin_t = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    blended = [cosine * b + sin_t * p for b, p in zip(base, pn)]
    bn = math.sqrt(sum(x * x for x in blended))
    if bn < 1e-9:
        bn = 1.0
    return [x / bn for x in blended]


def _make_lancedb_table(cwd: str, dim: int = DIM):
    """Open or create a LanceDB chunks table for a given cwd."""
    db = lancedb.connect(str(lancedb_dir(cwd)))
    if "chunks" in db.list_tables().tables:
        return db.open_table("chunks")
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
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])
    return db.create_table("chunks", schema=schema, mode="create")


def _insert_chunk(
    cwd: str,
    text: str,
    vec: list[float],
    kind: str = "stop",
    labile_until: int = 0,
    reconsolidation_count: int = 0,
    consolidation_round: int = 0,
    pinned: int = 0,
    retrieval_count_local: int = 0,
    salience: int = 0,
    dim: int = DIM,
) -> str:
    """Insert a chunk into both SQLite and LanceDB. Returns chunk_id."""
    proj = project_key(cwd)
    chunk_id = uuid.uuid4().hex
    ts = int(time.time())
    h = content_hash(text)

    meta = open_meta(cwd)
    meta.execute(
        "INSERT INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, "
        "superseded_by, last_accessed, labile_until, reconsolidation_count, "
        "consolidation_round, pinned, retrieval_count_local, salience) "
        "VALUES (?, ?, '', '', ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id, proj, kind, ts, h, ts,
            labile_until, reconsolidation_count,
            consolidation_round, pinned, retrieval_count_local, salience,
        ),
    )
    meta.commit()
    meta.close()

    tbl = _make_lancedb_table(cwd, dim)
    tbl.add([{
        "id": chunk_id,
        "project": proj,
        "session_id": "",
        "agent_id": "",
        "kind": kind,
        "ts": ts,
        "text": text,
        "content_hash": h,
        "superseded_by": "",
        "vector": [float(x) for x in vec],
    }])
    return chunk_id


# ---------------------------------------------------------------------------
# Test 1 — capture through recall
# ---------------------------------------------------------------------------

def test_flow_capture_through_recall(tmp_path):
    """Chunk flushed via capture path has correct v0.5 metadata after recall helpers run.

    Cross-stream path exercised: B1 Capture insert → B3 Recall post-retrieval ops.

    Assertions:
      - salience > 0 (set at insert time by compute_salience heuristic)
      - labile_until > now (set by _stamp_labile)
      - retrieval_count_local == 1 (incremented by _bump_retrieval_count_local)
      - edge_bumps row inserted when recalled together with a second chunk
    """
    cwd = str(tmp_path / "proj")
    Path(cwd).mkdir(parents=True)

    cfg = _make_cfg()
    proj = project_key(cwd)

    # Simulate B1 Capture: insert two chunks with non-zero salience (as flush would)
    text_a = "We use pytest for all unit and integration tests."
    text_b = "All API endpoints require JWT authentication tokens."
    vec_a = _fake_vec(text_a)
    vec_b = _fake_vec(text_b)

    chunk_a = _insert_chunk(cwd, text_a, vec_a, kind="stop", salience=6)
    chunk_b = _insert_chunk(cwd, text_b, vec_b, kind="stop", salience=4)

    # Verify salience was stored (B1 sets it at insert time)
    meta = open_meta(cwd)
    row_a = meta.execute("SELECT salience FROM chunks WHERE id = ?", (chunk_a,)).fetchone()
    assert row_a is not None and row_a[0] > 0, (
        f"salience must be > 0 after capture; got {row_a}"
    )

    # Simulate B3 Recall post-retrieval ops on these two chunks
    from effects import (
        bump_retrieval_count_local as _bump_retrieval_count_local,
        enqueue_co_retrievals as _enqueue_co_retrievals,
        stamp_labile as _stamp_labile,
    )

    now = int(time.time())
    retrieved = [chunk_a, chunk_b]

    _stamp_labile(meta, proj, retrieved, cfg)
    _bump_retrieval_count_local(meta, proj, retrieved)
    _enqueue_co_retrievals(meta, proj, retrieved, session_id="sess-it-01")
    meta.commit()

    # Assert labile_until > now
    row_a = meta.execute("SELECT labile_until FROM chunks WHERE id = ?", (chunk_a,)).fetchone()
    assert row_a is not None and row_a[0] > now, (
        f"labile_until must be > now after recall; got {row_a[0]}, now={now}"
    )

    # Assert retrieval_count_local == 1
    row_a = meta.execute("SELECT retrieval_count_local FROM chunks WHERE id = ?", (chunk_a,)).fetchone()
    assert row_a is not None and row_a[0] == 1, (
        f"retrieval_count_local must be 1 after one recall; got {row_a[0]}"
    )

    # Assert edge_bumps was enqueued for the (chunk_a, chunk_b) pair
    bumps = meta.execute(
        "SELECT COUNT(*) FROM edge_bumps WHERE project = ? "
        "AND ((chunk_a = ? AND chunk_b = ?) OR (chunk_a = ? AND chunk_b = ?))",
        (proj, chunk_a, chunk_b, chunk_b, chunk_a),
    ).fetchone()
    meta.close()
    assert bumps is not None and bumps[0] >= 1, (
        f"edge_bumps must have at least 1 row for the pair after co-recall; got {bumps[0]}"
    )


# ---------------------------------------------------------------------------
# Test 2 — recall to reconsolidation
# ---------------------------------------------------------------------------

def test_flow_recall_to_reconsolidation(tmp_path):
    """Recalled chunk becomes labile; near-duplicate triggers merge (reconsolidation_count +1).

    Cross-stream path exercised:
      B1 Capture insert A → B3 Recall stamps labile_until → B1 Capture insert A'
      → B4 Reconsolidation merges A' into A.

    Assertions:
      - After merge: no new row for A' in SQLite
      - A.reconsolidation_count == 1
    """
    cwd = str(tmp_path / "proj")
    Path(cwd).mkdir(parents=True)

    cfg = _make_cfg()
    proj = project_key(cwd)

    # Step 1: Insert chunk A (B1 Capture)
    text_a = "The authentication service validates JWT tokens on every request."
    vec_a = _fake_vec(text_a)
    chunk_a = _insert_chunk(cwd, text_a, vec_a, kind="stop")

    # Step 2: Simulate B3 Recall stamping labile_until on A
    meta = open_meta(cwd)
    from effects import stamp_labile as _stamp_labile
    _stamp_labile(meta, proj, [chunk_a], cfg)
    meta.commit()
    meta.close()

    # Verify labile_until is now set
    meta = open_meta(cwd)
    row = meta.execute("SELECT labile_until FROM chunks WHERE id = ?", (chunk_a,)).fetchone()
    meta.close()
    assert row is not None and row[0] > int(time.time()), (
        "labile_until must be in the future after recall"
    )

    # Step 3: Insert near-duplicate A' (B4 Reconsolidation path)
    # A' has cosine >= 0.82 to A → triggers merge
    text_a_prime = "The auth service checks JWT tokens for each incoming request."
    vec_a_prime = _near_vec(vec_a, cosine=0.92)  # well above 0.82 threshold

    # The merged text that distill would return
    merged_text = (
        "The authentication service validates JWT tokens on every incoming request."
    )
    merged_json = json.dumps({"merged_text": merged_text})

    tbl = _make_lancedb_table(cwd)
    meta_conn = open_meta(cwd)

    from flush import _try_reconsolidate

    # Stub ask() to return a deterministic merge result
    with patch("flush.ask", return_value=merged_json):
        # Stub _embed in flush to return a close vector (above drift threshold)
        with patch("flush._embed", return_value=vec_a):
            merged_id = _try_reconsolidate(
                cwd, tbl, meta_conn, text_a_prime, vec_a_prime, cfg
            )

    meta_conn.commit()
    meta_conn.close()

    # Assert: merge occurred — merged_id == chunk_a
    assert merged_id == chunk_a, (
        f"Expected A' to merge into A ({chunk_a[:8]}), got merged_id={merged_id}"
    )

    # Assert: no new row for A' (the caller would skip insert when merged_id is set)
    meta = open_meta(cwd)
    total_rows = meta.execute(
        "SELECT COUNT(*) FROM chunks WHERE project = ?", (proj,)
    ).fetchone()[0]
    assert total_rows == 1, (
        f"After merge, SQLite must have exactly 1 chunk row; got {total_rows}"
    )

    # Assert: A.reconsolidation_count == 1
    row = meta.execute(
        "SELECT reconsolidation_count FROM chunks WHERE id = ?", (chunk_a,)
    ).fetchone()
    meta.close()
    assert row is not None and row[0] == 1, (
        f"reconsolidation_count must be 1 after one merge; got {row}"
    )


# ---------------------------------------------------------------------------
# Test 3 — global admission
# ---------------------------------------------------------------------------

def test_flow_global_admission(tmp_path, monkeypatch):
    """Eligible lesson chunk is promoted to _global when cross_project.enabled=True.

    Cross-stream path exercised:
      B1 Capture (chunk with consolidation_round=2, pinned=1, kind='lesson')
      → B6 Global _stage_promote_to_global
      → _global meta.sqlite contains the promoted chunk, cross_eligible=1 in source.

    Second part: with cross_project.enabled=False a second eligible chunk must NOT promote.
    """
    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")

    # Re-import after monkeypatch to pick up updated BRAIN_ROOT
    import importlib
    import common as common_mod
    importlib.reload(common_mod)
    import dream as dream_mod
    importlib.reload(dream_mod)

    # Use the reload-resolved BRAIN_ROOT
    brain_root = common_mod.BRAIN_ROOT
    brain_root.mkdir(parents=True, exist_ok=True)

    # Create project directory
    proj_hash = "aabbccdd1122"
    pdir = brain_root / proj_hash
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "project_path.txt").write_text("/fake/project/alpha")

    # Build the full v0.5 CREATE TABLE DDL used for test project init
    _chunks_ddl = (
        "CREATE TABLE IF NOT EXISTS chunks ("
        "id TEXT PRIMARY KEY, project TEXT, session_id TEXT, agent_id TEXT, "
        "kind TEXT, ts INTEGER, content_hash TEXT, superseded_by TEXT DEFAULT '', "
        "pinned INTEGER DEFAULT 0, retrieval_count INTEGER DEFAULT 0, "
        "last_accessed INTEGER, consolidation_round INTEGER DEFAULT 0, "
        "last_dream_ts INTEGER DEFAULT 0, cross_eligible INTEGER DEFAULT 0, "
        "retrieval_count_local INTEGER DEFAULT 0, decay_score REAL DEFAULT 1.0, "
        "related_ids TEXT DEFAULT '', parent_id TEXT DEFAULT '', "
        "evidence_ids TEXT DEFAULT '', cross_links TEXT DEFAULT '[]', "
        "salience INTEGER DEFAULT 0, labile_until INTEGER DEFAULT 0, "
        "reconsolidation_count INTEGER DEFAULT 0, co_retrieval_count INTEGER DEFAULT 0)"
    )

    # Initialize project meta.sqlite with the full v0.5 schema (mirrors test_global.py)
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(_chunks_ddl)
    conn.commit()
    conn.close()

    # Pre-initialize the _global project directory so _ensure_global_project can
    # open it via _open_meta_direct (which requires the chunks table to exist).
    global_pdir = brain_root / "_global"
    global_pdir.mkdir(parents=True, exist_ok=True)
    (global_pdir / "project_path.txt").write_text("__global__")
    g_init = sqlite3.connect(str(global_pdir / "meta.sqlite"))
    g_init.execute(_chunks_ddl)
    g_init.commit()
    g_init.close()

    # Insert qualifying chunk: kind='lesson', consolidation_round=2, pinned=1
    chunk_id = uuid.uuid4().hex
    ts = int(time.time())
    h = content_hash("lesson content for global admission test")

    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(
        "INSERT OR REPLACE INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, "
        "superseded_by, pinned, retrieval_count_local, consolidation_round, "
        "cross_eligible, last_dream_ts) "
        "VALUES (?, ?, '', '', 'lesson', ?, ?, '', 1, 0, 2, 0, 0)",
        (chunk_id, proj_hash, ts, h),
    )
    conn.commit()
    conn.close()

    # Also insert in LanceDB so _two_phase_promote can copy the vector
    vec = _fake_vec("lesson content for global admission test")
    tbl = _make_lancedb_table(str(pdir), dim=DIM)
    tbl.add([{
        "id": chunk_id,
        "project": proj_hash,
        "session_id": "",
        "agent_id": "",
        "kind": "lesson",
        "ts": ts,
        "text": "lesson content for global admission test",
        "content_hash": h,
        "superseded_by": "",
        "vector": [float(x) for x in vec],
    }])

    # --- Part A: cross_project.enabled=True ---
    cfg_on = _make_cfg(cross_project_enabled=True, min_retrievals=5)
    # pinned=1 satisfies the frequency gate independently of min_retrievals

    promoted = dream_mod._stage_promote_to_global(pdir, cfg_on, dry_run=False)
    assert promoted == 1, f"Expected 1 chunk promoted to _global, got {promoted}"

    # Assert cross_eligible=1 in source SQLite
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    row = conn.execute(
        "SELECT cross_eligible FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    conn.close()
    assert row is not None and row[0] == 1, (
        f"cross_eligible must be 1 after global promotion; got {row}"
    )

    # Assert a row landed in the _global meta.sqlite
    global_pdir = brain_root / "_global"
    assert global_pdir.exists(), "_global directory must exist after promotion"
    global_db = global_pdir / "meta.sqlite"
    assert global_db.exists(), "_global/meta.sqlite must exist after promotion"

    g_conn = sqlite3.connect(str(global_db))
    global_row = g_conn.execute(
        "SELECT id FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    g_conn.close()
    assert global_row is not None, (
        f"chunk {chunk_id[:8]} must appear in _global meta.sqlite"
    )

    # --- Part B: cross_project.enabled=False → second chunk must NOT promote ---
    chunk_id2 = uuid.uuid4().hex
    h2 = content_hash("second lesson for disabled-path test")
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(
        "INSERT OR REPLACE INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, "
        "superseded_by, pinned, retrieval_count_local, consolidation_round, "
        "cross_eligible, last_dream_ts) "
        "VALUES (?, ?, '', '', 'lesson', ?, ?, '', 1, 0, 2, 0, 0)",
        (chunk_id2, proj_hash, ts, h2),
    )
    conn.commit()
    conn.close()

    vec2 = _fake_vec("second lesson for disabled-path test")
    tbl.add([{
        "id": chunk_id2,
        "project": proj_hash,
        "session_id": "",
        "agent_id": "",
        "kind": "lesson",
        "ts": ts,
        "text": "second lesson for disabled-path test",
        "content_hash": h2,
        "superseded_by": "",
        "vector": [float(x) for x in vec2],
    }])

    cfg_off = _make_cfg(cross_project_enabled=False)
    promoted_off = dream_mod._stage_promote_to_global(pdir, cfg_off, dry_run=False)
    assert promoted_off == 0, (
        f"With cross_project.enabled=False, no chunks should be promoted; got {promoted_off}"
    )

    # Verify chunk_id2 is NOT in _global
    g_conn = sqlite3.connect(str(global_db))
    global_row2 = g_conn.execute(
        "SELECT id FROM chunks WHERE id = ?", (chunk_id2,)
    ).fetchone()
    g_conn.close()
    assert global_row2 is None, (
        f"chunk {chunk_id2[:8]} must NOT appear in _global when cross_project disabled"
    )
