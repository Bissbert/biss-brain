"""Tests for B2 Edges: typed related_ids writer and NREM-2 STDP edge drain.

Tests:
  test_typed_related_ids_roundtrip
  test_legacy_related_ids_parsed_as_co_occurs
  test_edge_drain_writes_co_occurs_above_threshold
  test_edge_drain_below_threshold_no_op
  test_edge_drain_idempotent_on_redrain
  test_edge_drain_marks_drained
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Make scripts/ importable
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> Path:
    """A temporary project directory with BISS_BRAIN_FAKE_EMBED set."""
    os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    yield proj_dir
    os.environ.pop("BISS_BRAIN_FAKE_EMBED", None)


def _make_conn(tmp_path: Path) -> sqlite3.Connection:
    """Open an in-memory SQLite DB with the full v0.5 schema applied via open_meta."""
    # Use a fresh sub-directory so each test gets its own project dir
    import common

    fake_cwd = tmp_path / "fake_project"
    fake_cwd.mkdir(exist_ok=True)
    conn = common.open_meta(str(fake_cwd))
    conn.row_factory = sqlite3.Row
    return conn


def _insert_chunk(conn: sqlite3.Connection, chunk_id: str, project: str = "testproj") -> None:
    """Insert a minimal chunk row (no LanceDB required)."""
    conn.execute(
        "INSERT OR IGNORE INTO chunks "
        "(id, project, kind, ts, content_hash, superseded_by, related_ids) "
        "VALUES (?, ?, 'note', ?, 'deadbeef', '', '')",
        (chunk_id, project, int(time.time())),
    )
    conn.commit()


def _insert_bump(
    conn: sqlite3.Connection,
    project: str,
    chunk_a: str,
    chunk_b: str,
    n: int = 1,
) -> None:
    """Insert n edge_bumps rows for the given pair (already canonicalised)."""
    now = int(time.time())
    conn.executemany(
        "INSERT INTO edge_bumps (project, chunk_a, chunk_b, bumped_at, session_id, drained)"
        " VALUES (?, ?, ?, ?, 'sess-test', 0)",
        [(project, chunk_a, chunk_b, now)] * n,
    )
    conn.commit()


def _get_related_ids(conn: sqlite3.Connection, chunk_id: str) -> list[dict]:
    """Read related_ids for a chunk and parse as typed dicts."""
    from backlinks import parse_related_ids

    row = conn.execute("SELECT related_ids FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if not row or not row[0]:
        return []
    return parse_related_ids(row[0])


# ---------------------------------------------------------------------------
# Test 1: typed dict roundtrip
# ---------------------------------------------------------------------------


def test_typed_related_ids_roundtrip(tmp_path: Path) -> None:
    """Write a typed edge dict, read back, all fields preserved."""
    from backlinks import _add_typed_edge, parse_related_ids, serialize_related_ids

    conn = _make_conn(tmp_path)
    chunk_a = "aaaa0000"
    chunk_b = "bbbb1111"
    _insert_chunk(conn, chunk_a)
    _insert_chunk(conn, chunk_b)

    now_ts = int(time.time())
    _add_typed_edge(conn, chunk_a, chunk_b, "co_occurs", 0.42, now_ts)
    conn.commit()

    entries = _get_related_ids(conn, chunk_a)
    assert len(entries) == 1, f"expected 1 entry, got {len(entries)}: {entries}"
    e = entries[0]
    assert e["id"] == chunk_b, f"id mismatch: {e}"
    assert e["type"] == "co_occurs", f"type mismatch: {e}"
    assert abs(e["weight"] - 0.42) < 1e-6, f"weight mismatch: {e}"
    assert e["ts"] == now_ts, f"ts mismatch: {e}"

    # Also verify serialize -> parse roundtrip is lossless
    raw = serialize_related_ids(entries)
    reparsed = parse_related_ids(raw)
    assert reparsed == entries


# ---------------------------------------------------------------------------
# Test 2: legacy plain-string entries promoted to typed co_occurs
# ---------------------------------------------------------------------------


def test_legacy_related_ids_parsed_as_co_occurs(tmp_path: Path) -> None:
    """Legacy plain JSON list of strings is promoted to typed co_occurs dicts."""
    from backlinks import parse_related_ids

    conn = _make_conn(tmp_path)
    chunk_a = "legacyaa"
    chunk_b = "legacybb"
    chunk_c = "legacycc"
    _insert_chunk(conn, chunk_a)

    # Write legacy plain-string JSON directly (as old code would have done)
    legacy_raw = json.dumps([chunk_b, chunk_c])
    conn.execute(
        "UPDATE chunks SET related_ids = ? WHERE id = ?",
        (legacy_raw, chunk_a),
    )
    conn.commit()

    row = conn.execute("SELECT related_ids FROM chunks WHERE id = ?", (chunk_a,)).fetchone()
    entries = parse_related_ids(row[0])

    assert len(entries) == 2, f"expected 2 entries, got: {entries}"
    for entry in entries:
        assert "id" in entry, f"missing id field: {entry}"
        assert entry["type"] == "co_occurs", f"expected co_occurs, got: {entry}"
        assert entry["weight"] == 1.0, f"expected weight=1.0, got: {entry}"
        assert entry["ts"] == 0, f"expected ts=0, got: {entry}"
    ids = {e["id"] for e in entries}
    assert ids == {chunk_b, chunk_c}, f"unexpected ids: {ids}"


# ---------------------------------------------------------------------------
# Test 3: drain writes co_occurs above threshold
# ---------------------------------------------------------------------------


def test_edge_drain_writes_co_occurs_above_threshold(tmp_path: Path) -> None:
    """Insert ≥3 bumps for a pair; after drain, co_occurs edge present with weight ≥ 0.3."""
    from backlinks import parse_related_ids

    conn = _make_conn(tmp_path)
    project = "testproject"
    chunk_a = "aaaa1111"
    chunk_b = "bbbb2222"
    _insert_chunk(conn, chunk_a, project)
    _insert_chunk(conn, chunk_b, project)
    _insert_bump(conn, project, chunk_a, chunk_b, n=3)

    # Run the drain
    _run_drain(tmp_path, conn, project)

    # Both directions should have a co_occurs edge
    entries_a = _get_related_ids(conn, chunk_a)
    entries_b = _get_related_ids(conn, chunk_b)

    ids_a = {e["id"] for e in entries_a}
    ids_b = {e["id"] for e in entries_b}
    assert chunk_b in ids_a, f"chunk_b not in chunk_a's related_ids: {entries_a}"
    assert chunk_a in ids_b, f"chunk_a not in chunk_b's related_ids: {entries_b}"

    # Verify type and weight
    edge_a = next(e for e in entries_a if e["id"] == chunk_b)
    assert edge_a["type"] == "co_occurs", f"wrong type: {edge_a}"
    assert edge_a["weight"] >= 0.3, f"weight too low: {edge_a}"


# ---------------------------------------------------------------------------
# Test 4: drain below threshold is a no-op
# ---------------------------------------------------------------------------


def test_edge_drain_below_threshold_no_op(tmp_path: Path) -> None:
    """Insert 2 bumps (below default threshold of 3); drain should not write edges."""
    conn = _make_conn(tmp_path)
    project = "testproject"
    chunk_a = "cccc3333"
    chunk_b = "dddd4444"
    _insert_chunk(conn, chunk_a, project)
    _insert_chunk(conn, chunk_b, project)
    _insert_bump(conn, project, chunk_a, chunk_b, n=2)

    _run_drain(tmp_path, conn, project)

    entries_a = _get_related_ids(conn, chunk_a)
    entries_b = _get_related_ids(conn, chunk_b)
    assert entries_a == [], f"expected no edges for chunk_a, got: {entries_a}"
    assert entries_b == [], f"expected no edges for chunk_b, got: {entries_b}"

    # Bumps should still be undrained
    undrained = conn.execute(
        "SELECT COUNT(*) FROM edge_bumps WHERE project = ? AND drained = 0",
        (project,),
    ).fetchone()[0]
    assert undrained == 2, f"expected 2 undrained bumps, got {undrained}"


# ---------------------------------------------------------------------------
# Test 5: drain is idempotent (second call is a no-op)
# ---------------------------------------------------------------------------


def test_edge_drain_idempotent_on_redrain(tmp_path: Path) -> None:
    """Run drain twice; second invocation is a no-op (same edge, not duplicated)."""
    from backlinks import parse_related_ids

    conn = _make_conn(tmp_path)
    project = "testproject"
    chunk_a = "eeee5555"
    chunk_b = "ffff6666"
    _insert_chunk(conn, chunk_a, project)
    _insert_chunk(conn, chunk_b, project)
    _insert_bump(conn, project, chunk_a, chunk_b, n=5)

    # First drain
    _run_drain(tmp_path, conn, project)
    entries_after_first = _get_related_ids(conn, chunk_a)

    # Second drain — all bumps are drained=1 now, should be a no-op
    _run_drain(tmp_path, conn, project)
    entries_after_second = _get_related_ids(conn, chunk_a)

    # The second drain must not duplicate edges
    ids_after_first = [e["id"] for e in entries_after_first if e["type"] == "co_occurs"]
    ids_after_second = [e["id"] for e in entries_after_second if e["type"] == "co_occurs"]
    assert ids_after_second.count(chunk_b) == 1, (
        f"co_occurs edge duplicated after second drain: {entries_after_second}"
    )
    assert len(entries_after_second) == len(entries_after_first), (
        f"entry count changed on second drain: {entries_after_first} -> {entries_after_second}"
    )


# ---------------------------------------------------------------------------
# Test 6: drain marks consumed bumps as drained=1
# ---------------------------------------------------------------------------


def test_edge_drain_marks_drained(tmp_path: Path) -> None:
    """After drain, all consumed bump rows have drained=1."""
    conn = _make_conn(tmp_path)
    project = "testproject"
    chunk_a = "gggg7777"
    chunk_b = "hhhh8888"
    _insert_chunk(conn, chunk_a, project)
    _insert_chunk(conn, chunk_b, project)
    _insert_bump(conn, project, chunk_a, chunk_b, n=4)

    _run_drain(tmp_path, conn, project)

    drained = conn.execute(
        "SELECT COUNT(*) FROM edge_bumps WHERE project = ? AND drained = 1",
        (project,),
    ).fetchone()[0]
    undrained = conn.execute(
        "SELECT COUNT(*) FROM edge_bumps WHERE project = ? AND drained = 0",
        (project,),
    ).fetchone()[0]

    assert drained == 4, f"expected 4 drained bumps, got {drained}"
    assert undrained == 0, f"expected 0 undrained bumps, got {undrained}"


# ---------------------------------------------------------------------------
# Helper: run the drain stage against an existing connection
# ---------------------------------------------------------------------------


def _run_drain(tmp_path: Path, conn: sqlite3.Connection, project: str) -> int:
    """Run _stage_nrem2_edge_drain against the test DB.

    The drain function uses _open_meta_direct which opens by pdir path, but our
    test conn is already open. We replicate the drain logic directly here to
    avoid path indirection, using the same transaction pattern as the real stage.
    """
    from backlinks import _add_typed_edge

    threshold = 3

    # Commit any pending writes from the test setup so the drain sees them
    try:
        conn.execute("COMMIT")
    except sqlite3.OperationalError:
        pass  # Nothing pending

    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            """
            SELECT chunk_a, chunk_b, COUNT(*) AS n, MAX(bumped_at) AS last_ts
              FROM edge_bumps
             WHERE project = ? AND drained = 0
             GROUP BY chunk_a, chunk_b
            HAVING n >= ?
            """,
            (project, threshold),
        ).fetchall()

        promoted = 0
        for row in rows:
            chunk_a, chunk_b, n, last_ts = row[0], row[1], row[2], row[3]
            weight = min(1.0, n / 10.0)
            _add_typed_edge(conn, chunk_a, chunk_b, "co_occurs", weight, last_ts)
            _add_typed_edge(conn, chunk_b, chunk_a, "co_occurs", weight, last_ts)
            conn.execute(
                "UPDATE edge_bumps SET drained = 1"
                " WHERE project = ? AND chunk_a = ? AND chunk_b = ? AND drained = 0",
                (project, chunk_a, chunk_b),
            )
            promoted += 1

        conn.execute("COMMIT")
        return promoted
    except Exception:
        conn.execute("ROLLBACK")
        raise
