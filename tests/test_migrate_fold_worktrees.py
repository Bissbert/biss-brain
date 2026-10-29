#!/usr/bin/env python3
"""Tests for scripts/migrate_fold_worktrees.py.

DEPENDENCY NOTE
---------------
These tests auto-skip unless Wave 1 Builder B2's worktree-aware ``project_key``
is in effect. The skip guard works by checking whether::

    common.project_key("/foo/.claude/worktrees/agent-abc")
    != common.project_key("/foo")

On the current master the two calls produce the same hash (both resolve to
``/foo`` via ``os.path.realpath`` — the worktree dir may not exist on disk so
realpath returns the literal path, which differs). After B2 merges the new
``project_key`` will explicitly strip the ``/.claude/worktrees/<name>`` suffix
and return the repo-root hash for both inputs, making the two hashes *equal*,
which is precisely the condition we check for. The ``pytest.skip`` fires while
the hashes are **different** (B2 not yet merged); once they are **equal** the
tests run.

Test structure
--------------
- ``test_fold_move``: canonical bucket absent — orphan dir should be MOVED.
- ``test_fold_merge``: canonical bucket present — chunks should be MERGED,
  orphan dir deleted.
- ``test_dry_run_no_writes``: without --apply, nothing changes on disk.
- ``test_idempotent``: running --apply twice produces no errors on second pass.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402  (BISS_BRAIN_ROOT already set by conftest)

from common import project_key  # noqa: E402
from migrate_fold_worktrees import apply_plan, build_plan, main  # noqa: E402


# ---------------------------------------------------------------------------
# Skip guard: only run after B2 merges worktree-aware project_key
# ---------------------------------------------------------------------------

def _b2_merged() -> bool:
    """Return True if B2's worktree-aware project_key has landed.

    B2's `_normalize_worktree` reads the worktree's `.git` text file to
    discover the canonical repo, so it cannot be detected by hashing
    synthetic paths that do not exist on disk. Detect by importing the
    helper directly — its presence in common.py is the merge marker.
    """
    import common
    return hasattr(common, "_normalize_worktree")


_SKIP_REASON = (
    "requires B2 worktree-aware project_key — "
    "project_key('/foo/.claude/worktrees/X') must equal project_key('/foo')"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_db(db_path: Path, chunks: list[dict]) -> None:
    """Create a minimal meta.sqlite with the given chunk rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            agent_id TEXT DEFAULT '',
            kind TEXT NOT NULL,
            ts INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            last_accessed INTEGER DEFAULT 0,
            raw_ref TEXT DEFAULT '',
            decay_score REAL NOT NULL DEFAULT 1.0,
            salience INTEGER DEFAULT 0
        )
    """)
    for row in chunks:
        row_id = row.get("id", uuid.uuid4().hex)
        conn.execute(
            "INSERT OR IGNORE INTO chunks "
            "(id, project, kind, ts, content_hash, pinned, salience) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row_id,
                row.get("project", "proj"),
                row.get("kind", "note"),
                row.get("ts", 1700000000),
                row.get("content_hash", uuid.uuid4().hex),
                row.get("pinned", 0),
                row.get("salience", 10),
            ),
        )
    conn.commit()
    conn.close()


def _count_chunks(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return n


def _chunk_ids(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ids = {row[0] for row in conn.execute("SELECT id FROM chunks").fetchall()}
    conn.close()
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fold_move(tmp_path: Path) -> None:
    """Orphan with no canonical bucket → dir should be MOVED to canonical hash."""
    if not _b2_merged():
        pytest.skip(_SKIP_REASON)

    brain_root = tmp_path / "brain"
    brain_root.mkdir()

    # Build a fake repo root that B2's project_key would canonicalise
    fake_repo = "/projects/myrepo"
    fake_wt_path = f"{fake_repo}/.claude/worktrees/agent-deadbeef"

    # The worktree bucket hash: what the OLD project_key produced
    orphan_hash = project_key(fake_wt_path)  # after B2: same as canonical
    # The canonical hash: what project_key(fake_repo) produces
    canonical_hash = project_key(fake_repo)
    # After B2 these are equal — but for the MOVE test we need them different.
    # We simulate the pre-B2 state by manually using a different hash for orphan.
    # Use a deterministic fake orphan hash that won't collide.
    orphan_hash = "aabbccddeeff"  # synthetic pre-B2 hash

    orphan_dir = brain_root / orphan_hash
    orphan_dir.mkdir()
    (orphan_dir / "project_path.txt").write_text(fake_wt_path)
    chunk_id = uuid.uuid4().hex
    _make_db(orphan_dir / "meta.sqlite", [{"id": chunk_id, "project": orphan_hash}])

    # canonical bucket must NOT exist for the MOVE path
    canonical_dir = brain_root / canonical_hash
    assert not canonical_dir.exists()

    plan = build_plan(brain_root)
    assert any(p["orphan_hash"] == orphan_hash for p in plan["folded"]), (
        f"Expected orphan_hash {orphan_hash!r} in folded plan, got: {plan}"
    )

    report = apply_plan(brain_root, plan)
    assert not report["errors"], f"Unexpected errors: {report['errors']}"
    assert any(r["orphan_hash"] == orphan_hash for r in report["folded"])

    # Orphan dir gone, canonical dir appeared
    assert not orphan_dir.exists(), "Orphan dir should have been removed"
    assert canonical_dir.exists(), "Canonical dir should exist after move"

    # Chunks preserved
    assert _count_chunks(canonical_dir / "meta.sqlite") == 1
    assert chunk_id in _chunk_ids(canonical_dir / "meta.sqlite")

    # project_path.txt updated to canonical path
    label = (canonical_dir / "project_path.txt").read_text().strip()
    assert label == fake_repo, f"Expected canonical path {fake_repo!r}, got {label!r}"


def test_fold_merge(tmp_path: Path) -> None:
    """Orphan + existing canonical bucket → chunks merged, orphan deleted."""
    if not _b2_merged():
        pytest.skip(_SKIP_REASON)

    brain_root = tmp_path / "brain"
    brain_root.mkdir()

    fake_repo = "/projects/myrepo2"
    fake_wt_path = f"{fake_repo}/.claude/worktrees/agent-cafebabe"
    canonical_hash = project_key(fake_repo)
    orphan_hash = "112233445566"  # synthetic pre-B2 hash

    # Create orphan bucket with 2 chunks
    orphan_dir = brain_root / orphan_hash
    orphan_dir.mkdir()
    (orphan_dir / "project_path.txt").write_text(fake_wt_path)
    chunk_a = uuid.uuid4().hex
    chunk_b = uuid.uuid4().hex
    _make_db(
        orphan_dir / "meta.sqlite",
        [
            {"id": chunk_a, "project": orphan_hash},
            {"id": chunk_b, "project": orphan_hash},
        ],
    )

    # Create canonical bucket with 1 pre-existing chunk (different id)
    canonical_dir = brain_root / canonical_hash
    canonical_dir.mkdir()
    (canonical_dir / "project_path.txt").write_text(fake_repo)
    chunk_existing = uuid.uuid4().hex
    _make_db(
        canonical_dir / "meta.sqlite",
        [{"id": chunk_existing, "project": canonical_hash}],
    )

    plan = build_plan(brain_root)
    assert any(p["orphan_hash"] == orphan_hash for p in plan["merged"]), (
        f"Expected orphan_hash {orphan_hash!r} in merged plan, got: {plan}"
    )

    report = apply_plan(brain_root, plan)
    assert not report["errors"], f"Unexpected errors: {report['errors']}"
    assert any(r["orphan_hash"] == orphan_hash for r in report["merged"])

    # Orphan deleted
    assert not orphan_dir.exists(), "Orphan dir should have been deleted"

    # Canonical has all 3 chunks
    ids = _chunk_ids(canonical_dir / "meta.sqlite")
    assert chunk_existing in ids, "Pre-existing chunk should be preserved"
    assert chunk_a in ids, "Orphan chunk_a should have been merged"
    assert chunk_b in ids, "Orphan chunk_b should have been merged"
    assert len(ids) == 3


def test_dry_run_no_writes(tmp_path: Path) -> None:
    """Without --apply, no files are created, moved, or deleted."""
    if not _b2_merged():
        pytest.skip(_SKIP_REASON)

    brain_root = tmp_path / "brain"
    brain_root.mkdir()

    fake_repo = "/projects/dryrunrepo"
    fake_wt_path = f"{fake_repo}/.claude/worktrees/agent-dryrun"
    orphan_hash = "ffeeddccbbaa"

    orphan_dir = brain_root / orphan_hash
    orphan_dir.mkdir()
    (orphan_dir / "project_path.txt").write_text(fake_wt_path)
    _make_db(orphan_dir / "meta.sqlite", [{"id": uuid.uuid4().hex, "project": orphan_hash}])

    # Capture stdout from main() dry run
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--brain-root", str(brain_root)])

    output = buf.getvalue()
    plan = json.loads(output)
    assert "_note" in plan
    assert "Dry run" in plan["_note"]

    # Nothing on disk should have changed
    assert orphan_dir.exists(), "Orphan dir must still exist after dry run"
    canonical_hash = project_key(fake_repo)
    assert not (brain_root / canonical_hash).exists(), (
        "Canonical dir must NOT be created during dry run"
    )


def test_idempotent(tmp_path: Path) -> None:
    """Running --apply twice should not error on the second pass."""
    if not _b2_merged():
        pytest.skip(_SKIP_REASON)

    brain_root = tmp_path / "brain"
    brain_root.mkdir()

    fake_repo = "/projects/idempotentrepo"
    fake_wt_path = f"{fake_repo}/.claude/worktrees/agent-idem"
    orphan_hash = "aabbccdd1122"

    orphan_dir = brain_root / orphan_hash
    orphan_dir.mkdir()
    (orphan_dir / "project_path.txt").write_text(fake_wt_path)
    _make_db(orphan_dir / "meta.sqlite", [{"id": uuid.uuid4().hex, "project": orphan_hash}])

    # First apply
    plan1 = build_plan(brain_root)
    report1 = apply_plan(brain_root, plan1)
    assert not report1["errors"], f"First apply had errors: {report1['errors']}"

    # Second apply — orphan is gone, plan should find nothing to fold/merge
    plan2 = build_plan(brain_root)
    report2 = apply_plan(brain_root, plan2)
    assert not report2["errors"], f"Second apply had errors: {report2['errors']}"
    assert len(report2["folded"]) == 0
    assert len(report2["merged"]) == 0
