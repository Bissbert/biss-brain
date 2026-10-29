#!/usr/bin/env python3
"""Validation test for scripts/decay.py.

Fabricates chunks with old ts (60d ago) and retrieval_count=0; runs decay;
asserts decay_score < 0.4.

Also tests that pinned + recent + high retrieval produces decay_score > 1.0.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import project_key


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_sqlite_with_chunk(
    db_path: Path,
    chunk_id: str,
    text: str,
    ts: int,
    retrieval_count: int = 0,
    pinned: int = 0,
    kind: str = "stop",
    proj: str = "testproj",
) -> None:
    """Create a meta.sqlite with the chunks table and one row, directly via sqlite3."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            session_id TEXT,
            agent_id TEXT,
            kind TEXT NOT NULL,
            ts INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            last_accessed INTEGER,
            raw_ref TEXT,
            decay_score REAL NOT NULL DEFAULT 1.0
        )
    """)
    h = _content_hash(text)
    conn.execute(
        "INSERT OR REPLACE INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, "
        "superseded_by, last_accessed, pinned, retrieval_count, decay_score) "
        "VALUES (?, ?, '', '', ?, ?, ?, '', ?, ?, ?, 1.0)",
        (chunk_id, proj, kind, ts, h, ts, pinned, retrieval_count),
    )
    conn.commit()
    conn.close()


def test_old_unpinned_chunk_low_decay() -> None:
    """60-day-old chunk with no retrieval should have decay_score < 0.4."""
    # Import _compute_decay directly — no patching needed
    import decay
    importlib.reload(decay)

    old_ts = int(time.time()) - 60 * 86400  # 60 days ago
    score = decay._compute_decay(old_ts, 0, 0, 30.0)

    print(f"  decay_score for 60d-old chunk: {score:.4f}")
    assert score < 0.4, f"expected < 0.4, got {score:.4f}"
    print("  PASS: old unpinned chunk has low decay score")


def test_pinned_recent_high_retrieval_above_1() -> None:
    """Pinned + recent + high retrieval_count should produce decay_score > 1.0."""
    import decay
    importlib.reload(decay)

    recent_ts = int(time.time()) - 1 * 86400  # 1 day ago
    score = decay._compute_decay(recent_ts, retrieval_count=10, pinned=1, half_life=30.0)

    print(f"  decay_score for pinned+recent+high-retrieval: {score:.4f}")
    assert score > 1.0, f"expected > 1.0, got {score:.4f}"
    print("  PASS: pinned recent chunk has high decay score")


def test_decay_updates_sqlite() -> None:
    """run_decay() should write decay_score into SQLite for known chunks."""
    with tempfile.TemporaryDirectory() as fake_brain_str:
        fake_root = Path(fake_brain_str)

        # Create a fake project dir inside fake_root
        fake_proj = fake_root / "testproj123abc"
        fake_proj.mkdir()
        db_path = fake_proj / "meta.sqlite"

        chunk_id = uuid.uuid4().hex
        old_ts = int(time.time()) - 60 * 86400  # 60 days ago

        _make_sqlite_with_chunk(db_path, chunk_id, "test chunk for decay update", ts=old_ts)

        cfg = {"hygiene": {"decay_half_life_days": 30}}

        # Reload decay with patched BRAIN_ROOT
        import decay as _decay_mod
        importlib.reload(_decay_mod)

        with patch.object(_decay_mod, "BRAIN_ROOT", fake_root):
            count = _decay_mod.run_decay(cfg)

        assert count >= 1, f"expected at least 1 update, got {count}"

        # Read back the decay_score directly
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT decay_score FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        conn.close()

        assert row is not None, "chunk not found after decay run"
        score = row[0]
        print(f"  decay_score after run_decay: {score:.4f}")
        assert score < 0.4, f"expected < 0.4 for 60d-old chunk, got {score:.4f}"
        print("  PASS: run_decay updated decay_score in SQLite")


if __name__ == "__main__":
    print("test_decay.py")
    test_old_unpinned_chunk_low_decay()
    test_pinned_recent_high_retrieval_above_1()
    test_decay_updates_sqlite()
    print("\nAll decay tests passed.")
