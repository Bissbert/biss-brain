#!/usr/bin/env python3
"""Tests for scripts/migrate_recover_orphans.py.

Test matrix:
- ``test_recover_pinned_and_high_salience``: 1 pinned + 1 high-salience chunk
  recovered, 3 noise chunks dropped, orphan bucket deleted.
- ``test_high_value_kinds_always_recovered``: decision/lesson/plan chunks
  recovered regardless of salience score.
- ``test_target_project_required_with_apply``: --apply without --target-project
  exits non-zero.
- ``test_dry_run_no_writes``: plan printed, nothing written.
- ``test_idempotent``: second --apply pass is a no-op with no errors.
- ``test_empty_shell_deleted``: orphan dir with no meta.sqlite is just deleted.
- ``test_home_dir_path_detected_as_orphan``: project_path.txt == $HOME counts
  as orphan.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402  (BISS_BRAIN_ROOT set by conftest)

from migrate_recover_orphans import (  # noqa: E402
    _is_orphan,
    apply_plan,
    build_plan,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOME = str(Path.home())


def _make_brain(tmp_path: Path) -> Path:
    br = tmp_path / "brain"
    br.mkdir()
    return br


def _make_project_dir(brain_root: Path, hash_name: str) -> Path:
    d = brain_root / hash_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_db(
    db_path: Path,
    chunks: list[dict],
) -> None:
    """Create a meta.sqlite with the given chunk rows (minimal schema)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            agent_id TEXT DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'note',
            ts INTEGER NOT NULL DEFAULT 1700000000,
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
        conn.execute(
            "INSERT OR IGNORE INTO chunks "
            "(id, project, kind, ts, content_hash, pinned, salience) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row.get("id", uuid.uuid4().hex),
                row.get("project", "testproj"),
                row.get("kind", "note"),
                row.get("ts", 1700000000),
                row.get("content_hash", uuid.uuid4().hex),
                row.get("pinned", 0),
                row.get("salience", 0),
            ),
        )
    conn.commit()
    conn.close()


def _chunk_ids(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ids = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
    conn.close()
    return ids


def _count_chunks(db_path: Path) -> int:
    return len(_chunk_ids(db_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recover_pinned_and_high_salience(tmp_path: Path) -> None:
    """1 pinned + 1 salience=80 chunk recovered; 3 noise chunks dropped."""
    brain_root = _make_brain(tmp_path)

    orphan_hash = "aabbccddeeff"
    target_hash = "112233445566"

    orphan_dir = _make_project_dir(brain_root, orphan_hash)
    # No project_path.txt → orphan

    pinned_id = uuid.uuid4().hex
    high_sal_id = uuid.uuid4().hex
    noise_ids = [uuid.uuid4().hex for _ in range(3)]

    chunks = [
        {"id": pinned_id, "project": orphan_hash, "pinned": 1, "salience": 5, "kind": "note"},
        {"id": high_sal_id, "project": orphan_hash, "pinned": 0, "salience": 80, "kind": "note"},
    ] + [
        {"id": nid, "project": orphan_hash, "pinned": 0, "salience": 10, "kind": "note"}
        for nid in noise_ids
    ]
    _make_db(orphan_dir / "meta.sqlite", chunks)

    target_dir = _make_project_dir(brain_root, target_hash)
    (target_dir / "project_path.txt").write_text("/projects/myproject")

    plan = build_plan(brain_root, min_salience=50, target_hash=target_hash)
    orphan_entry = next((o for o in plan["orphans"] if o["hash"] == orphan_hash), None)
    assert orphan_entry is not None, f"Orphan {orphan_hash} not in plan"
    assert orphan_entry["recoverable"] == 2, (
        f"Expected 2 recoverable chunks, got {orphan_entry['recoverable']}"
    )
    assert orphan_entry["dropped"] == 3, (
        f"Expected 3 dropped chunks, got {orphan_entry['dropped']}"
    )

    report = apply_plan(brain_root, plan)

    assert not report["errors"], f"Unexpected errors: {report['errors']}"
    assert report["chunks_recovered"] == 2
    assert report["chunks_dropped"] == 3
    assert orphan_hash in report["buckets_deleted"]
    assert not orphan_dir.exists(), "Orphan dir must be deleted after apply"

    target_db = target_dir / "meta.sqlite"
    recovered = _chunk_ids(target_db)
    assert pinned_id in recovered, "Pinned chunk must be in target"
    assert high_sal_id in recovered, "High-salience chunk must be in target"
    for nid in noise_ids:
        assert nid not in recovered, f"Noise chunk {nid} must NOT be in target"


def test_high_value_kinds_always_recovered(tmp_path: Path) -> None:
    """decision / lesson / plan chunks are recovered even with salience < threshold."""
    brain_root = _make_brain(tmp_path)

    orphan_hash = "deadbeef0011"
    target_hash = "cafebabe2233"

    orphan_dir = _make_project_dir(brain_root, orphan_hash)

    decision_id = uuid.uuid4().hex
    lesson_id = uuid.uuid4().hex
    plan_id = uuid.uuid4().hex
    low_sal_note_id = uuid.uuid4().hex

    chunks = [
        {"id": decision_id, "project": orphan_hash, "kind": "decision", "salience": 5},
        {"id": lesson_id, "project": orphan_hash, "kind": "lesson", "salience": 0},
        {"id": plan_id, "project": orphan_hash, "kind": "plan", "salience": 1},
        {"id": low_sal_note_id, "project": orphan_hash, "kind": "note", "salience": 2},
    ]
    _make_db(orphan_dir / "meta.sqlite", chunks)

    target_dir = _make_project_dir(brain_root, target_hash)
    (target_dir / "project_path.txt").write_text("/projects/other")

    plan = build_plan(brain_root, min_salience=50, target_hash=target_hash)
    report = apply_plan(brain_root, plan)

    assert not report["errors"]
    target_db = target_dir / "meta.sqlite"
    recovered = _chunk_ids(target_db)

    assert decision_id in recovered, "decision chunk must be recovered"
    assert lesson_id in recovered, "lesson chunk must be recovered"
    assert plan_id in recovered, "plan chunk must be recovered"
    assert low_sal_note_id not in recovered, "low-salience note must be dropped"
    assert report["chunks_recovered"] == 3
    assert report["chunks_dropped"] == 1


def test_target_project_required_with_apply(tmp_path: Path) -> None:
    """--apply without --target-project should exit with non-zero status."""
    brain_root = _make_brain(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(["--apply", "--brain-root", str(brain_root)])

    assert exc_info.value.code != 0


def test_dry_run_no_writes(tmp_path: Path) -> None:
    """Without --apply, JSON plan is printed and nothing is written to disk."""
    brain_root = _make_brain(tmp_path)

    orphan_hash = "ff00ff00ff00"
    orphan_dir = _make_project_dir(brain_root, orphan_hash)
    chunk_id = uuid.uuid4().hex
    _make_db(
        orphan_dir / "meta.sqlite",
        [{"id": chunk_id, "project": orphan_hash, "pinned": 1, "salience": 90}],
    )

    target_hash = "ee11ee11ee11"
    target_dir = _make_project_dir(brain_root, target_hash)
    (target_dir / "project_path.txt").write_text("/projects/tgt")

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        main([
            "--brain-root", str(brain_root),
            "--target-project", target_hash,
        ])

    output = buf.getvalue()
    plan = json.loads(output)
    assert "_note" in plan
    assert "Dry run" in plan["_note"]

    # Orphan dir must still exist
    assert orphan_dir.exists(), "Orphan dir must be untouched during dry run"
    # Target DB must NOT have been created yet
    assert not (target_dir / "meta.sqlite").exists(), (
        "Target meta.sqlite must not be created during dry run"
    )


def test_idempotent(tmp_path: Path) -> None:
    """Running --apply twice produces no errors on the second pass."""
    brain_root = _make_brain(tmp_path)

    orphan_hash = "aabbccddeef0"
    target_hash = "001122334455"

    orphan_dir = _make_project_dir(brain_root, orphan_hash)
    _make_db(
        orphan_dir / "meta.sqlite",
        [
            {"id": uuid.uuid4().hex, "project": orphan_hash, "pinned": 1, "salience": 0},
        ],
    )

    target_dir = _make_project_dir(brain_root, target_hash)
    (target_dir / "project_path.txt").write_text("/projects/idem")

    # First apply
    plan1 = build_plan(brain_root, min_salience=50, target_hash=target_hash)
    report1 = apply_plan(brain_root, plan1)
    assert not report1["errors"], f"First apply errors: {report1['errors']}"
    assert not orphan_dir.exists()

    # Second apply — orphan gone, plan should find nothing
    plan2 = build_plan(brain_root, min_salience=50, target_hash=target_hash)
    report2 = apply_plan(brain_root, plan2)
    assert not report2["errors"], f"Second apply errors: {report2['errors']}"
    assert report2["chunks_recovered"] == 0
    assert report2["chunks_dropped"] == 0
    assert report2["buckets_deleted"] == []


def test_empty_shell_deleted(tmp_path: Path) -> None:
    """Orphan dir with no meta.sqlite (empty shell) is simply deleted."""
    brain_root = _make_brain(tmp_path)

    orphan_hash = "dead00001111"
    target_hash = "cafe00002222"

    orphan_dir = _make_project_dir(brain_root, orphan_hash)
    # No project_path.txt, no meta.sqlite — completely empty

    target_dir = _make_project_dir(brain_root, target_hash)
    (target_dir / "project_path.txt").write_text("/projects/shell")

    plan = build_plan(brain_root, min_salience=50, target_hash=target_hash)
    entry = next((o for o in plan["orphans"] if o["hash"] == orphan_hash), None)
    assert entry is not None
    assert entry["action"] == "delete_empty"

    report = apply_plan(brain_root, plan)
    assert not report["errors"]
    assert orphan_hash in report["buckets_deleted"]
    assert not orphan_dir.exists()


def test_home_dir_path_detected_as_orphan(tmp_path: Path) -> None:
    """project_path.txt containing $HOME is classified as an orphan."""
    brain_root = _make_brain(tmp_path)

    fake_hash = "abcdef012345"
    bucket_dir = _make_project_dir(brain_root, fake_hash)
    (bucket_dir / "project_path.txt").write_text(_HOME)

    assert _is_orphan(bucket_dir), (
        f"Bucket with project_path.txt == $HOME ({_HOME!r}) must be an orphan"
    )


def test_min_salience_cli_flag(tmp_path: Path) -> None:
    """--min-salience flag is passed through and affects recovery decisions."""
    brain_root = _make_brain(tmp_path)

    orphan_hash = "aabb00112233"
    target_hash = "ccdd44556677"

    orphan_dir = _make_project_dir(brain_root, orphan_hash)
    mid_sal_id = uuid.uuid4().hex
    low_sal_id = uuid.uuid4().hex
    _make_db(
        orphan_dir / "meta.sqlite",
        [
            {"id": mid_sal_id, "project": orphan_hash, "salience": 60, "kind": "note"},
            {"id": low_sal_id, "project": orphan_hash, "salience": 20, "kind": "note"},
        ],
    )

    target_dir = _make_project_dir(brain_root, target_hash)
    (target_dir / "project_path.txt").write_text("/projects/sal")

    # With high threshold (70), mid_sal_id is dropped too
    plan_high = build_plan(brain_root, min_salience=70, target_hash=target_hash)
    entry_high = next(o for o in plan_high["orphans"] if o["hash"] == orphan_hash)
    assert entry_high["recoverable"] == 0
    assert entry_high["dropped"] == 2

    # With low threshold (50), mid_sal_id is recovered
    plan_low = build_plan(brain_root, min_salience=50, target_hash=target_hash)
    entry_low = next(o for o in plan_low["orphans"] if o["hash"] == orphan_hash)
    assert entry_low["recoverable"] == 1
    assert entry_low["dropped"] == 1
