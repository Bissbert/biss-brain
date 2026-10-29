#!/usr/bin/env python3
"""Migration: fold worktree-derived project buckets into their canonical parent.

DEPENDENCY: This script imports ``common.project_key``. The current master
implementation of ``project_key`` hashes the raw realpath of ``cwd``, which
means a worktree path like ``/repo/.claude/worktrees/agent-abc`` produces a
*different* hash than the canonical repo root ``/repo``.

Wave 1 Builder B2 will land a worktree-aware ``project_key`` in
``scripts/common.py`` that strips the ``.claude/worktrees/<name>`` suffix
before hashing. Until that change is merged, the companion test
``tests/test_migrate_fold_worktrees.py`` auto-skips itself (it detects
whether the new behaviour is in effect). Once B2 merges, the test will
auto-enable and this migration will function correctly end-to-end.

Usage::

    # Dry run (no writes — just print a JSON plan)
    python scripts/migrate_fold_worktrees.py

    # Apply the plan
    python scripts/migrate_fold_worktrees.py --apply

    # Custom brain root
    python scripts/migrate_fold_worktrees.py --apply --brain-root /tmp/test-brain

JSON report shape::

    {
        "folded":  [{"orphan_hash": "...", "canonical_hash": "...", "action": "moved"}],
        "merged":  [{"orphan_hash": "...", "canonical_hash": "...", "chunks_copied": N}],
        "skipped": [{"orphan_hash": "...", "reason": "..."}],
        "errors":  [{"orphan_hash": "...", "error": "..."}]
    }

Idempotency: the script can be run multiple times safely. On a second pass the
worktree dirs will no longer exist (already moved/deleted), so they will simply
not be found.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directly
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from common import BRAIN_ROOT, project_key  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_worktree_path(path: str) -> bool:
    """Return True if *path* contains a .claude/worktrees/ segment."""
    return ".claude/worktrees/" in path or "/.claude/worktrees/" in path


def _canonical_path_from_worktree(wt_path: str) -> str:
    """Strip the .claude/worktrees/<name> suffix to get the repo root.

    Example::
        /Users/fab/repos/foo/.claude/worktrees/agent-abc
        -> /Users/fab/repos/foo
    """
    p = Path(wt_path)
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "worktrees" and i >= 2 and parts[i - 1] == ".claude":
            # The repo root is everything before the ".claude" segment
            return str(Path(*parts[: i - 1]))
    # Fallback: try splitting on the literal string
    idx = wt_path.find("/.claude/worktrees/")
    if idx != -1:
        return wt_path[:idx]
    return wt_path


def _read_all_chunks(conn: sqlite3.Connection) -> list[dict]:
    """Return every row from the chunks table as a list of dicts."""
    cursor = conn.execute("SELECT * FROM chunks")
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Run the standard schema bootstrap so the target DB is ready."""
    # Import open_meta to reuse the full schema bootstrap, but we can't
    # call it with a path directly. We bootstrap manually for the minimal
    # columns we need to INSERT OR IGNORE.
    # All ALTER TABLE additions are idempotent — the target may already have
    # been opened by open_meta previously.
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
    }
    if "id" not in existing_cols:
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
                raw_ref TEXT
            )
        """)


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _merge_chunks(src_conn: sqlite3.Connection, dst_conn: sqlite3.Connection) -> int:
    """INSERT OR IGNORE all chunks from src into dst. Returns count inserted."""
    rows = _read_all_chunks(src_conn)
    if not rows:
        return 0
    inserted = 0
    for row in rows:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        values = [row[c] for c in cols]
        try:
            dst_conn.execute(
                f"INSERT OR IGNORE INTO chunks ({col_names}) VALUES ({placeholders})",
                values,
            )
            inserted += dst_conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            # Column may not exist in dst yet — open_meta handles additive
            # migrations, but in the migration context we do a best-effort
            # copy of columns that exist in dst.
            dst_cols = {
                r[1] for r in dst_conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            filtered_cols = [c for c in cols if c in dst_cols]
            placeholders2 = ", ".join("?" for _ in filtered_cols)
            col_names2 = ", ".join(filtered_cols)
            values2 = [row[c] for c in filtered_cols]
            dst_conn.execute(
                f"INSERT OR IGNORE INTO chunks ({col_names2}) VALUES ({placeholders2})",
                values2,
            )
            inserted += dst_conn.execute("SELECT changes()").fetchone()[0]
    dst_conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_plan(brain_root: Path) -> dict:
    """Walk brain_root and return a plan dict without writing anything."""
    plan: dict[str, list] = {"folded": [], "merged": [], "skipped": [], "errors": []}

    for entry in sorted(brain_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if len(name) != 12 or not all(c in "0123456789abcdef" for c in name):
            continue

        label_file = entry / "project_path.txt"
        if not label_file.exists():
            continue

        stored_path = label_file.read_text().strip()
        if not stored_path or not _is_worktree_path(stored_path):
            continue

        # This bucket was written from a worktree CWD
        canonical_path = _canonical_path_from_worktree(stored_path)
        try:
            canonical_hash = project_key(canonical_path)
        except Exception as exc:
            plan["errors"].append({"orphan_hash": name, "error": f"project_key failed: {exc}"})
            continue

        if canonical_hash == name:
            plan["skipped"].append({
                "orphan_hash": name,
                "reason": "project_key(canonical) == worktree hash (B2 not yet merged or paths identical)",
            })
            continue

        canonical_bucket = brain_root / canonical_hash

        if not canonical_bucket.exists():
            plan["folded"].append({
                "orphan_hash": name,
                "canonical_hash": canonical_hash,
                "canonical_path": canonical_path,
                "stored_path": stored_path,
                "action": "move",
            })
        else:
            # Count chunks to estimate merge work
            orphan_db = entry / "meta.sqlite"
            chunk_count = 0
            if orphan_db.exists():
                try:
                    c = sqlite3.connect(f"file:{orphan_db}?mode=ro", uri=True)
                    row = c.execute("SELECT COUNT(*) FROM chunks").fetchone()
                    c.close()
                    chunk_count = row[0] if row else 0
                except Exception:
                    pass
            plan["merged"].append({
                "orphan_hash": name,
                "canonical_hash": canonical_hash,
                "canonical_path": canonical_path,
                "stored_path": stored_path,
                "action": "merge",
                "estimated_chunks": chunk_count,
            })

    return plan


def apply_plan(brain_root: Path, plan: dict) -> dict:
    """Execute the plan in-place. Returns the report dict."""
    report: dict[str, list] = {
        "folded": [],
        "merged": [],
        "skipped": list(plan["skipped"]),
        "errors": list(plan["errors"]),
    }

    # ---- MOVE (orphan → canonical slot is empty) -------------------------
    for item in plan["folded"]:
        orphan_dir = brain_root / item["orphan_hash"]
        canonical_dir = brain_root / item["canonical_hash"]
        try:
            # Rewrite project_path.txt to the canonical path
            label_file = orphan_dir / "project_path.txt"
            label_file.write_text(item["canonical_path"])
            # Atomic rename
            orphan_dir.rename(canonical_dir)
            report["folded"].append({
                "orphan_hash": item["orphan_hash"],
                "canonical_hash": item["canonical_hash"],
                "action": "moved",
            })
        except Exception as exc:
            report["errors"].append({"orphan_hash": item["orphan_hash"], "error": str(exc)})

    # ---- MERGE (canonical bucket already exists) -------------------------
    for item in plan["merged"]:
        orphan_dir = brain_root / item["orphan_hash"]
        canonical_dir = brain_root / item["canonical_hash"]
        orphan_db = orphan_dir / "meta.sqlite"
        canonical_db = canonical_dir / "meta.sqlite"

        if not orphan_dir.exists():
            # Already gone (e.g. applied twice)
            report["skipped"].append({"orphan_hash": item["orphan_hash"], "reason": "already deleted"})
            continue

        try:
            # Bootstrap canonical schema if the DB doesn't exist yet
            dst_conn = _open_db(canonical_db)
            _ensure_schema(dst_conn)

            chunks_copied = 0
            if orphan_db.exists():
                src_conn = _open_db(orphan_db)
                chunks_copied = _merge_chunks(src_conn, dst_conn)
                src_conn.close()

            dst_conn.close()

            # Delete orphan lance dir (will be regenerated on next embed run)
            orphan_lance = orphan_dir / "lancedb"
            if orphan_lance.exists():
                shutil.rmtree(orphan_lance)

            # Delete the orphan directory
            shutil.rmtree(orphan_dir)

            report["merged"].append({
                "orphan_hash": item["orphan_hash"],
                "canonical_hash": item["canonical_hash"],
                "chunks_copied": chunks_copied,
            })
        except Exception as exc:
            report["errors"].append({"orphan_hash": item["orphan_hash"], "error": str(exc)})

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fold worktree-derived project buckets into their canonical parent bucket."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the moves/merges. Without this flag, prints a dry-run JSON plan.",
    )
    parser.add_argument(
        "--brain-root",
        default=None,
        metavar="PATH",
        help="Override the brain root directory (default: BRAIN_ROOT from common.py).",
    )
    args = parser.parse_args(argv)

    brain_root = Path(args.brain_root).expanduser() if args.brain_root else BRAIN_ROOT

    if not brain_root.exists():
        sys.exit(f"Brain root does not exist: {brain_root}")

    plan = build_plan(brain_root)

    if not args.apply:
        plan["_note"] = "Dry run — pass --apply to execute"
        print(json.dumps(plan, indent=2))
        return

    report = apply_plan(brain_root, plan)
    print(json.dumps(report, indent=2))

    if report["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
