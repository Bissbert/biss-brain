#!/usr/bin/env python3
"""Migration: recover high-value chunks from no-path orphan buckets.

An "orphan bucket" is a project directory under brain_root where:
  - ``project_path.txt`` is missing, OR
  - ``project_path.txt`` contains only ``$HOME``, ``/``, or whitespace.

These buckets were typically created when biss-brain ran in an unidentified
working directory (e.g. cwd == $HOME). They accumulate ``note`` / ``decision``
chunks that are often low value, but occasionally contain pinned items or
high-salience insights worth preserving.

This script scans each orphan's ``meta.sqlite``, identifies *recoverable*
chunks (pinned=1, salience >= min_salience, or kind in decision/lesson/plan),
moves them into a target project bucket via ``INSERT OR IGNORE``, then deletes
the orphan directory.

Usage::

    # Dry run — no writes
    python scripts/migrate_recover_orphans.py --target-project <12-hex-hash>

    # Apply
    python scripts/migrate_recover_orphans.py --apply --target-project <12-hex-hash>

    # Recover into global project, raise salience threshold
    python scripts/migrate_recover_orphans.py --apply \\
        --target-project 5da0e6b1ad8a \\
        --min-salience 70

    # Custom brain root
    python scripts/migrate_recover_orphans.py --apply \\
        --target-project <hash> \\
        --brain-root /tmp/test-brain

JSON report shape::

    {
        "orphans_found": 5,
        "chunks_recovered": 3,
        "chunks_dropped": 12,
        "buckets_deleted": ["abc123...", "def456..."],
        "errors": []
    }

Idempotency: if an orphan bucket has already been deleted by a prior run, it
simply won't appear in the scan on subsequent runs.
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

from common import BRAIN_ROOT, GLOBAL_PROJECT_HASH  # noqa: E402

# Kinds that are always worth recovering regardless of salience
_HIGH_VALUE_KINDS = frozenset({"decision", "lesson", "plan"})

# Paths that indicate a "home-dir orphan" (no real project)
_HOME = str(Path.home())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_orphan(entry: Path) -> bool:
    """Return True if *entry* is a no-path orphan bucket."""
    label_file = entry / "project_path.txt"
    if not label_file.exists():
        return True
    stored = label_file.read_text().strip()
    if not stored:
        return True
    # Treat home dir, root, or single-slash variants as orphans
    if stored in (_HOME, "/", os.path.expanduser("~")):
        return True
    return False


def _open_db_ro(db_path: Path) -> sqlite3.Connection | None:
    """Open a SQLite DB in read-only mode. Returns None on failure."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except Exception:
        return None


def _open_db_rw(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _scan_orphan(
    conn: sqlite3.Connection, min_salience: int
) -> tuple[list[dict], int]:
    """Scan *conn* and return (recoverable_rows, drop_count).

    A row is recoverable if any of:
    - pinned = 1
    - salience >= min_salience (if the column exists)
    - kind IN ('decision', 'lesson', 'plan')
    """
    if not _table_exists(conn, "chunks"):
        return [], 0

    cols = _get_columns(conn, "chunks")
    has_salience = "salience" in cols
    has_pinned = "pinned" in cols

    cursor = conn.execute("SELECT * FROM chunks")
    col_names = [d[0] for d in cursor.description]
    all_rows = [dict(zip(col_names, row)) for row in cursor.fetchall()]

    recoverable: list[dict] = []
    drop_count = 0

    for row in all_rows:
        pinned = bool(row.get("pinned", 0)) if has_pinned else False
        salience = int(row.get("salience", 0)) if has_salience else 0
        kind = str(row.get("kind", ""))

        if pinned or salience >= min_salience or kind in _HIGH_VALUE_KINDS:
            recoverable.append(row)
        else:
            drop_count += 1

    return recoverable, drop_count


def _insert_rows(dst_conn: sqlite3.Connection, rows: list[dict]) -> int:
    """INSERT OR IGNORE rows into dst_conn's chunks table. Returns count inserted."""
    if not rows:
        return 0

    dst_cols = _get_columns(dst_conn, "chunks")
    inserted = 0

    for row in rows:
        # Only insert columns that exist in the destination schema
        filtered = {k: v for k, v in row.items() if k in dst_cols}
        if not filtered:
            continue
        cols_str = ", ".join(filtered.keys())
        placeholders = ", ".join("?" for _ in filtered)
        values = list(filtered.values())
        dst_conn.execute(
            f"INSERT OR IGNORE INTO chunks ({cols_str}) VALUES ({placeholders})",
            values,
        )
        inserted += dst_conn.execute("SELECT changes()").fetchone()[0]

    dst_conn.commit()
    return inserted


def _ensure_target_chunks_table(conn: sqlite3.Connection) -> None:
    """Bootstrap the chunks table in the target DB if it doesn't exist yet."""
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
            decay_score REAL NOT NULL DEFAULT 1.0,
            related_ids TEXT NOT NULL DEFAULT '',
            parent_id TEXT NOT NULL DEFAULT '',
            evidence_ids TEXT NOT NULL DEFAULT '',
            consolidation_round INTEGER NOT NULL DEFAULT 0,
            last_dream_ts INTEGER NOT NULL DEFAULT 0,
            salience INTEGER DEFAULT 0,
            labile_until INTEGER DEFAULT 0,
            reconsolidation_count INTEGER DEFAULT 0,
            co_retrieval_count INTEGER DEFAULT 0,
            cross_links TEXT DEFAULT '[]',
            cross_eligible INTEGER DEFAULT 0,
            retrieval_count_local INTEGER DEFAULT 0
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Scan / plan
# ---------------------------------------------------------------------------

def build_plan(brain_root: Path, min_salience: int, target_hash: str) -> dict:
    """Return a plan dict describing what would happen, without writing."""
    plan: dict = {
        "orphans": [],  # list of per-orphan detail dicts
        "target_hash": target_hash,
        "min_salience": min_salience,
    }

    for entry in sorted(brain_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if len(name) != 12 or not all(c in "0123456789abcdef" for c in name):
            continue
        if name == target_hash:
            continue  # never process the target itself

        if not _is_orphan(entry):
            continue

        db_path = entry / "meta.sqlite"
        if not db_path.exists():
            # Empty shell — will be deleted, nothing to recover
            plan["orphans"].append({
                "hash": name,
                "recoverable": 0,
                "dropped": 0,
                "action": "delete_empty",
                "error": None,
            })
            continue

        conn = _open_db_ro(db_path)
        if conn is None:
            plan["orphans"].append({
                "hash": name,
                "recoverable": 0,
                "dropped": 0,
                "action": "error",
                "error": "could not open meta.sqlite",
            })
            continue

        try:
            recoverable, drop_count = _scan_orphan(conn, min_salience)
        except Exception as exc:
            plan["orphans"].append({
                "hash": name,
                "recoverable": 0,
                "dropped": 0,
                "action": "error",
                "error": str(exc),
            })
            conn.close()
            continue
        finally:
            conn.close()

        plan["orphans"].append({
            "hash": name,
            "recoverable": len(recoverable),
            "dropped": drop_count,
            "action": "recover_and_delete",
            "error": None,
        })

    return plan


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_plan(brain_root: Path, plan: dict) -> dict:
    """Execute the recovery plan. Returns the JSON-ready report dict."""
    target_hash: str = plan["target_hash"]
    min_salience: int = plan["min_salience"]
    target_dir = brain_root / target_hash
    target_db = target_dir / "meta.sqlite"

    report: dict = {
        "orphans_found": len(plan["orphans"]),
        "chunks_recovered": 0,
        "chunks_dropped": 0,
        "buckets_deleted": [],
        "errors": [],
    }

    # Ensure target directory and schema exist
    target_dir.mkdir(parents=True, exist_ok=True)
    dst_conn = _open_db_rw(target_db)
    _ensure_target_chunks_table(dst_conn)

    for item in plan["orphans"]:
        name: str = item["hash"]
        orphan_dir = brain_root / name

        if item["action"] == "error":
            report["errors"].append({"hash": name, "error": item["error"]})
            continue

        if item["action"] == "delete_empty":
            try:
                if orphan_dir.exists():
                    shutil.rmtree(orphan_dir)
                    report["buckets_deleted"].append(name)
            except Exception as exc:
                report["errors"].append({"hash": name, "error": str(exc)})
            continue

        # action == "recover_and_delete"
        db_path = orphan_dir / "meta.sqlite"
        if not orphan_dir.exists():
            # Already processed in a prior run
            continue

        src_conn = _open_db_ro(db_path) if db_path.exists() else None
        try:
            if src_conn is not None:
                recoverable, drop_count = _scan_orphan(src_conn, min_salience)
                src_conn.close()
                src_conn = None

                inserted = _insert_rows(dst_conn, recoverable)
                report["chunks_recovered"] += inserted
                report["chunks_dropped"] += drop_count
            else:
                report["chunks_dropped"] += 0

            shutil.rmtree(orphan_dir)
            report["buckets_deleted"].append(name)

        except Exception as exc:
            if src_conn:
                src_conn.close()
            report["errors"].append({"hash": name, "error": str(exc)})

    dst_conn.close()
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover high-value chunks from no-path orphan buckets and delete the orphans."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the recovery and deletions. Without this flag, prints a dry-run JSON plan.",
    )
    parser.add_argument(
        "--target-project",
        default=None,
        metavar="HASH",
        help=(
            "12-hex hash of the destination project bucket for recovered chunks. "
            "Required when --apply is set."
        ),
    )
    parser.add_argument(
        "--min-salience",
        type=int,
        default=50,
        metavar="N",
        help="Minimum salience score to recover a chunk (default: 50).",
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

    if args.apply and not args.target_project:
        sys.exit(
            "Error: --target-project is required when --apply is set.\n"
            f"Hint: use --target-project {GLOBAL_PROJECT_HASH} to recover into the global bucket."
        )

    target_hash = args.target_project or GLOBAL_PROJECT_HASH

    plan = build_plan(brain_root, args.min_salience, target_hash)

    if not args.apply:
        plan["_note"] = "Dry run — pass --apply to execute"
        # Compute summary totals for convenience
        plan["_summary"] = {
            "orphans_found": len(plan["orphans"]),
            "chunks_to_recover": sum(o["recoverable"] for o in plan["orphans"]),
            "chunks_to_drop": sum(o["dropped"] for o in plan["orphans"]),
        }
        print(json.dumps(plan, indent=2))
        return

    report = apply_plan(brain_root, plan)
    print(json.dumps(report, indent=2))

    if report["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
