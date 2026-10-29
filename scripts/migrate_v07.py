#!/usr/bin/env python3
"""Bring an existing store to the v0.7 schema in place (the alternative to fresh_start.py).

  migrate_v07.py            dry run: what would change
  migrate_v07.py --apply    back up, then migrate

--apply refuses while any brain process runs, then writes a tar of the whole brain to <parent>/brain-backups/ and
stops if that fails. Then, per project: the additive SQLite columns and
tables (provenance, temporal validity, distill state; the same code every
process runs on open) and a complete FTS index. Finally store.json gets the
schema version, but no embed fingerprint: stored vectors cannot prove which
model made them, so a migrated store stays "unverified" in the doctor.

Nothing is deleted or rewritten; old 2,000-char passthrough chunks stay as
they are (see decision D8: the fresh start replaces the data migration).
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from fresh_start import flush_locked, running_writers  # noqa: E402


class Failed(RuntimeError):
    pass


def backup(root: Path, now: float | None = None) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    dest = root.parent / "brain-backups" / f"brain-{ts}.tar.gz"
    if dest.exists():
        raise Failed(f"{dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["tar", "-czf", str(dest), "-C", str(root.parent), root.name],
                       capture_output=True, text=True)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise Failed(f"backup failed: {r.stderr.strip() or 'empty archive'}")
    return dest


def _missing_columns(meta: Path) -> int:
    """Columns ensure_schema would add (reads a throwaway copy of the schema only)."""
    conn = sqlite3.connect(f"file:{meta}?mode=ro", uri=True, timeout=10)
    have = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    conn.close()
    ref = sqlite3.connect(":memory:")
    common.ensure_schema(ref)
    want = {r[1] for r in ref.execute("PRAGMA table_info(chunks)")}
    ref.close()
    return len(want - have)


def _fts_state(pdir: Path) -> str:
    import lancedb  # noqa: PLC0415

    ld = pdir / "lancedb"
    if not ld.is_dir():
        return "no vectors"
    db = lancedb.connect(str(ld))
    if "chunks" not in db.list_tables().tables:
        return "no vectors"
    n = common.fts_unindexed_rows(db.open_table("chunks"))
    return "no FTS index" if n is None else (f"{n} rows outside FTS" if n else "ok")


def plan(root: Path) -> list[dict]:
    out = []
    for meta in sorted(root.glob("*/meta.sqlite")):
        pdir = meta.parent
        out.append({"project": pdir.name, "columns": _missing_columns(meta), "fts": _fts_state(pdir)})
    return out


def migrate(root: Path, cfg: dict) -> int:
    n = 0
    for meta in sorted(root.glob("*/meta.sqlite")):
        conn = sqlite3.connect(str(meta), timeout=10)
        common.ensure_schema(conn)
        conn.close()
        ld = meta.parent / "lancedb"
        if ld.is_dir():
            import lancedb  # noqa: PLC0415

            db = lancedb.connect(str(ld))
            if "chunks" in db.list_tables().tables and not common.refresh_fts(db.open_table("chunks")):
                print(f"  ! {meta.parent.name}: FTS refresh failed (keyword search falls back to dense)")
        n += 1
    common.write_store_marker(cfg, root, schema_only=True)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    root = Path(common.BRAIN_ROOT)
    if not root.is_dir():
        print(f"{root} does not exist; nothing to migrate")
        return 1
    marker = common.read_store_marker(root) or {}
    print(f"brain: {root} (schema {marker.get('schema_version', 'pre-v0.7')} -> {common.SCHEMA_VERSION})")
    for p in plan(root):
        print(f"  {p['project']}: {p['columns']} columns to add, FTS {p['fts']}")
    if not a.apply:
        print("\ndry run: nothing changed. Run with --apply (a backup is written first).")
        return 0
    if flush_locked(root):
        print("refused: the flush daemon is running (flush.pid is locked); quit Claude Code sessions and retry",
              file=sys.stderr)
        return 1
    if writers := running_writers(root):
        print("refused: biss-brain processes are running, so the backup would not be consistent:\n  "
              + "\n  ".join(writers), file=sys.stderr)
        return 1
    try:
        dest = backup(root)
    except Failed as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(f"backup: {dest}")
    n = migrate(root, common.load_config())
    print(f"migrated {n} projects; store.json schema_version = {common.SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
