#!/usr/bin/env python3
"""Migration: fold every store whose path now keys to another project into that store.

`repo_root` gives a folder outside git the highest marked ancestor as its root
(common._marked_root), and a subfolder of a git repo the repo. A store written
before either rule (minecraft-modpack/pack/kubejs, a repo's app/fonts) keeps
its old hash: its memories are invisible from the project they belong to.
This script recomputes `project_key(project_path.txt)` for every store and,
where it differs from the directory name, moves the rows into the target:

  - LanceDB rows (text and vectors, no re-embedding) with `project` rewritten,
  - meta.sqlite `chunks` and `edge_bumps` with `project` rewritten,
  - queue/, pending/ and sessions/ files (names already present are kept),
  - open core proposals (core.sqlite `core_proposals.project`),
  - shared-pool copies (`_global` meta `source_project` / `source_projects`),
    so the dream reconcile does not retract them as orphans.

The folded store directory is then moved aside to BRAIN_ROOT.parent /
"brain-folded" / <stamp> / <hash>, never deleted. Stores without
project_path.txt, `_global` and anything not a 12-hex name are left alone.

Usage::

    # Dry run (default): print the plan as JSON, write nothing
    migrate_fold_stores.py

    # Apply: snapshot the affected stores and core.sqlite first, then fold
    migrate_fold_stores.py --apply [--backup DIR]

--apply refuses while the flush daemon runs (stop it, or wait for it to idle
out) and always writes the backup first; it never runs without one.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import common  # noqa: E402
from common import GLOBAL_PROJECT_HASH  # noqa: E402

_MOVED_DIRS = ("queue", "pending", "sessions")


def _is_store(name: str) -> bool:
    return len(name) == 12 and all(c in "0123456789abcdef" for c in name)


def _count(db: Path) -> int:
    if not db.exists():
        return 0
    try:
        c = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
        try:
            return c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            c.close()
    except sqlite3.Error:
        return 0


def build_plan(root: Path) -> dict:
    """Every fold this root needs, without writing anything."""
    plan: dict[str, list] = {"fold": [], "skipped": []}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not _is_store(d.name) or d.name == GLOBAL_PROJECT_HASH:
            continue
        label = d / "project_path.txt"
        if not label.is_file():
            plan["skipped"].append({"hash": d.name, "reason": "no project_path.txt"})
            continue
        path = label.read_text().strip()
        if not path or not Path(path).is_dir():
            plan["skipped"].append({"hash": d.name, "path": path, "reason": "path no longer exists"})
            continue
        target = common.project_key(path)
        if target == d.name:
            continue
        plan["fold"].append({
            "hash": d.name, "path": path,
            "target": target, "target_path": str(common.repo_root(path)),
            "target_exists": (root / target).is_dir(),
            "chunks": _count(d / "meta.sqlite"),
        })
    return plan


def flush_running(root: Path) -> bool:
    try:
        pid = int((root / "flush.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


# ── one fold ─────────────────────────────────────────────────────────────

def _fold_lance(src: Path, dst: Path, target: str) -> int:
    import lancedb  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.compute as pc  # noqa: PLC0415

    if not (src / "lancedb").is_dir():
        return 0
    sdb = lancedb.connect(str(src / "lancedb"))
    if "chunks" not in sdb.list_tables().tables:
        return 0
    rows = sdb.open_table("chunks").to_arrow()
    ddb = lancedb.connect(str(dst / "lancedb"))
    if "chunks" in ddb.list_tables().tables:
        tbl = ddb.open_table("chunks")
        have = set(tbl.to_arrow().column("id").to_pylist())
        if have:
            rows = rows.filter(pc.invert(pc.is_in(rows.column("id"), pa.array(sorted(have)))))
    else:
        tbl = ddb.create_table("chunks", schema=rows.schema, mode="create")
    if rows.num_rows:
        i = rows.schema.get_field_index("project")
        rows = rows.set_column(i, "project", pa.array([target] * rows.num_rows, pa.string()))
        tbl.add(rows)
    common.compact_store(tbl)
    return rows.num_rows


def _fold_meta(src: Path, dst: Path, target: str) -> int:
    if not (src / "meta.sqlite").exists():
        return 0
    s = sqlite3.connect(str(src / "meta.sqlite"), timeout=10)
    common.ensure_schema(s)  # an old store gets today's columns first
    s.close()
    d = sqlite3.connect(str(dst / "meta.sqlite"), timeout=10)
    try:
        d.execute("PRAGMA journal_mode=WAL")
        common.ensure_schema(d)
        d.execute("ATTACH DATABASE ? AS src", (str(src / "meta.sqlite"),))
        cols = [r[1] for r in d.execute("PRAGMA src.table_info(chunks)")]
        dcols = {r[1] for r in d.execute("PRAGMA main.table_info(chunks)")}
        cols = [c for c in cols if c in dcols]
        sel = ", ".join("?" if c == "project" else c for c in cols)
        before = d.total_changes
        d.execute(f"INSERT OR IGNORE INTO main.chunks ({', '.join(cols)}) SELECT {sel} FROM src.chunks",
                  (target,))
        n = d.total_changes - before
        d.execute("""INSERT INTO main.edge_bumps (project, chunk_a, chunk_b, bumped_at, session_id, drained)
                     SELECT ?, chunk_a, chunk_b, bumped_at, session_id, drained FROM src.edge_bumps""",
                  (target,))
        d.commit()
        d.execute("DETACH DATABASE src")
        return n
    finally:
        d.close()


def _move_files(src: Path, dst: Path) -> int:
    n = 0
    for name in _MOVED_DIRS:
        if not (src / name).is_dir():
            continue
        (dst / name).mkdir(exist_ok=True)
        for f in (src / name).iterdir():
            if f.is_file() and not (dst / name / f.name).exists():
                f.rename(dst / name / f.name)
                n += 1
    return n


def _rekey_proposals(root: Path, old: str, new: str) -> int:
    db = root / "core.sqlite"
    if not db.exists():
        return 0
    c = sqlite3.connect(str(db), timeout=10)
    try:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name = 'core_proposals'").fetchone():
            return 0
        n = c.execute("UPDATE core_proposals SET project = ? WHERE project = ?", (new, old)).rowcount
        c.commit()
        return n
    finally:
        c.close()


def _rekey_pool(root: Path, old: str, new: str) -> int:
    """Point pool copies that came from *old* at *new*; the rows keep their ids."""
    db = root / "_global" / "meta.sqlite"
    if not db.exists():
        return 0
    c = sqlite3.connect(str(db), timeout=10)
    try:
        cols = {r[1] for r in c.execute("PRAGMA table_info(chunks)")}
        if not {"source_project", "source_projects"} <= cols:
            return 0
        rows = c.execute("SELECT id, source_project, source_projects FROM chunks WHERE project = ? "
                         "AND (source_project = ? OR source_projects LIKE ?)",
                         (GLOBAL_PROJECT_HASH, old, f'%"{old}"%')).fetchall()
        for gid, sp, sps in rows:
            try:
                projs = json.loads(sps or "[]")
            except ValueError:
                projs = []
            projs = list(dict.fromkeys(new if h == old else h for h in projs))
            c.execute("UPDATE chunks SET source_project = ?, source_projects = ? WHERE id = ? AND project = ?",
                      (new if sp == old else sp, json.dumps(projs), gid, GLOBAL_PROJECT_HASH))
        c.commit()
        return len(rows)
    finally:
        c.close()


def fold_one(root: Path, item: dict, aside: Path) -> dict:
    src, dst = root / item["hash"], root / item["target"]
    dst.mkdir(exist_ok=True)
    if not (dst / "project_path.txt").exists():
        (dst / "project_path.txt").write_text(item["target_path"])
    out = {
        "hash": item["hash"], "target": item["target"],
        "rows": _fold_meta(src, dst, item["target"]),
        "vectors": _fold_lance(src, dst, item["target"]),
        "files": _move_files(src, dst),
        "proposals": _rekey_proposals(root, item["hash"], item["target"]),
        "pool": _rekey_pool(root, item["hash"], item["target"]),
    }
    aside.mkdir(parents=True, exist_ok=True)
    src.rename(aside / item["hash"])
    out["moved_to"] = str(aside / item["hash"])
    return out


def backup(root: Path, plan: dict, dest: Path) -> Path:
    """Snapshot every store the plan touches, plus core.sqlite and config."""
    import brain_snapshot  # noqa: PLC0415

    hashes = sorted({h for f in plan["fold"] for h in (f["hash"], f["target"], "_global")
                     if (root / h).is_dir()})
    brain_snapshot.snapshot(dest, src=root, hashes=hashes)
    return dest


def apply_plan(root: Path, plan: dict, backup_dir: Path) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup(root, plan, backup_dir)
    aside = root.parent / f"{root.name}-folded" / stamp
    report: dict[str, list] = {"folded": [], "errors": []}
    for item in plan["fold"]:
        try:
            report["folded"].append(fold_one(root, item, aside))
        except Exception as e:  # noqa: BLE001
            report["errors"].append({"hash": item["hash"], "error": str(e)})
    report["backup"] = str(backup_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--apply", action="store_true", help="fold (default: dry run)")
    p.add_argument("--brain-root", type=Path, default=None)
    p.add_argument("--backup", type=Path, default=None,
                   help="snapshot dir (default: <brain-root>-backups/fold-<stamp>)")
    a = p.parse_args(argv)
    root = (a.brain_root or common.BRAIN_ROOT).expanduser()
    if not root.is_dir():
        print(f"error: brain root {root} does not exist", file=sys.stderr)
        return 2
    plan = build_plan(root)
    if not a.apply:
        plan["_note"] = "dry run: pass --apply to fold (a backup is taken first)"
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not plan["fold"]:
        print(json.dumps({"folded": [], "note": "nothing to fold"}))
        return 0
    if flush_running(root):
        print("error: the flush daemon is running; stop it or wait for it to exit", file=sys.stderr)
        return 3
    dest = a.backup or root.parent / f"{root.name}-backups" / f"fold-{time.strftime('%Y%m%d-%H%M%S')}"
    report = apply_plan(root, plan, dest)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
