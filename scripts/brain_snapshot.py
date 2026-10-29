#!/usr/bin/env python3
"""Read-only snapshot of a brain root, for evals and migrations dry-runs.

Usage:
  brain_snapshot.py DEST [--src ROOT] [--project HASH ...] [--path-prefix DIR ...]

Copies each selected project directory (meta.sqlite via the SQLite backup API,
so WAL contents are included consistently; lancedb/ and project_path.txt via
copytree) plus config.toml into DEST. Never writes to SRC.

DEST must not be inside SRC and SRC must not be inside DEST. Point
BISS_BRAIN_ROOT at DEST to run any brain script against the copy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_SRC = Path.home() / ".claude" / "brain"
_PROJECT_FILES = ("project_path.txt",)
_PROJECT_DIRS = ("lancedb",)


class SnapshotError(RuntimeError):
    pass


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def guard_paths(src: Path, dest: Path) -> None:
    """Refuse any src/dest pair where one contains the other."""
    if _is_within(dest, src):
        raise SnapshotError(f"refusing: destination {dest} is inside source {src}")
    if _is_within(src, dest):
        raise SnapshotError(f"refusing: source {src} is inside destination {dest}")


def _project_path(proj_dir: Path) -> str:
    try:
        return (proj_dir / "project_path.txt").read_text().strip()
    except OSError:
        return ""


def select_projects(
    src: Path, hashes: list[str] | None, prefixes: list[str] | None
) -> list[Path]:
    out = []
    for d in sorted(src.iterdir()):
        if not d.is_dir() or not ((d / "meta.sqlite").exists() or (d / "lancedb").exists()):
            continue
        if hashes and d.name not in hashes:
            continue
        if prefixes:
            p = _project_path(d)
            if not any(p == pre or p.startswith(pre.rstrip("/") + "/") for pre in prefixes):
                continue
        out.append(d)
    return out


def _backup_sqlite(src_db: Path, dest_db: Path) -> None:
    # mode=ro: never create or checkpoint the source
    s = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    try:
        d = sqlite3.connect(dest_db)
        try:
            s.backup(d)
        finally:
            d.close()
    finally:
        s.close()


def snapshot(
    dest: Path,
    src: Path = DEFAULT_SRC,
    hashes: list[str] | None = None,
    prefixes: list[str] | None = None,
) -> dict:
    src = src.expanduser()
    dest = dest.expanduser()
    if not src.is_dir():
        raise SnapshotError(f"source {src} does not exist")
    guard_paths(src, dest)
    if dest.exists() and any(dest.iterdir()):
        raise SnapshotError(f"destination {dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)

    projects = select_projects(src, hashes, prefixes)
    manifest = {"src": str(src), "created": int(time.time()), "projects": []}
    for d in projects:
        out = dest / d.name
        out.mkdir()
        if (d / "meta.sqlite").exists():
            _backup_sqlite(d / "meta.sqlite", out / "meta.sqlite")
        for name in _PROJECT_DIRS:
            if (d / name).is_dir():
                shutil.copytree(d / name, out / name, symlinks=True)
        for name in _PROJECT_FILES:
            if (d / name).exists():
                shutil.copy2(d / name, out / name)
        manifest["projects"].append({"hash": d.name, "path": _project_path(d)})
    if (src / "config.toml").exists():
        shutil.copy2(src / "config.toml", dest / "config.toml")
    (dest / "SNAPSHOT.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dest", type=Path)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--project", action="append", dest="hashes", help="project hash (repeatable)")
    p.add_argument("--path-prefix", action="append", dest="prefixes",
                   help="include projects whose project_path is under DIR (repeatable)")
    a = p.parse_args()
    try:
        m = snapshot(a.dest, a.src, a.hashes, a.prefixes)
    except SnapshotError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"snapshot: {len(m['projects'])} projects -> {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
