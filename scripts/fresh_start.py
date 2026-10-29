#!/usr/bin/env python3
"""Start a new, empty store and archive the old one (v0.7 decision D8).

  fresh_start.py            dry run: what moves, what is carried over
  fresh_start.py --apply    do it

The old brain is renamed to <parent>/brain-archive/brain-<UTC ts>/ and never
deleted. The new brain gets the old config.toml, core memory (core/ with its
git history, and core.sqlite) and a store.json with the embed fingerprint of
the current config. Memories, queues, logs and caches stay in the archive.

Refuses when the flush daemon holds its lock, when dream, dmn, the UI or an
MCP server is running, when the archive target exists, or when the archive
would be on another filesystem (a rename must not turn into a copy).
"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

CARRY = ("config.toml", "core")
# processes that write the store; matched against full command lines
WRITERS = ("scripts/flush.py", "scripts/dream.py", "scripts/dmn.py", "mcp/server.py", "ui.server:app",
           "scripts/launch_ui.py")


class Refused(RuntimeError):
    pass


DEFAULT_ROOT = Path.home() / ".claude" / "brain"


def running_writers(root: Path | None = None) -> list[str]:
    """`pid command` lines of biss-brain processes that may write the store (read-only probe).

    A command line does not show which brain root a process uses, so for a
    root other than ~/.claude/brain (tests, scratch copies) nothing is probed;
    the flush lock is still checked.
    """
    if root is not None and Path(root).resolve() != DEFAULT_ROOT.resolve():
        return []
    out = []
    for pat in WRITERS:
        try:
            r = subprocess.run(["pgrep", "-fl", pat], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        out += [ln for ln in r.stdout.splitlines() if ln.strip() and "fresh_start" not in ln]
    return out


def flush_locked(root: Path) -> bool:
    pid = root / "flush.pid"
    if not pid.exists():
        return False
    fd = os.open(str(pid), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def inventory(root: Path) -> dict:
    """Counts of what stays in the archive. Opens every store read-only."""
    inv = {"projects": 0, "chunks": 0, "pinned": 0, "queued": 0}
    for meta in sorted(root.glob("*/meta.sqlite")):
        inv["projects"] += 1
        try:
            conn = sqlite3.connect(f"file:{meta}?mode=ro", uri=True, timeout=5)
            n, p = conn.execute("SELECT COUNT(*), COALESCE(SUM(pinned = 1), 0) FROM chunks").fetchone()
            conn.close()
        except sqlite3.Error:
            continue
        inv["chunks"] += n
        inv["pinned"] += p
    inv["queued"] = sum(1 for _ in root.glob("*/queue/*.json"))
    return inv


def plan(root: Path, now: float | None = None) -> dict:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    archive = root.parent / "brain-archive" / f"brain-{ts}"
    return {"root": root, "archive": archive, "staging": root.parent / f".brain-fresh-{ts}",
            "carry": [n for n in (*CARRY, "core.sqlite") if (root / n).exists()]}


def check(p: dict, writers: list[str] | None = None) -> None:
    root, archive = p["root"], p["archive"]
    if not root.is_dir():
        raise Refused(f"{root} does not exist; nothing to archive")
    if root.is_symlink():
        raise Refused(f"{root} is a symlink; move its target by hand")
    if archive.exists() or p["staging"].exists():
        raise Refused(f"{archive} already exists; wait a second and run again")
    if root.stat().st_dev != root.parent.stat().st_dev:
        raise Refused(f"{root} is on another filesystem than {root.parent}; the archive would be a copy")
    if flush_locked(root):
        raise Refused("the flush daemon is running (flush.pid is locked); quit Claude Code sessions and retry")
    writers = running_writers(root) if writers is None else writers
    if writers:
        raise Refused("biss-brain processes are running:\n  " + "\n  ".join(writers)
                      + "\nQuit Claude Code sessions, stop the launchd jobs (see the README), then retry")


def _copy_sqlite(src: Path, dest: Path) -> None:
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=10)
    d = sqlite3.connect(str(dest))
    with d:
        s.backup(d)
    s.close()
    d.close()


def apply(p: dict, cfg: dict, writers: list[str] | None = None) -> Path:
    """Build the new root beside the old one, then swap. Returns the archive path."""
    check(p, writers)
    root, archive, staging = p["root"], p["archive"], p["staging"]
    staging.mkdir(mode=0o700)
    try:
        for name in p["carry"]:
            src, dest = root / name, staging / name
            if name == "core.sqlite":
                _copy_sqlite(src, dest)
            elif src.is_dir():
                shutil.copytree(src, dest, symlinks=True)
            else:
                shutil.copy2(src, dest)
        common.write_store_marker(cfg, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    archive.parent.mkdir(parents=True, exist_ok=True)
    os.rename(root, archive)
    try:
        os.rename(staging, root)
    except OSError:
        os.rename(archive, root)  # put the old brain back; the staging dir stays for inspection
        raise
    return archive


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="archive the old brain and create the new one")
    a = ap.parse_args(argv)

    root = Path(common.BRAIN_ROOT)
    cfg = common.load_config()
    p = plan(root)
    fp = common.embed_fingerprint(cfg)
    print(f"brain:    {root}")
    print(f"archive:  {p['archive']}")
    print(f"carried:  {', '.join(p['carry']) or '(nothing)'}")
    print(f"embed:    {fp['model']} ({fp['dim']}-dim)" + (" with task prefixes" if fp["query_prefix"] else ""))
    if root.is_dir():
        inv = inventory(root)
        print(f"archived: {inv['projects']} projects, {inv['chunks']} memories "
              f"({inv['pinned']} pinned), {inv['queued']} queued events")
        if inv["pinned"]:
            print("          pinned memories are not carried; move any you need into core first "
                  "(/biss-brain:remember --core)")
    try:
        if not a.apply:
            check(p)
            print("\ndry run: nothing changed. Run with --apply to do it.")
            return 0
        archived = apply(p, cfg)
    except Refused as e:
        print(f"\nrefused: {e}", file=sys.stderr)
        return 1
    print(f"\ndone. The old brain is at {archived}; the new store starts empty.")
    print("Next: run the doctor (/biss-brain:doctor).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
