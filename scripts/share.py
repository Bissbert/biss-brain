#!/usr/bin/env python3
"""Share memories of this project with every project, or take them back.

  share.py <chunk_id> ...         pool each one now (pinned, scope=general)
  share.py --off <chunk_id> ...   take each one out of the pool; it stays here

A shared memory is checked for secrets only; tool and web output, hypotheses
and quarantined notes never leave their project. --off keeps the pin and the
general tag but stops the nightly dream from pooling it again; sharing it
again lifts that.
"""
from __future__ import annotations

import os
import sys

import globalpool
from common import load_config, open_meta, project_key

_WHY = {
    "saved": "marked; sharing is off (cross_project.enabled), so it is pooled once that is on",
    "missing": "no live memory with that id",
    "untrusted": "came from tool or web output, which stays in its project",
    "kind": "this kind of memory never leaves its project",
    "secret": "looks like it holds a secret, so it stays here",
}


def main() -> int:
    args = sys.argv[1:]
    off = bool(args) and args[0] == "--off"
    ids = args[1:] if off else args
    if not ids:
        print("usage: share.py [--off] <chunk_id> [<chunk_id> ...]", file=sys.stderr)
        return 1
    cwd = os.getcwd()
    proj = project_key(cwd)
    cfg = load_config()
    meta = open_meta(cwd)
    try:
        full = []
        for cid in ids:
            if not cid or not all(c in "0123456789abcdef-" for c in cid.lower()):
                print(f"  not an id: {cid}")
                continue
            row = meta.execute("SELECT id FROM chunks WHERE project = ? AND id LIKE ? LIMIT 1",
                               (proj, cid.lower() + "%")).fetchone()
            if row is None:
                print(f"  no match: {cid}")
            else:
                full.append(row[0])
    finally:
        meta.close()
    for cid in full:
        if off:
            globalpool.unshare(proj, cid)
            print(f"  unshared: {cid[:8]} (kept in this project)")
            continue
        status, _ = globalpool.share(proj, cid, cfg)
        print(f"  shared: {cid[:8]}" if status == "shared"
              else f"  not shared: {cid[:8]}: {_WHY.get(status, status)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
