#!/usr/bin/env python3
"""Print brain stats for the current project. Backs the /biss-brain:stats command."""
from __future__ import annotations

import os

from common import open_meta, project_dir, project_key


def main() -> int:
    cwd = os.getcwd()
    proj = project_key(cwd)
    pdir = project_dir(cwd)
    meta = open_meta(cwd)
    total = meta.execute(
        "SELECT COUNT(*) FROM chunks WHERE project = ?", (proj,)
    ).fetchone()[0]
    by_kind = meta.execute(
        "SELECT kind, COUNT(*) FROM chunks WHERE project = ? GROUP BY kind ORDER BY 2 DESC",
        (proj,),
    ).fetchall()
    pinned = meta.execute(
        "SELECT COUNT(*) FROM chunks WHERE project = ? AND pinned = 1", (proj,)
    ).fetchone()[0]
    superseded = meta.execute(
        "SELECT COUNT(*) FROM chunks WHERE project = ? AND superseded_by != ''",
        (proj,),
    ).fetchone()[0]
    top = meta.execute(
        "SELECT id, kind, retrieval_count FROM chunks "
        "WHERE project = ? ORDER BY retrieval_count DESC LIMIT 5",
        (proj,),
    ).fetchall()
    meta.close()

    print(f"project_key:  {proj}")
    print(f"storage:      {pdir}")
    print(f"total chunks: {total}    pinned: {pinned}    superseded: {superseded}")
    print("by kind:")
    for k, n in by_kind:
        print(f"  {k:20s} {n}")
    if top:
        print("top retrieved:")
        for cid, k, rc in top:
            print(f"  {cid[:8]}  {k:18s}  retrieved={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
