#!/usr/bin/env python3
"""Mark one or more chunks as superseded. Soft-delete; chunks remain queryable
via explicit id but never surface in default search."""
from __future__ import annotations

import os
import sys

import lancedb

import lance_filter as F
from common import lancedb_dir, open_meta, project_key, tombstone_id


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: forget.py <chunk_id> [<chunk_id> ...]", file=sys.stderr)
        return 1
    cwd = os.getcwd()
    proj = project_key(cwd)
    meta = open_meta(cwd)
    tomb = tombstone_id()  # opaque tombstone id

    # Collect full ids for LanceDB update
    full_ids: list[str] = []
    for cid in sys.argv[1:]:
        # Allow short prefix match within this project; ids are hex
        if not cid or not all(c in "0123456789abcdef-" for c in cid.lower()):
            print(f"  not an id: {cid}")
            continue
        row = meta.execute(
            "SELECT id FROM chunks WHERE project = ? AND id LIKE ? LIMIT 1",
            (proj, cid.lower() + "%"),
        ).fetchone()
        if not row:
            print(f"  no match: {cid}")
            continue
        meta.execute(
            "UPDATE chunks SET superseded_by = ? WHERE id = ? AND project = ?",
            (tomb, row[0], proj),
        )
        full_ids.append(row[0])
        print(f"  forgot: {row[0][:8]}")
    meta.commit()
    meta.close()

    # Mirror superseded_by update into LanceDB so vector-side filters stay consistent
    if full_ids:
        ld = lancedb_dir(cwd)
        if ld.exists():
            try:
                db = lancedb.connect(str(ld))
                if "chunks" in db.list_tables().tables:
                    tbl = db.open_table("chunks")
                    tbl.update(
                        where=F.isin("id", full_ids),
                        values={"superseded_by": tomb},
                    )
            except Exception as e:  # noqa: BLE001
                print(f"  warn: LanceDB update failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
