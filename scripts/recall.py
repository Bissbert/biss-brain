#!/usr/bin/env python3
"""CLI used by the /biss-brain:recall slash command.

Usage:
  recall.py <query>
  recall.py --kind decision --since 7d <query>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import lancedb

import json

import lancedb as _lancedb_pkg

import common
from common import (
    BRAIN_ROOT,
    embed,
    lancedb_dir,
    load_config,
    now_unix,
    open_meta,
    project_key,
    scoped,
)
import temporal
from effects import apply_post_retrieval
from retrieve import retrieve


def _embed(text: str, cfg: dict) -> list[float]:
    return embed(text, cfg, purpose="query", timeout=30)


def _parse_since(s: str) -> int:
    if s.endswith("d"):
        return int(time.time()) - int(s[:-1]) * 86400
    if s.endswith("h"):
        return int(time.time()) - int(s[:-1]) * 3600
    return int(s)


def _open_global_lance():
    """Open the _global LanceDB table. Returns (db, tbl) or (None, None)."""
    global_pdir = BRAIN_ROOT / "_global"
    lance_dir = global_pdir / "lancedb"
    if not lance_dir.exists():
        return None, None
    try:
        db = _lancedb_pkg.connect(str(lance_dir))
        if "chunks" not in db.list_tables().tables:
            return db, None
        return db, db.open_table("chunks")
    except Exception:  # noqa: BLE001
        return None, None


def _recall_global(query_embedding: list[float], cfg: dict) -> list[dict]:
    """Fan-out recall to the _global pool (retrieve.global_candidates).

    Called AFTER local recall. Returns up to 10 rows that pass the cosine gate
    cfg["cross_project"]["recall"]["cosine_gate"] (default 0.85), each with
    source_project (where it came from) and _is_cross_project=True.

    R3 invariant: does NOT bump retrieval_count_local for cross-project hits.
    """
    from retrieve import global_candidates  # noqa: PLC0415

    gate = float(cfg.get("cross_project", {}).get("recall", {}).get("cosine_gate", 0.85))
    _, tbl = _open_global_lance()
    if tbl is None:
        return []
    return global_candidates("", query_embedding, cfg, 10, min_cosine=gate, tbl=tbl)


def _write_cross_link(
    meta,
    src_id: str,
    global_id: str,
    weight: float,
    ts: int | None = None,
    *,
    project: str | None = None,
) -> None:
    """Write a cross-project link from src_id to global_id into cross_links column.

    Uses the JSON-encoded cross_links column. NEVER writes to related_ids (R4).

    The cross_links schema is a JSON list of dicts:
      [{"global_id": "...", "weight": ..., "ts": ...}]

    Idempotent: if global_id is already in cross_links the entry is updated
    in-place with the new weight.
    """
    if ts is None:
        ts = now_unix()

    row = meta.execute(*scoped("SELECT cross_links FROM chunks WHERE id = ?", (src_id,), project)).fetchone()
    if row is None:
        return

    raw = row[0] or "[]"
    try:
        links: list[dict] = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        links = []

    # Upsert: update existing entry or append new one
    updated = False
    for entry in links:
        if entry.get("global_id") == global_id:
            entry["weight"] = weight
            entry["ts"] = ts
            updated = True
            break
    if not updated:
        links.append({"global_id": global_id, "weight": weight, "ts": ts})

    meta.execute(*scoped(
        "UPDATE chunks SET cross_links = ? WHERE id = ?",
        (json.dumps(links, ensure_ascii=False), src_id),
        project,
    ))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("query", type=str)
    p.add_argument("--kind", default=None)
    p.add_argument("--since", default=None, help="e.g. 7d, 24h, or unix ts")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--as-of", default=None, help="what was true then: YYYY-MM-DD, 30d (ago), or unix ts")
    p.add_argument("--history", action="store_true", help="show the memories each result replaced")
    args = p.parse_args()

    cfg = load_config()
    bad = common.store_mismatch(cfg)
    if bad:
        print(f"(error: embed settings differ from the store: {bad}. Fix: {common.MISMATCH_HINT})")
        return 2
    cwd = os.getcwd()
    ld = lancedb_dir(cwd)
    if not ld.exists():
        print("(no memory yet for this project)")
        return 0

    db = lancedb.connect(str(ld))
    if "chunks" not in db.list_tables().tables:
        print("(no memory yet for this project)")
        return 0
    tbl = db.open_table("chunks")
    proj = project_key(cwd)

    qtext = args.query
    try:
        as_of = temporal.parse_as_of(args.as_of)
    except ValueError as e:
        print(f"(error: {e})")
        return 2
    qvec = _embed(qtext, cfg)

    since_ts = _parse_since(args.since) if args.since else None
    summary_first = (
        cfg.get("dream", {}).get("prefer_summaries_in_recall", False) and not args.kind
    )
    meta = open_meta(cwd)
    results = retrieve(
        tbl, meta, proj, qtext, qvec, cfg,
        top_k=args.top_k, kind=args.kind, since_ts=since_ts, summary_first=summary_first,
        as_of=as_of, include_history=args.history,
    )

    if not results:
        meta.close()
        print(f"(no matches for: {qtext})")
        return 0

    live = [r for r in results if not r.get("_invalid_at")]
    apply_post_retrieval(meta, tbl, proj, live, cfg, os.environ.get("CLAUDE_SESSION_ID", ""))
    meta.close()

    for r in results:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        until = f"  valid until {temporal.iso(r['_invalid_at'])}" if r.get("_invalid_at") else ""
        print(f"--- [{r['kind']}] {ts}  id={r['id'][:8]}  [source: {r['_source']}]{until}")
        print(r["text"])
        for h in r.get("_history", []):
            print(f"    replaced: {temporal.iso(h['valid_from'])} → {temporal.iso(h['invalid_at'])}"
                  f"  id={h['id'][:8]}  {h['text']}")
        print()

    # v0.5: cross-project recall fan-out (gated on cross_project.enabled)
    if cfg.get("cross_project", {}).get("enabled", False):
        cross_results = _recall_global(qvec, cfg)
        if cross_results:
            print("\n=== Cross-project results ===")
            for r in cross_results:
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
                source = r.get("source_project", "__global__")
                print(f"--- [{r['kind']}] {ts}  id={r['id'][:8]}  project={source}  [source: {r['_source']}]")
                print(r.get("text", ""))
                print()

            # R4: write cross-project links from each top-K local chunk to each
            # cross-project hit.  Cap at cfg cross_links limit per local chunk.
            # NEVER writes to related_ids (see _write_cross_link docstring).
            if results:
                cross_links_limit = (
                    cfg.get("cross_project", {}).get("cross_links", {}).get("limit", 5)
                )
                meta2 = open_meta(cwd)
                try:
                    for local_id in (r["id"] for r in results):
                        for hit in cross_results[:cross_links_limit]:
                            _write_cross_link(
                                meta2,
                                local_id,
                                hit["id"],
                                weight=hit.get("_cosine", 0.0),
                                project=proj,
                            )
                    meta2.commit()
                finally:
                    meta2.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
