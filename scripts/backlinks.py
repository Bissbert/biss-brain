"""A-MEM-style backlink helper.

Public API:
  write_backlinks(cwd, proj, chunk_id, vec, cfg) -> None
  parse_related_ids(raw)              -> list[dict]   (v0.5 typed adapter)
  serialize_related_ids(items)        -> str           (v0.5 symmetry helper)

Called after every insert in flush.py and remember.py.
Finds top-3 same-project, active, similar chunks (cosine >= 0.7 proxy)
and writes bidirectional related_ids in SQLite.
"""
from __future__ import annotations

import json
import logging
import sqlite3

import lancedb

import lance_filter as F
from common import lancedb_dir, now_unix, open_meta, project_key, scoped, vector_search

log = logging.getLogger("biss-brain.backlinks")

SIMILARITY_THRESHOLD = 0.70  # cosine gate for a backlink
MAX_RELATED = 10


# ── Legacy codec (plain comma-separated string) ───────────────────────────


def _parse_ids(raw: str) -> list[str]:
    """Parse comma-separated id string, filtering empty strings."""
    return [x.strip() for x in raw.split(",") if x.strip()]


def _serialize_ids(ids: list[str]) -> str:
    return ",".join(ids)


def _cap(ids: list[str]) -> list[str]:
    """Keep only the newest MAX_RELATED entries (rightmost = newest)."""
    return ids[-MAX_RELATED:]


# ── v0.5 typed related_ids codec ─────────────────────────────────────────


def parse_related_ids(raw: str | None) -> list[dict]:
    """Parse a related_ids JSON string. Returns a list of typed dicts.

    Tolerant read adapter for both legacy and v0.5+ typed shapes:
      - Legacy (plain list of strings): ["id1", "id2"]
        -> [{id, type="co_occurs", weight=1.0, ts=0}, ...]
      - Typed (list of dicts): [{"id": ..., "type": ..., "weight": ..., "ts": ...}, ...]
        -> returned as-is
      - Legacy comma-separated (pre-JSON): stored as plain TEXT, not JSON
        -> each token promoted to {id, type="co_occurs", weight=1.0, ts=0}

    A None or empty string returns an empty list.
    """
    if not raw:
        return []
    # Fast path: try JSON decode first (new shape and legacy JSON list)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Fallback: legacy comma-separated plain text (pre-v0.5 format)
        ids = _parse_ids(raw)
        return [{"id": x, "type": "co_occurs", "weight": 1.0, "ts": 0} for x in ids]
    if not data:
        return []
    if isinstance(data[0], str):
        # JSON list of plain strings — legacy shape
        return [{"id": x, "type": "co_occurs", "weight": 1.0, "ts": 0} for x in data]
    # Already typed dicts — return as-is
    return data


def serialize_related_ids(items: list[dict]) -> str:
    """Serialise a list of typed related_id dicts to a JSON string.

    Symmetry helper for parse_related_ids. Writers (B2 Edges) call this;
    parse_related_ids is the canonical read path.
    """
    return json.dumps(items, ensure_ascii=False)


# ── v0.5 typed edge writer ────────────────────────────────────────────────


def _cap_typed(items: list[dict]) -> list[dict]:
    """Keep only the newest MAX_RELATED typed entries."""
    return items[-MAX_RELATED:]


def _add_typed_edge(
    conn: sqlite3.Connection,
    chunk_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    now_ts: int,
    cap: int = MAX_RELATED,
    *,
    project: str | None = None,
) -> None:
    """Read existing typed related_ids for chunk_id, upsert the edge, write back.

    De-duplicates by (id, type): on hit refreshes weight and ts; on miss appends.
    Respects cap (default MAX_RELATED=10, keeps newest entries).
    Read-compatible with legacy plain-string entries via parse_related_ids.
    With project set, both the read and the write are limited to that project.
    """
    row = conn.execute(*scoped("SELECT related_ids FROM chunks WHERE id = ?", (chunk_id,), project)).fetchone()
    raw = row[0] if row else None
    items: list[dict] = parse_related_ids(raw)

    # Upsert: find existing entry with same (id, type) pair
    updated = False
    for entry in items:
        if entry.get("id") == target_id and entry.get("type") == edge_type:
            entry["weight"] = weight
            entry["ts"] = now_ts
            updated = True
            break

    if not updated:
        items.append({"id": target_id, "type": edge_type, "weight": weight, "ts": now_ts})

    # Apply cap (keep newest = rightmost)
    if len(items) > cap:
        items = items[-cap:]

    conn.execute(*scoped(
        "UPDATE chunks SET related_ids = ? WHERE id = ?",
        (serialize_related_ids(items), chunk_id),
        project,
    ))


# ── Updated write_backlinks (emits typed dicts) ───────────────────────────


def write_backlinks(cwd: str, proj: str, chunk_id: str, vec: list[float], cfg: dict) -> None:  # noqa: ARG001
    """Find ≤3 similar neighbours and write bidirectional typed related_ids.

    Emits typed {id, type, weight, ts} dicts for new entries.
    Legacy plain-string entries in existing DBs round-trip unchanged
    via parse_related_ids on next read.

    Never raises — errors are logged and silently ignored.
    """
    try:
        _write_backlinks_inner(cwd, proj, chunk_id, vec)
    except Exception as e:  # noqa: BLE001
        log.warning("backlinks write failed (non-fatal): %s", e)


def _write_backlinks_inner(cwd: str, proj: str, chunk_id: str, vec: list[float]) -> None:
    db = lancedb.connect(str(lancedb_dir(cwd)))
    if "chunks" not in db.list_tables().tables:
        return
    tbl = db.open_table("chunks")

    candidates = vector_search(tbl, vec, F.live_in(proj, F.ne("id", chunk_id)), 3)
    if not candidates:
        return

    neighbours = [c for c in candidates if c["_cosine"] >= SIMILARITY_THRESHOLD]
    if not neighbours:
        return

    neighbour_ids = [c["id"] for c in neighbours]
    ts = now_unix()

    meta = open_meta(cwd)
    try:
        # Update new chunk's related_ids to include neighbour ids (typed)
        _add_typed_edge_multi(meta, chunk_id, neighbour_ids, "co_occurs", 1.0, ts, project=proj)

        # Update each neighbour's related_ids to include new chunk's id (typed)
        for nid in neighbour_ids:
            _add_typed_edge(meta, nid, chunk_id, "co_occurs", 1.0, ts, project=proj)

        meta.commit()
        log.debug(
            "backlinks: chunk %s linked to %s neighbour(s)",
            chunk_id[:8],
            len(neighbour_ids),
        )
    finally:
        meta.close()


def _add_typed_edge_multi(
    conn: sqlite3.Connection,
    chunk_id: str,
    target_ids: list[str],
    edge_type: str,
    weight: float,
    now_ts: int,
    *,
    project: str | None = None,
) -> None:
    """Add multiple typed edges to chunk_id in a single read-modify-write."""
    row = conn.execute(*scoped("SELECT related_ids FROM chunks WHERE id = ?", (chunk_id,), project)).fetchone()
    raw = row[0] if row else None
    items: list[dict] = parse_related_ids(raw)

    for target_id in target_ids:
        # Upsert: de-duplicate by (id, type)
        found = False
        for entry in items:
            if entry.get("id") == target_id and entry.get("type") == edge_type:
                entry["weight"] = weight
                entry["ts"] = now_ts
                found = True
                break
        if not found:
            items.append({"id": target_id, "type": edge_type, "weight": weight, "ts": now_ts})

    # Cap
    if len(items) > MAX_RELATED:
        items = items[-MAX_RELATED:]

    conn.execute(*scoped(
        "UPDATE chunks SET related_ids = ? WHERE id = ?",
        (serialize_related_ids(items), chunk_id),
        project,
    ))


def fetch_related(cwd: str, chunk_id: str) -> list[str]:
    """Return the stored related_ids for a chunk as plain id strings (may be empty).

    Reads typed or legacy format transparently via parse_related_ids.
    """
    meta = open_meta(cwd)
    try:
        row = meta.execute(
            "SELECT related_ids FROM chunks WHERE id = ? AND project = ?",
            (chunk_id, project_key(cwd)),
        ).fetchone()
        if not row or not row[0]:
            return []
        typed = parse_related_ids(row[0])
        return [e["id"] for e in typed]
    finally:
        meta.close()
