"""Post-retrieval side effects, shared by recall, MCP search and prompt inject.

After a read returns rows, the brain records that they were used:
  - retrieval_count / last_accessed (legacy) and retrieval_count_local
  - STDP: one edge_bumps row per co-retrieved pair
  - RIF: siblings that share many dense neighbours with a hit decay slightly
  - labile window: hits may be reconsolidated by new evidence for a while

Every write is scoped to the caller's project (SEC-005).
"""
from __future__ import annotations

import time
from typing import Any

import lance_filter as F
from common import vector_search


def enqueue_co_retrievals(conn, project: str, retrieved_ids: list[str], session_id: str) -> None:
    """Insert one edge_bumps row per canonical (min, max) pair of the retrieved set."""
    if len(retrieved_ids) < 2:
        return
    now = int(time.time())
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, str, int, str]] = []
    for i, a in enumerate(retrieved_ids):
        for b in retrieved_ids[i + 1:]:
            pair = (min(a, b), max(a, b))
            if a != b and pair not in seen:
                seen.add(pair)
                rows.append((project, pair[0], pair[1], now, session_id))
    if rows:
        conn.executemany(
            "INSERT INTO edge_bumps (project, chunk_a, chunk_b, bumped_at, session_id)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def apply_rif(
    conn,
    tbl: Any,
    project: str,
    retrieved_ids: list[str],
    retrieved_embeddings: dict[str, list[float]],
    cfg: dict,
) -> None:
    """Retrieval-induced forgetting on siblings of the retrieved chunks.

    A sibling of hit A is one of A's top-10 dense neighbours (same project, not
    itself retrieved) that shares >= rif.shared_neighbour_threshold of A's
    neighbours. Its decay_score is multiplied by rif.decay_multiplier, floored
    at forget.decay_floor. Runs inline: about 11 ANN queries per hit.
    """
    if not cfg.get("rif", {}).get("enabled", True):
        return
    threshold = cfg["rif"]["shared_neighbour_threshold"]
    multiplier = cfg["rif"]["decay_multiplier"]
    floor = cfg.get("forget", {}).get("decay_floor", 0.05)
    top_n = 10
    retrieved_set = set(retrieved_ids)

    for chunk_id, emb in retrieved_embeddings.items():
        try:
            nbr_rows = vector_search(tbl, emb, F.and_(F.eq("project", project), F.ne("id", chunk_id)), top_n)
        except Exception:  # noqa: BLE001
            continue
        neighbour_ids_a = {r["id"] for r in nbr_rows}
        if not neighbour_ids_a:
            continue

        sibling_ids: list[str] = []
        for nbr in nbr_rows:
            nbr_id = nbr["id"]
            if nbr_id in retrieved_set or nbr.get("vector") is None:
                continue
            try:
                nbr_nbr_rows = vector_search(
                    tbl, nbr["vector"], F.and_(F.eq("project", project), F.ne("id", nbr_id)), top_n
                )
            except Exception:  # noqa: BLE001
                continue
            if len(neighbour_ids_a & {r["id"] for r in nbr_nbr_rows}) >= threshold:
                sibling_ids.append(nbr_id)

        if sibling_ids:
            placeholders = ",".join("?" * len(sibling_ids))
            conn.execute(
                "UPDATE chunks SET decay_score = MAX(?, decay_score * ?)"
                f" WHERE project = ? AND id IN ({placeholders}) AND pinned = 0",
                (floor, multiplier, project, *sibling_ids),
            )


def stamp_labile(conn, project: str, retrieved_ids: list[str], cfg: dict) -> None:
    """Set labile_until = now + labile_window_seconds on every retrieved chunk."""
    if not retrieved_ids:
        return
    labile_until = int(time.time()) + cfg["reconsolidation"]["labile_window_seconds"]
    placeholders = ",".join("?" * len(retrieved_ids))
    conn.execute(
        f"UPDATE chunks SET labile_until = ? WHERE project = ? AND id IN ({placeholders})",
        (labile_until, project, *retrieved_ids),
    )


def bump_retrieval_count_local(conn, project: str, retrieved_ids: list[str]) -> None:
    """Increment retrieval_count_local (not legacy retrieval_count) for local hits."""
    if not retrieved_ids:
        return
    placeholders = ",".join("?" * len(retrieved_ids))
    conn.execute(
        "UPDATE chunks SET retrieval_count_local = retrieval_count_local + 1"
        f" WHERE project = ? AND id IN ({placeholders})",
        (project, *retrieved_ids),
    )


def bump_access(conn, project: str, retrieved_ids: list[str]) -> None:
    """Legacy retrieval_count + last_accessed."""
    if not retrieved_ids:
        return
    placeholders = ",".join("?" * len(retrieved_ids))
    conn.execute(
        "UPDATE chunks SET retrieval_count = retrieval_count + 1, last_accessed = ?"
        f" WHERE project = ? AND id IN ({placeholders})",
        (int(time.time()), project, *retrieved_ids),
    )


def touch_injected(conn, project: str, ids: list[str]) -> None:
    """Side effect of a proactive (UserPromptSubmit) injection: last_accessed only.

    The model did not ask for these, so they don't count as retrievals: no
    retrieval_count (which feeds dream's core proposals), no STDP pairing,
    no retrieval-induced forgetting of neighbours, no labile window.
    """
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"UPDATE chunks SET last_accessed = ? WHERE project = ? AND id IN ({placeholders})",
        (int(time.time()), project, *ids),
    )
    conn.commit()


def apply_post_retrieval(conn, tbl: Any, project: str, rows: list[dict], cfg: dict, session_id: str) -> None:
    """All side effects of a local read, in one transaction (committed here)."""
    ids = [r["id"] for r in rows]
    if not ids:
        return
    embeddings = {r["id"]: r["vector"] for r in rows if r.get("vector") is not None}
    bump_access(conn, project, ids)
    if cfg.get("stdp", {}).get("enabled", True):
        enqueue_co_retrievals(conn, project, ids, session_id)
    apply_rif(conn, tbl, project, ids, embeddings, cfg)
    stamp_labile(conn, project, ids, cfg)
    bump_retrieval_count_local(conn, project, ids)
    conn.commit()
