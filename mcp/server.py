#!/usr/bin/env python3
"""FastMCP stdio server. Exposes brain tools to the model.

Tools:
  - search_memory(query, kind?, since?, top_k=5, as_of?, include_history?)
                                                  → list of matching chunks
  - recent_sessions(days=7)                       → recent session ids
  - related_to(chunk_id)                          → semantic neighbours
  - pin(chunk_id)                                 → mark immune to decay
  - unpin(chunk_id)
  - core_propose_add(text, section?, layer?)      → queue a core-memory addition
  - core_propose_remove(memory_id, reason)        → queue a core-memory removal

Core memory is injected in full every session, so the model never writes it
directly: both core tools queue a proposal the user reviews with /core review.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add scripts/ to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lancedb  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

import lance_filter as F  # noqa: E402
import provenance  # noqa: E402
import temporal  # noqa: E402
import common  # noqa: E402
from common import (  # noqa: E402
    embed,
    fetch_by_ids,
    lancedb_dir,
    load_config,
    open_meta,
    project_key,
    vector_search,
)
from effects import apply_post_retrieval  # noqa: E402
from backlinks import fetch_related  # noqa: E402
from retrieve import retrieve  # noqa: E402
from recall import _recall_global, _write_cross_link  # noqa: E402

mcp = FastMCP("biss-brain")
_cfg = load_config()


def _embed(text: str) -> list[float]:
    return embed(text, _cfg, purpose="query", timeout=30)


def _table(cwd: str):
    ld = lancedb_dir(cwd)
    if not ld.exists():
        return None
    db = lancedb.connect(str(ld))
    if "chunks" not in db.list_tables().tables:
        return None
    return db.open_table("chunks")


def _parse_since(s: str | None) -> int | None:
    if not s:
        return None
    if s.endswith("d"):
        return int(time.time()) - int(s[:-1]) * 86400
    if s.endswith("h"):
        return int(time.time()) - int(s[:-1]) * 3600
    return int(s)


@mcp.tool()
def search_memory(
    query: str,
    kind: str | None = None,
    since: str | None = None,
    top_k: int = 5,
    as_of: str | None = None,
    include_history: bool = False,
) -> list[dict]:
    """Search the brain. `kind` filters by event kind (stop, subagent_stop,
    pre_compact, decision, note). `since` accepts e.g. "7d", "24h", or a unix ts.

    Memories that a later one contradicted are hidden by default. `as_of`
    ("YYYY-MM-DD", "30d" ago, or a unix ts) returns what was true then.
    `include_history=True` adds to each result the memories it replaced, with
    their valid_from / invalid_at dates."""
    # Quarantined memories (low-trust text that reads like an instruction)
    # are for the user's /recall and the UI only, never the model.
    if kind is not None and (kind not in F.KINDS or kind == "quarantine"):
        kinds = sorted(F.KINDS - {"quarantine"})
        return [{"error": f"unknown kind {kind!r}; one of: {', '.join(kinds)}"}]
    try:
        as_of_ts = temporal.parse_as_of(as_of)
    except ValueError as e:
        return [{"error": str(e)}]
    bad = common.store_mismatch(_cfg)
    if bad:
        return [{"error": f"embed settings differ from the store: {bad}. Fix: {common.MISMATCH_HINT}"}]
    cwd = os.getcwd()
    tbl = _table(cwd)
    if tbl is None:
        return []
    proj = project_key(cwd)
    since_ts = _parse_since(since) if since else None
    qvec = _embed(query)

    meta = open_meta(cwd)
    summary_first = _cfg.get("dream", {}).get("prefer_summaries_in_recall", False) and not kind
    rows = retrieve(
        tbl, meta, proj, query, qvec, _cfg,
        top_k=top_k, kind=kind, since_ts=since_ts, summary_first=summary_first,
        as_of=as_of_ts, include_history=include_history,
    )

    # Reinforce only memories that are still true: recalling an outdated
    # fact must not strengthen it.
    session_id = os.environ.get("CLAUDE_SESSION_ID", "")
    apply_post_retrieval(meta, tbl, proj, [r for r in rows if not r.get("_invalid_at")], _cfg, session_id)
    meta.close()

    local_results = [
        {
            "id": r["id"],
            "kind": r["kind"],
            "ts": r["ts"],
            "session_id": r.get("session_id"),
            "agent_id": r.get("agent_id"),
            "text": r["text"],
            "source": r["_source"],
            "valid_from": temporal.iso(r.get("_valid_from")),
            "invalid_at": temporal.iso(r.get("_invalid_at")),
            **({"history": [
                {**h, "valid_from": temporal.iso(h["valid_from"]), "invalid_at": temporal.iso(h["invalid_at"])}
                for h in r.get("_history", [])
            ]} if include_history else {}),
            "_is_cross_project": False,
        }
        for r in rows
    ]

    # v0.5: cross-project recall fan-out (gated on cross_project.enabled)
    # R3: does NOT bump retrieval_count_local for cross-project hits.
    cross_results = []
    if _cfg.get("cross_project", {}).get("enabled", False):
        global_hits = _recall_global(qvec, _cfg)
        # Exclude any IDs already in local results
        local_ids = {r["id"] for r in local_results}
        for r in global_hits:
            if r["id"] not in local_ids:
                cross_results.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "ts": r["ts"],
                        "session_id": r.get("session_id"),
                        "agent_id": r.get("agent_id"),
                        "text": r.get("text", ""),
                        "source": r.get("_source", "imported"),
                        "source_project": r.get("source_project", "__global__"),
                        "_is_cross_project": True,
                    }
                )

        # R4: write cross-project links from each top-K local chunk to each
        # cross-project hit.  Cap at cfg cross_links limit per local chunk.
        # NEVER writes to related_ids (see _write_cross_link docstring).
        if cross_results and local_results:
            cross_links_limit = (
                _cfg.get("cross_project", {}).get("cross_links", {}).get("limit", 5)
            )
            meta_cl = open_meta(cwd)
            try:
                for local_chunk in local_results:
                    for hit in global_hits[:cross_links_limit]:
                        _write_cross_link(
                            meta_cl,
                            local_chunk["id"],
                            hit["id"],
                            weight=hit.get("_cosine", 0.0),
                        )
                meta_cl.commit()
            finally:
                meta_cl.close()

    return local_results + cross_results


@mcp.tool()
def recent_sessions(days: int = 7) -> list[dict]:
    """Distinct session ids active within the last N days for this project."""
    cwd = os.getcwd()
    proj = project_key(cwd)
    cutoff = int(time.time()) - days * 86400
    meta = open_meta(cwd)
    rows = meta.execute(
        "SELECT session_id, MAX(ts) as last_ts, COUNT(*) as n "
        "FROM chunks WHERE project = ? AND ts >= ? AND session_id != '' "
        "GROUP BY session_id ORDER BY last_ts DESC LIMIT 50",
        (proj, cutoff),
    ).fetchall()
    meta.close()
    return [
        {"session_id": r[0], "last_ts": r[1], "chunks": r[2]} for r in rows
    ]


@mcp.tool()
def related_to(chunk_id: str, top_k: int = 5) -> list[dict]:
    """Semantic neighbours of a given chunk (excluding itself).

    Reads pre-computed backlinks from SQLite when available (fast, deterministic).
    Falls back to on-demand vector similarity search for legacy chunks with no
    stored related_ids.
    """
    bad = common.store_mismatch(_cfg)
    if bad:
        return [{"error": f"embed settings differ from the store: {bad}. Fix: {common.MISMATCH_HINT}"}]
    cwd = os.getcwd()
    tbl = _table(cwd)
    if tbl is None:
        return []

    proj = project_key(cwd)
    # Fast path: pre-computed backlinks
    related_ids = fetch_related(cwd, chunk_id)
    if related_ids:
        rows = fetch_by_ids(tbl, related_ids, limit=len(related_ids))
        rows = [r for r in rows if r.get("kind") not in ("hypothesis", "quarantine")][:top_k]
    else:
        # Fallback: on-demand similarity search (legacy chunks without backlinks)
        seed = fetch_by_ids(tbl, [chunk_id], limit=1)
        if not seed:
            return []
        qvec = seed[0]["vector"]
        rows = vector_search(
            tbl, qvec, F.live_in(proj, F.NOT_HYPOTHESIS, F.NOT_QUARANTINE, F.ne("id", chunk_id)), top_k
        )
    meta = open_meta(cwd)
    try:
        provs = provenance.lookup(meta, proj, [r["id"] for r in rows])
    finally:
        meta.close()
    return [
        {"id": r["id"], "kind": r["kind"], "text": r["text"], "ts": r["ts"],
         "source": provenance.label(provs.get(r["id"]))}
        for r in rows
    ]


def _set_pinned(chunk_id: str, value: int) -> dict:
    """Pin state is only ever changed for a chunk of the caller's project."""
    cwd = os.getcwd()
    meta = open_meta(cwd)
    try:
        cur = meta.execute(
            "UPDATE chunks SET pinned = ? WHERE id = ? AND project = ?",
            (value, chunk_id, project_key(cwd)),
        )
        meta.commit()
        found = cur.rowcount > 0
    finally:
        meta.close()
    if not found:
        return {"id": chunk_id, "error": "no such chunk in this project"}
    return {"id": chunk_id, "pinned": bool(value)}


@mcp.tool()
def pin(chunk_id: str) -> dict:
    """Pin a chunk so it survives decay and ranks higher in retrieval."""
    return _set_pinned(chunk_id, 1)


@mcp.tool()
def unpin(chunk_id: str) -> dict:
    return _set_pinned(chunk_id, 0)


@mcp.tool()
def core_propose_add(text: str, section: str = "Decisions", layer: str = "private") -> dict:
    """Propose a durable fact for core memory (always-in-context notes).

    Nothing changes until the user accepts it with /biss-brain:core review.
    section: Decisions | Conventions | Lessons | Pinned. layer: private (this
    repo) | global (every project).
    """
    import core

    cwd = os.getcwd()
    try:
        pid = core.propose(project_key(cwd), "add", "mcp", "assistant",
                           layer=layer, section=section.strip().capitalize(), text=text)
    except core.CoreError as e:
        return {"error": str(e)}
    return {"proposal": pid, "status": "queued for user review" if pid else "already proposed"}


@mcp.tool()
def core_propose_remove(memory_id: str, reason: str = "") -> dict:
    """Propose retiring a core memory by its m:id (shown in the injected core block).

    Nothing changes until the user accepts it with /biss-brain:core review.
    """
    import core

    cwd = os.getcwd()
    mid = memory_id.strip().removeprefix("m:")
    if core.find(cwd, mid) is None:
        return {"error": f"no core memory {mid!r} in this project"}
    pid = core.propose(project_key(cwd), "remove", "mcp", "assistant", target=mid, text=reason)
    return {"proposal": pid, "status": "queued for user review" if pid else "already proposed"}


if __name__ == "__main__":
    mcp.run()
