"""Shared data helpers for biss-brain UI panel agents.

Architecture note
-----------------
There are TWO separate stores per project:

  meta.sqlite   – metadata-only: id, kind, ts, session_id, agent_id,
                  superseded_by, pinned, retrieval_count, decay_score,
                  related_ids, content_hash.  NO chunk text here.

  lancedb/      – chunk text lives in the "chunks" table, column "text".
                  Also stores the dense embedding vector (768-dim) and
                  redundant metadata columns used for WHERE filtering.

Panel agents MUST use fetch_chunks_meta() for metadata queries and
fetch_chunk_texts() / hybrid_search() for anything touching text.
Never look for a "text" or "content" column in SQLite – it does not exist.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup – allow `from common import ...` when imported from ui/ context.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent      # ui/
_ROOT = _HERE.parent                          # plugin root
_SCRIPTS = _ROOT / "scripts"
for _p in (_ROOT, _SCRIPTS):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import lancedb  # noqa: E402 – after sys.path setup
from fastapi import HTTPException  # noqa: E402

import lance_filter as F  # noqa: E402
from common import (  # noqa: E402
    BRAIN_ROOT,
    GLOBAL_PROJECT_HASH,
    embed,
    fetch_by_ids,
    hybrid_search as _common_hybrid_search,
    load_config,
    tombstone_id,  # re-export
    vector_search,
)
from retrieve import relevance  # noqa: E402
from scrub import scrub  # noqa: E402

# ---------------------------------------------------------------------------
# LanceDB connection cache – safe for concurrent read-only access.
# Keyed by project hash so we open one connection per project per process.
# ---------------------------------------------------------------------------
_lance_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Project path helpers
# ---------------------------------------------------------------------------

def _project_dir(project_hash: str) -> Path:
    """Resolve project_hash to its on-disk directory.

    GLOBAL_PROJECT_HASH maps to BRAIN_ROOT/_global; every other hash
    maps to BRAIN_ROOT/<hash>. Centralised here so every UI helper
    agrees with the CLI tools (dream.py, recall.py, forget.py).
    """
    if project_hash == GLOBAL_PROJECT_HASH:
        return BRAIN_ROOT / "_global"
    return BRAIN_ROOT / project_hash


def project_paths(project_hash: str) -> tuple[Path, Path]:
    """Return (sqlite_path, lancedb_path) for a project hash.

    Raises HTTPException 404 if the project directory does not exist.
    """
    pdir = _project_dir(project_hash)
    if not pdir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown project hash: {project_hash}")
    return pdir / "meta.sqlite", pdir / "lancedb"


def validate_project(project_hash: str) -> Path:
    """Return the project dir Path, raising HTTPException 404 if unknown.

    Extracted from server.py so router modules can call this directly.
    """
    pdir = _project_dir(project_hash)
    if not pdir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown project hash: {project_hash}")
    return pdir


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def open_meta_ro(project_hash: str) -> sqlite3.Connection:
    """Open meta.sqlite read-only for the given project hash.

    Caller is responsible for calling conn.close().
    Raises HTTPException 404 if the database file does not exist.
    """
    sqlite_path, _ = project_paths(project_hash)
    if not sqlite_path.exists():
        raise HTTPException(status_code=404, detail="meta.sqlite not found for this project")
    conn = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


_META_COLUMNS = (
    "id, kind, ts, session_id, agent_id, superseded_by, pinned, "
    "retrieval_count, decay_score, related_ids"
)


def fetch_chunks_meta(
    conn: sqlite3.Connection,
    where_sql: str = "1=1",
    params: tuple = (),
    limit: int = 200,
    offset: int = 0,
) -> list[dict]:
    """Fetch chunk metadata rows from the open SQLite connection.

    Selects: id, kind, ts, session_id, agent_id, superseded_by, pinned,
             retrieval_count, decay_score, related_ids
    Ordered by ts DESC.  Returns a list of plain dicts.
    """
    query = (
        f"SELECT {_META_COLUMNS} FROM chunks "
        f"WHERE {where_sql} "
        f"ORDER BY ts DESC LIMIT ? OFFSET ?"
    )
    rows = conn.execute(query, (*params, limit, offset)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# LanceDB
# ---------------------------------------------------------------------------

def open_lance(project_hash: str) -> Any | None:
    """Return a cached LanceDB Table for the project's 'chunks' table.

    Returns None if the lancedb directory or 'chunks' table does not yet
    exist (empty/new project).  Result is cached per-process.
    """
    if project_hash in _lance_cache:
        return _lance_cache[project_hash]

    _, lance_path = project_paths(project_hash)
    if not lance_path.exists():
        return None

    # The cached table must follow writers: they prune versions older than ten
    # minutes (common.VERSION_RETENTION), and a pinned old version would vanish.
    db = lancedb.connect(str(lance_path), read_consistency_interval=timedelta(seconds=5))
    tables = db.list_tables()
    # list_tables() returns either a plain list or an object with .tables
    table_names: list[str] = tables.tables if hasattr(tables, "tables") else list(tables)
    if "chunks" not in table_names:
        return None

    tbl = db.open_table("chunks")
    _lance_cache[project_hash] = tbl
    return tbl


def fetch_chunk_texts(table: Any, ids: list[str]) -> dict[str, str]:
    """Bulk-fetch chunk text from LanceDB by id list.

    The text column in LanceDB is named 'text' (not 'content').
    Uses a single WHERE id IN (...) query, not per-row round trips.
    Returns {id: text}.  Missing ids are silently omitted.
    """
    if not ids or table is None:
        return {}
    safe_ids = [safe_id(i) for i in ids]
    safe_ids = [i for i in safe_ids if i is not None]
    if not safe_ids:
        return {}
    rows = fetch_by_ids(table, safe_ids)
    return {r["id"]: r["text"] for r in rows if "text" in r}


def hybrid_search(
    table: Any,
    query: str,
    limit: int = 20,
    where_sql: str | None = None,
) -> list[dict]:
    """Hybrid BM25 + dense vector search over the chunks LanceDB table.

    Generates an embedding via Ollama (or fake embed when
    BISS_BRAIN_FAKE_EMBED is set), then delegates to the shared
    hybrid_search() implementation in scripts/common.py which handles
    the fluent API call and graceful fallback to pure dense search.

    Returns rows with all LanceDB columns including id, text, kind,
    plus _score: the rank relevance (RRF score, or cosine after a fallback).
    """
    if table is None:
        return []

    cfg = load_config()
    qvec = _embed(query, cfg)

    rows = _common_hybrid_search(table, qvec, query, where_sql, limit)

    return [{**r, "_score": relevance(r)} for r in rows]


# ---------------------------------------------------------------------------
# Internal: embedding helper (mirrors recall.py)
# ---------------------------------------------------------------------------

def _embed(text: str, cfg: dict) -> list[float]:
    return embed(text, cfg, purpose="query", timeout=30)

# ---------------------------------------------------------------------------
# Module-level cache for fetch_kinds_present (per-project, 60-second TTL)
# ---------------------------------------------------------------------------
_kinds_cache: dict[str, tuple[float, list[str]]] = {}

# Compiled regex for safe_kind validation
_UNSAFE_KIND_RE = re.compile(r"[^A-Za-z0-9_-]")


# ---------------------------------------------------------------------------
# New Wave-0 helpers
# ---------------------------------------------------------------------------

def fetch_kinds_present(conn: sqlite3.Connection, project_hash: str) -> list[str]:
    """Distinct active kinds in this project's chunks table. Cached 60s per project.

    Returns a sorted list of kind strings for chunks where superseded_by is empty.
    """
    now = time.monotonic()
    cached = _kinds_cache.get(project_hash)
    if cached is not None:
        ts, kinds = cached
        if now - ts < 60.0:
            return kinds

    try:
        rows = conn.execute(
            "SELECT DISTINCT kind FROM chunks WHERE superseded_by = '' ORDER BY 1"
        ).fetchall()
        kinds = [row[0] for row in rows if row[0]]
    except sqlite3.OperationalError:
        kinds = []

    _kinds_cache[project_hash] = (now, kinds)
    return kinds


def fetch_lineage(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    """What each chunk on a page replaced, and what replaced it.

    Returns {id: {superseded_by, replaced_by, forgotten, replaces}}. A
    superseded_by naming a chunk that exists is a replacement (a contradiction
    resolved by a newer note); one naming no chunk is a forget tombstone.
    """
    if not ids:
        return {}
    ph = ", ".join("?" for _ in ids)
    out: dict[str, dict] = {
        cid: {"superseded_by": sup or "", "replaced_by": "", "forgotten": False, "replaces": []}
        for cid, sup in conn.execute(f"SELECT id, superseded_by FROM chunks WHERE id IN ({ph})", ids)
    }
    sups = sorted({v["superseded_by"] for v in out.values() if v["superseded_by"]})
    if sups:
        ph2 = ", ".join("?" for _ in sups)
        live = {r[0] for r in conn.execute(f"SELECT id FROM chunks WHERE id IN ({ph2})", sups)}
        for v in out.values():
            if v["superseded_by"] in live:
                v["replaced_by"] = v["superseded_by"]
            elif v["superseded_by"]:
                v["forgotten"] = True
    for sup, old in conn.execute(
        f"SELECT superseded_by, id FROM chunks WHERE superseded_by IN ({ph}) ORDER BY ts DESC", ids
    ):
        out[sup]["replaces"].append(old)
    return out


def fetch_chunk_v05(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    """Bulk-read v0.5 columns for chunks.

    Returns {id: {salience, labile_until, reconsolidation_count, co_retrieval_count,
    cross_links, cross_eligible, retrieval_count_local, consolidation_round}}.
    Empty list -> empty dict. Missing ids omitted from result.
    cross_links is returned as the raw JSON string (caller uses parse_cross_links).
    """
    if not ids:
        return {}

    placeholders = ", ".join("?" for _ in ids)
    try:
        rows = conn.execute(
            f"SELECT id, salience, labile_until, reconsolidation_count, co_retrieval_count, "
            f"cross_links, cross_eligible, retrieval_count_local, consolidation_round "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    result: dict[str, dict] = {}
    for row in rows:
        result[row[0]] = {
            "salience": row[1],
            "labile_until": row[2],
            "reconsolidation_count": row[3],
            "co_retrieval_count": row[4],
            "cross_links": row[5],
            "cross_eligible": row[6],
            "retrieval_count_local": row[7],
            "consolidation_round": row[8],
        }
    return result


def _count_cli_failures_24h() -> int:
    """Count entries in BRAIN_ROOT/cli_failures.jsonl whose ts is within 24h.

    distill.py / judge.py append one line per claude CLI failure. Best-effort:
    silently returns 0 if the file is missing, malformed, or unreadable.
    """
    log = BRAIN_ROOT / "cli_failures.jsonl"
    if not log.exists():
        return 0
    cutoff = int(time.time()) - 86400
    count = 0
    try:
        with log.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if int(row.get("ts", 0)) >= cutoff:
                        count += 1
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except OSError:
        return 0
    return count


_CLI_FAILURE_TEXT = {
    "claude_cli_missing": "Claude CLI not found",
    "claude_cli_timeout": "Claude CLI timed out",
    "claude_cli_error": "Claude CLI returned an error",
}


def _last_cli_failure() -> tuple[str, int]:
    """The most recent claude CLI failure as (plain text, ts), or ("", 0).

    Rows carry only a fixed kind and a timestamp; the reason itself is in
    flush.log, so nothing from a transcript reaches the page.
    """
    log = BRAIN_ROOT / "cli_failures.jsonl"
    try:
        lines = log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", 0
    for line in reversed(lines):
        try:
            row = json.loads(line)
            kind = str(row.get("kind", ""))
            return _CLI_FAILURE_TEXT.get(kind, "Claude CLI call failed"), int(row.get("ts", 0))
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            continue
    return "", 0


def fetch_brain_status(project_hash: str) -> dict:
    """Read dream_state row + edge_bumps queue depth + DMN/dream daemon plist presence.

    Returns:
    {
      'dream_stage': str,           # 'IDLE' if no row
      'dream_complete_ts': int,     # 0 if no row
      'dream_error': str,           # '' if no row
      'dream_error_ts': int,
      'edge_queue_depth': int,      # COUNT(*) FROM edge_bumps WHERE drained=0
      'dmn_plist_present': bool,    # ~/Library/LaunchAgents/ch.bissbert.biss-brain.dmn.plist
      'dream_plist_present': bool,
      'cli_failures_24h': int,      # tail of BRAIN_ROOT/cli_failures.jsonl in last 24h
      'cli_failure_last': str,      # plain text of the newest failure, '' if none
      'cli_failure_last_ts': int,
      'log_path': str,              # where the daemon logs the reasons
    }
    Degrades gracefully (returns zeros/empties) if tables don't exist.
    Opens and closes its own SQLite connection.
    """
    result: dict = {
        "dream_stage": "IDLE",
        "dream_complete_ts": 0,
        "dream_error": "",
        "dream_error_ts": 0,
        "edge_queue_depth": 0,
        "dmn_plist_present": False,
        "dream_plist_present": False,
        "cli_failures_24h": _count_cli_failures_24h(),
        "log_path": str(BRAIN_ROOT / "flush.log").replace(str(Path.home()), "~", 1),
    }
    result["cli_failure_last"], result["cli_failure_last_ts"] = _last_cli_failure()

    # Check plist presence via filesystem
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    dmn_plist = launch_agents / "ch.bissbert.biss-brain.dmn.plist"
    dream_plist = launch_agents / "ch.bissbert.biss-brain.dream.plist"
    result["dmn_plist_present"] = dmn_plist.exists()
    result["dream_plist_present"] = dream_plist.exists()

    # Open SQLite connection
    sqlite_path = _project_dir(project_hash) / "meta.sqlite"
    if not sqlite_path.exists():
        return result

    try:
        conn = sqlite3.connect(
            f"file:{sqlite_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            # dream_state row
            try:
                row = conn.execute(
                    "SELECT last_stage, last_complete_ts, last_error, last_error_ts "
                    "FROM dream_state WHERE project = ?",
                    (project_hash,),
                ).fetchone()
                if row:
                    result["dream_stage"] = row["last_stage"] or "IDLE"
                    result["dream_complete_ts"] = row["last_complete_ts"] or 0
                    result["dream_error"] = scrub(row["last_error"] or "")  # exception text can quote a secret
                    result["dream_error_ts"] = row["last_error_ts"] or 0
            except sqlite3.OperationalError:
                pass

            # edge_bumps queue depth
            try:
                edge_row = conn.execute(
                    "SELECT COUNT(*) FROM edge_bumps WHERE drained = 0"
                ).fetchone()
                if edge_row:
                    result["edge_queue_depth"] = edge_row[0] or 0
            except sqlite3.OperationalError:
                pass
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        pass

    return result


def fetch_neighbours(project_hash: str, chunk_id: str, k: int = 5) -> list[dict]:
    """Return up to k semantic neighbours of chunk_id.

    Fast path: reads backlinks.fetch_related for pre-computed related ids,
    then fetches their text and metadata from LanceDB.
    Fallback: fetches the seed chunk's vector from LanceDB and performs an
    ANN search, excluding the seed itself.
    Each result dict: {id, kind, text, ts, score (or None for backlink path)}.
    Returns [] if chunk doesn't exist.
    """
    from backlinks import fetch_related  # noqa: PLC0415

    safe_seed = safe_id(chunk_id)
    if safe_seed is None:
        return []

    brain_dir = _project_dir(project_hash)

    # Fast path: pre-computed backlinks
    related_ids = fetch_related(str(brain_dir), safe_seed)
    if related_ids:
        safe_ids = [i for i in (safe_id(rid) for rid in related_ids[:k]) if i]
        if not safe_ids:
            return []
        table = open_lance(project_hash)
        if table is None:
            return []
        texts_and_meta = fetch_by_ids(table, safe_ids)
        return [
            {
                "id": r["id"],
                "kind": r.get("kind", ""),
                "text": r.get("text", ""),
                "ts": r.get("ts"),
                "score": None,
            }
            for r in texts_and_meta
        ]

    # Fallback: on-demand ANN search using seed vector
    table = open_lance(project_hash)
    if table is None:
        return []

    seed_rows = fetch_by_ids(table, [safe_seed], limit=1)
    if not seed_rows:
        return []

    seed = seed_rows[0]
    qvec = seed.get("vector")
    if qvec is None:
        return []

    rows = vector_search(table, list(qvec), F.and_(F.LIVE, F.ne("id", safe_seed)), k)
    return [
        {
            "id": r["id"],
            "kind": r.get("kind", ""),
            "text": r.get("text", ""),
            "ts": r.get("ts"),
            "score": r["_cosine"],
        }
        for r in rows
    ]


def parse_cross_links(json_str: str | None) -> list[dict]:
    """Safe JSON parse of cross_links column. Returns [] on None, empty, or parse error."""
    if not json_str:
        return []
    try:
        result = json.loads(json_str)
        if not isinstance(result, list):
            return []
        return result
    except (json.JSONDecodeError, ValueError):
        return []


def is_global_project(project_hash: str) -> bool:
    """Return True if project_hash is the cross-project global pool."""
    return project_hash == GLOBAL_PROJECT_HASH


def safe_kind(kind: str | None) -> str | None:
    """Return kind if it's safe to interpolate into SQL, else None.

    Rejects: None, empty string, length > 32, SQL comment markers, or any
    character outside [A-Za-z0-9_-]. Single hyphens are allowed for kebab-
    case kinds; consecutive hyphens (--) are rejected as SQL comment markers.
    """
    if not kind:
        return None
    if len(kind) > 32:
        return None
    if "--" in kind:
        return None
    if _UNSAFE_KIND_RE.search(kind):
        return None
    return kind


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def safe_id(chunk_id: str | None) -> str | None:
    """Return chunk_id if safe to interpolate into a LanceDB .where() string.

    chunk_id values are uuid hex (32 chars) or short ulid-style tokens; the
    accepted alphabet is [A-Za-z0-9_-] up to 64 characters. Any other input
    (single quotes, spaces, SQL comments) returns None so callers can short-
    circuit before issuing the predicate. LanceDB .where() is parsed as a
    string predicate and does not support parameter binding, so this guard
    is the only line of defence at that boundary.
    """
    if not chunk_id:
        return None
    if not _SAFE_ID_RE.match(chunk_id):
        return None
    return chunk_id


# tombstone_id is imported from common and re-exported above
