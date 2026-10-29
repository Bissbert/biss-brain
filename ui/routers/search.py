"""Search panel router – V5 implementation (UI v0.6).

Endpoints
---------
GET  /search              – full page or HTMX fragment
POST /api/pin             – toggle pinned=1 for a chunk
POST /api/unpin           – toggle pinned=0 for a chunk
POST /api/forget          – soft-delete: set superseded_by to a tombstone uuid
GET  /api/neighbours      – semantic neighbours for a chunk (B1, F-011)
POST /api/compose         – compose selected chunk ids into markdown (B1, F-015)

Features added in v0.6:
- Dynamic kinds list from DB (fetch_kinds_present / safe_kind)
- Extended filter bar: date range, session id, pinned_only, salience, round
- Pagination: limit/offset params with next/prev links
- Total count badge: "Showing X of Y matching"
- Server-side HTML highlighting of query terms in text preview
- Text-less filtering: when q is empty and filters are set, use fetch_chunks_meta
"""
from __future__ import annotations

import html
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Path / template setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # ui/
_TEMPLATES_DIR = _HERE / "templates"
_ROOT = _HERE.parent  # plugin root
_SCRIPTS = _ROOT / "scripts"
for _p in (_ROOT, _SCRIPTS):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import lance_filter as F  # noqa: E402
import provenance  # noqa: E402
from common import tombstone_id  # noqa: E402
from ui.data import (  # noqa: E402
    fetch_chunk_texts,
    fetch_chunk_v05,
    fetch_chunks_meta,
    fetch_kinds_present,
    fetch_neighbours,
    hybrid_search,
    open_lance,
    open_meta_ro,
    project_paths,
    safe_kind,
    validate_project,
)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
log = logging.getLogger(__name__)


# Expose datetime helper inside Jinja
def _fmt_ts(ts: int | None) -> str:
    if not ts:
        return "unknown"
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


templates.env.filters["fmt_ts"] = _fmt_ts

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date_to_ts(date_str: str, end_of_day: bool = False) -> int | None:
    """Parse a YYYY-MM-DD string to a unix timestamp.

    If end_of_day is True, returns 23:59:59 of that day.
    Returns None if parsing fails; logs a warning.
    """
    if not date_str or not date_str.strip():
        return None
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.timestamp())
    except ValueError:
        log.warning("search: invalid date param %r — ignoring", date_str)
        return None


def _build_where(
    include_superseded: bool,
    kind: str,
    from_ts: int | None = None,
    to_ts: int | None = None,
    session: str | None = None,
    pinned_only: bool = False,
    salience_min: int = 0,
    round_min: int = 0,
) -> str | None:
    """Build a SQL WHERE clause string from the validated filter parameters.

    All string values go through safe_kind() or a strict regex guard to
    prevent SQL injection.  Returns None when no filters are active.
    """
    parts: list[str] = []

    if not include_superseded:
        parts.append("superseded_by = ''")

    safe = safe_kind(kind)
    if safe:
        parts.append(f"kind = '{safe}'")
    elif kind:
        log.warning("search: kind param %r failed safe_kind validation — ignored", kind)

    if from_ts is not None:
        parts.append(f"ts >= {int(from_ts)}")
    if to_ts is not None:
        parts.append(f"ts <= {int(to_ts)}")

    if session:
        if re.match(r'^[A-Za-z0-9_-]{1,64}$', session):
            parts.append(f"session_id = '{session}'")
        else:
            log.warning("search: session param %r failed validation — ignored", session)

    if pinned_only:
        parts.append("pinned = 1")

    if salience_min > 0:
        parts.append(f"salience >= {int(salience_min)}")

    if round_min > 0:
        parts.append(f"consolidation_round >= {int(round_min)}")

    return " AND ".join(parts) if parts else None


def _build_lance_where(
    include_superseded: bool,
    kind: str,
    from_ts: int | None = None,
    to_ts: int | None = None,
    session: str | None = None,
) -> str | None:
    """The LanceDB side of the filters: only columns the chunks table has.

    pinned, salience and consolidation_round live in SQLite only; text search
    applies them afterwards through _meta_filter.
    """
    parts: list[str] = []
    if not include_superseded:
        parts.append(F.LIVE)
    for col, value, build in (
        ("kind", kind, F.eq),
        ("session_id", session, F.eq),
        ("ts", from_ts, F.ge),
    ):
        if value is None or value == "":
            continue
        try:
            parts.append(build(col, value))
        except F.FilterError:
            log.warning("search: %s param %r failed validation — ignored", col, value)
    if to_ts is not None:
        parts.append(f"ts <= {int(to_ts)}")
    return F.and_(*parts) or None


def _meta_filter(m: dict, pinned_only: bool, salience_min: int, round_min: int) -> bool:
    """True when a row's SQLite metadata passes the SQLite-only filters."""
    if pinned_only and not m.get("pinned"):
        return False
    if salience_min > 0 and (m.get("salience") or 0) < salience_min:
        return False
    return not (round_min > 0 and (m.get("consolidation_round") or 0) < round_min)


def _highlight(text: str, query: str, max_len: int = 400) -> str:
    """Escape HTML and wrap query terms in <mark> tags.

    Terms shorter than 2 characters are skipped.  The result is safe to
    render with |safe in Jinja (no raw user input survives unescaped).
    """
    if not query.strip():
        return html.escape(text[:max_len])

    terms = [re.escape(t) for t in query.split() if len(t) >= 2]
    if not terms:
        return html.escape(text[:max_len])

    snippet = text[:max_len]
    escaped = html.escape(snippet)
    pattern = re.compile(r"(" + "|".join(terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped)


def _meta_by_ids(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    """Bulk-fetch SQLite metadata for a list of chunk ids. Returns {id: row_dict}."""
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    extra = [c for c in ("salience", "consolidation_round", "source", "trust", "taint") if c in cols]
    rows = conn.execute(
        "SELECT id, pinned, decay_score, retrieval_count, superseded_by"
        + "".join(f", {c}" for c in extra)
        + f" FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    names = ["pinned", "decay_score", "retrieval_count", "superseded_by", *extra]
    return {row[0]: dict(zip(names, row[1:])) for row in rows}


def _render_card(request: Request, project: str, chunk: dict) -> str:
    return templates.TemplateResponse(
        request,
        "_chunk_card.html",
        {"project": project, "chunk": chunk},
    ).body.decode()


def _open_meta_rw(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(sqlite_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _qs(params: dict, **overrides: object) -> str:
    """Build a URL-encoded query string, omitting falsy values.

    Merges params with overrides; any value equal to None, "", 0, or False
    is omitted so links stay clean.
    """
    merged = {**params, **overrides}
    filtered = {k: v for k, v in merged.items() if v not in (None, "", 0, False)}
    return urlencode(filtered)


def _fetch_texts(table: object, ids: list[str]) -> dict[str, str]:
    """Fetch text column from LanceDB for a list of ids."""
    if not ids or table is None:
        return {}
    return fetch_chunk_texts(table, ids)


def _enrich_with_v05(chunk: dict, v05_map: dict[str, dict]) -> dict:
    """Merge v0.5 columns into a chunk dict. Falls back to zeros if missing."""
    v05 = v05_map.get(chunk["id"], {})
    chunk["salience"] = v05.get("salience", 0)
    chunk["labile_until"] = v05.get("labile_until", 0)
    chunk["reconsolidation_count"] = v05.get("reconsolidation_count", 0)
    chunk["co_retrieval_count"] = v05.get("co_retrieval_count", 0)
    chunk["cross_links"] = v05.get("cross_links", "[]")
    chunk["cross_eligible"] = v05.get("cross_eligible", False)
    chunk["retrieval_count_local"] = v05.get("retrieval_count_local", 0)
    chunk["consolidation_round"] = v05.get("consolidation_round", 0)
    chunk["_now_ts"] = int(time.time())
    return chunk


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------

@router.get("/search")
async def search(
    request: Request,
    project: str = Query("", description="Project hash"),
    q: str = Query("", description="Search query"),
    kind: str = Query("", description="Filter by chunk kind"),
    include_superseded: int = Query(0, description="Include superseded chunks (0|1)"),
    from_date: str = Query("", description="Start date YYYY-MM-DD"),
    to_date: str = Query("", description="End date YYYY-MM-DD"),
    session: str = Query("", description="Filter by session id"),
    pinned_only: int = Query(0, description="Show pinned chunks only (0|1)"),
    salience_min: int = Query(0, ge=0, le=100, description="Minimum salience score"),
    round_min: int = Query(0, ge=0, description="Minimum consolidation round"),
    limit: int = Query(50, ge=1, le=200, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> HTMLResponse:
    from common import discover_projects  # noqa: PLC0415

    if not project:
        return RedirectResponse(url="/", status_code=303)
    validate_project(project)
    projects = discover_projects()

    from_ts = _parse_date_to_ts(from_date, end_of_day=False)
    to_ts = _parse_date_to_ts(to_date, end_of_day=True)

    conn = open_meta_ro(project)
    try:
        kinds_present = fetch_kinds_present(conn, project)
    finally:
        conn.close()

    where_sql = _build_where(
        include_superseded=bool(include_superseded),
        kind=kind,
        from_ts=from_ts,
        to_ts=to_ts,
        session=session or None,
        pinned_only=bool(pinned_only),
        salience_min=salience_min,
        round_min=round_min,
    )

    ctx: dict = {
        "request": request,
        "current_project": project,
        "projects": projects,
        "q": q,
        "kind": kind,
        "include_superseded": include_superseded,
        "from_date": from_date,
        "to_date": to_date,
        "session": session,
        "pinned_only": pinned_only,
        "salience_min": salience_min,
        "round_min": round_min,
        "limit": limit,
        "offset": offset,
        "results": [],
        "result_count": 0,
        "total_count": 0,
        "has_next": False,
        "kinds_present": kinds_present,
        "qs_prev": "",
        "qs_next": "",
    }

    # Determine total_count from SQLite (independent of vector search)
    conn2 = open_meta_ro(project)
    try:
        count_sql = where_sql or "1=1"
        try:
            count_row = conn2.execute(
                f"SELECT COUNT(*) FROM chunks WHERE {count_sql}"
            ).fetchone()
            ctx["total_count"] = count_row[0] if count_row else 0
        except sqlite3.OperationalError as e:
            log.warning("search: count query failed: %s", e)
            ctx["total_count"] = 0
    finally:
        conn2.close()

    has_any_filter = bool(where_sql)
    do_text_search = bool(q.strip())

    if do_text_search:
        table = open_lance(project)
        fetch_limit = offset + limit + 1
        meta_filters = bool(pinned_only or salience_min > 0 or round_min > 0)
        lance_where = _build_lance_where(
            include_superseded=bool(include_superseded),
            kind=kind,
            from_ts=from_ts,
            to_ts=to_ts,
            session=session or None,
        )
        # SQLite-only filters thin the pool afterwards, so fetch a wider one
        pool = max(fetch_limit * 5, 200) if meta_filters else fetch_limit
        lance_rows = hybrid_search(table, q, limit=pool, where_sql=lance_where)

        meta_map: dict[str, dict] = {}
        if lance_rows:
            conn3 = open_meta_ro(project)
            try:
                meta_map = _meta_by_ids(conn3, [r["id"] for r in lance_rows])
            finally:
                conn3.close()
        if meta_filters:
            lance_rows = [
                r for r in lance_rows
                if _meta_filter(meta_map.get(r["id"], {}), bool(pinned_only), salience_min, round_min)
            ]

        paged_rows = lance_rows[offset : offset + limit + 1]
        has_next = len(paged_rows) > limit
        visible_rows = paged_rows[:limit]

        if visible_rows:
            conn3 = open_meta_ro(project)
            try:
                v05_map = fetch_chunk_v05(conn3, [r["id"] for r in visible_rows])
            finally:
                conn3.close()

            enriched: list[dict] = []
            for r in visible_rows:
                cid = r["id"]
                m = meta_map.get(cid, {})
                text = r.get("text", "")
                chunk = {
                    "id": cid,
                    "kind": r.get("kind", ""),
                    "ts": r.get("ts"),
                    "session_id": r.get("session_id", ""),
                    "text": text,
                    "text_html": _highlight(text, q),
                    "pinned": m.get("pinned", 0),
                    "decay_score": m.get("decay_score", 1.0),
                    "retrieval_count": m.get("retrieval_count", 0),
                    "score": r.get("_score"),
                    "is_superseded": bool(m.get("superseded_by", "")),
                    "source": provenance.label(
                        provenance.from_row(m.get("source"), m.get("trust"), m.get("taint")) if m else None
                    ),
                }
                _enrich_with_v05(chunk, v05_map)
                enriched.append(chunk)
            ctx["results"] = enriched
            ctx["result_count"] = len(enriched)
            ctx["has_next"] = has_next

    elif has_any_filter:
        # No text query but filters are active — use SQLite fetch_chunks_meta
        conn4 = open_meta_ro(project)
        try:
            count_sql = where_sql or "1=1"
            meta_rows = fetch_chunks_meta(
                conn4,
                where_sql=count_sql,
                limit=limit + 1,
            )
            has_next = len(meta_rows) > limit
            visible_meta = meta_rows[:limit]
            ids = [r["id"] for r in visible_meta]
            v05_map = fetch_chunk_v05(conn4, ids) if ids else {}
        finally:
            conn4.close()

        table = open_lance(project)
        texts = _fetch_texts(table, ids) if ids else {}

        enriched2: list[dict] = []
        for r in visible_meta:
            cid = r["id"]
            text = texts.get(cid, "")
            chunk = {
                "id": cid,
                "kind": r.get("kind", ""),
                "ts": r.get("ts"),
                "session_id": r.get("session_id", ""),
                "text": text,
                "text_html": html.escape(text[:400]),
                "pinned": r.get("pinned", 0),
                "decay_score": r.get("decay_score", 1.0),
                "retrieval_count": r.get("retrieval_count", 0),
                "score": None,
                "is_superseded": bool(r.get("superseded_by", "")),
            }
            _enrich_with_v05(chunk, v05_map)
            enriched2.append(chunk)
        ctx["results"] = enriched2
        ctx["result_count"] = len(enriched2)
        ctx["has_next"] = has_next

    base_params: dict = {
        "project": project,
        "q": q,
        "kind": kind,
        "from_date": from_date,
        "to_date": to_date,
        "session": session,
        "pinned_only": pinned_only,
        "salience_min": salience_min,
        "round_min": round_min,
        "include_superseded": include_superseded,
        "limit": limit,
    }
    ctx["qs_prev"] = _qs(base_params, offset=max(0, offset - limit))
    ctx["qs_next"] = _qs(base_params, offset=offset + limit)

    is_htmx = request.headers.get("hx-request")
    ctx_no_req = {k: v for k, v in ctx.items() if k != "request"}
    if is_htmx:
        return templates.TemplateResponse(request, "_search_results.html", ctx_no_req)
    return templates.TemplateResponse(request, "search.html", ctx_no_req)


# ---------------------------------------------------------------------------
# POST /api/pin   &   POST /api/unpin
# ---------------------------------------------------------------------------

@router.post("/api/pin")
async def api_pin(
    request: Request,
    project: str = Query(...),
    id: str = Query(...),
) -> HTMLResponse:
    return await _set_pin(request, project, id, pinned=1)


@router.post("/api/unpin")
async def api_unpin(
    request: Request,
    project: str = Query(...),
    id: str = Query(...),
) -> HTMLResponse:
    return await _set_pin(request, project, id, pinned=0)


async def _set_pin(request: Request, project: str, chunk_id: str, pinned: int) -> HTMLResponse:
    validate_project(project)
    sqlite_path, _ = project_paths(project)
    conn = _open_meta_rw(sqlite_path)
    try:
        conn.execute(
            "UPDATE chunks SET pinned = ? WHERE id = ? AND project = ?",
            (pinned, chunk_id, project),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, kind, ts, session_id, superseded_by, pinned, decay_score, retrieval_count "
            "FROM chunks WHERE id = ? AND project = ?",
            (chunk_id, project),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")

    table = open_lance(project)
    texts = fetch_chunk_texts(table, [chunk_id])

    conn2 = open_meta_ro(project)
    try:
        v05_map = fetch_chunk_v05(conn2, [chunk_id])
    finally:
        conn2.close()

    text = texts.get(row[0], "")
    chunk = {
        "id": row[0],
        "kind": row[1],
        "ts": row[2],
        "session_id": row[3],
        "text": text,
        "text_html": html.escape(text[:400]),
        "pinned": row[5],
        "decay_score": row[6],
        "retrieval_count": row[7],
        "score": None,
        "is_superseded": bool(row[4]),
    }
    _enrich_with_v05(chunk, v05_map)
    return templates.TemplateResponse(
        request, "_chunk_card.html", {"project": project, "chunk": chunk}
    )


# ---------------------------------------------------------------------------
# POST /api/forget
# ---------------------------------------------------------------------------

@router.post("/api/forget")
async def api_forget(
    request: Request,
    project: str = Query(...),
    id: str = Query(...),
) -> HTMLResponse:
    validate_project(project)
    sqlite_path, _ = project_paths(project)
    tomb = tombstone_id()

    conn = _open_meta_rw(sqlite_path)
    try:
        conn.execute(
            "UPDATE chunks SET superseded_by = ? WHERE id = ? AND project = ?",
            (tomb, id, project),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, kind, ts, session_id, superseded_by, pinned, decay_score, retrieval_count "
            "FROM chunks WHERE id = ? AND project = ?",
            (id, project),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {id}")

    # Mirror into LanceDB
    try:
        import lancedb  # noqa: PLC0415
        _, lance_path = project_paths(project)
        if lance_path.exists():
            db = lancedb.connect(str(lance_path))
            tables = db.list_tables()
            tnames = tables.tables if hasattr(tables, "tables") else list(tables)
            if "chunks" in tnames:
                tbl = db.open_table("chunks")
                tbl.update(where=F.eq("id", id), values={"superseded_by": tomb})
    except Exception:
        pass  # best-effort mirror; SQLite is source of truth

    table = open_lance(project)
    texts = fetch_chunk_texts(table, [id])

    conn3 = open_meta_ro(project)
    try:
        v05_map = fetch_chunk_v05(conn3, [id])
    finally:
        conn3.close()

    text = texts.get(row[0], "")
    chunk = {
        "id": row[0],
        "kind": row[1],
        "ts": row[2],
        "session_id": row[3],
        "text": text,
        "text_html": html.escape(text[:400]),
        "pinned": row[5],
        "decay_score": row[6],
        "retrieval_count": row[7],
        "score": None,
        "is_superseded": bool(row[4]),
    }
    _enrich_with_v05(chunk, v05_map)
    return templates.TemplateResponse(
        request, "_chunk_card.html", {"project": project, "chunk": chunk}
    )


# ---------------------------------------------------------------------------
# GET /api/neighbours  (B1, F-011)
# ---------------------------------------------------------------------------

@router.get("/api/neighbours")
async def api_neighbours(
    request: Request,
    project: str = Query(...),
    id: str = Query(...),
    k: int = Query(5, ge=1, le=20),
) -> HTMLResponse:
    validate_project(project)
    nbrs = fetch_neighbours(project, id, k=k)
    return templates.TemplateResponse(
        request,
        "_neighbours.html",
        {"neighbours": nbrs, "project": project, "seed_id": id},
    )


# ---------------------------------------------------------------------------
# POST /api/compose  (B1, F-015)
# ---------------------------------------------------------------------------

@router.post("/api/compose")
async def api_compose(
    request: Request,
    project: str = Query(...),
) -> HTMLResponse:
    validate_project(project)
    body = await request.json()
    ids = body.get("ids", [])
    if not ids or len(ids) > 50:
        raise HTTPException(400, "ids must be 1..50 items")

    table = open_lance(project)
    texts = fetch_chunk_texts(table, ids)

    conn = open_meta_ro(project)
    try:
        placeholders = ", ".join("?" for _ in ids)
        meta_rows = conn.execute(
            f"SELECT id, kind, ts, session_id FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        conn.close()

    meta_by_id = {r[0]: r for r in meta_rows}
    chunks = []
    for cid in ids:
        if cid not in meta_by_id:
            continue
        r = meta_by_id[cid]
        chunks.append({
            "id": r[0],
            "kind": r[1],
            "ts": r[2],
            "session_id": r[3],
            "text": texts.get(cid, ""),
        })

    # Reuse scripts/inject_context._format_chunks
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location(
        "inject_context", _SCRIPTS / "inject_context.py"
    )
    ic = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ic)  # type: ignore[union-attr]
    md = ic._format_chunks(chunks, header=f"Composed context ({len(chunks)} chunks)")

    return templates.TemplateResponse(
        request,
        "_compose_drawer.html",
        {"markdown": md, "count": len(chunks), "project": project},
    )
