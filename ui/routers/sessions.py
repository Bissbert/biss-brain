"""Sessions panel router – V5 implementation.

Endpoints
---------
GET  /sessions                     – session index (HTML)
GET  /session/{sid}                – single-session replay (HTML)
GET  /api/sessions/recent          – recent sessions (JSON, for dashboard tiles)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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

from ui.data import (  # noqa: E402
    fetch_chunk_texts,
    open_lance,
    open_meta_ro,
    validate_project,
)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Jinja filters
# ---------------------------------------------------------------------------

def _relative_time(ts: int | float | None) -> str:
    """Return a human-readable relative timestamp.

    Returns 'unknown' for None/0, a relative string (e.g. '2 days ago') for
    timestamps within 7 days, or an ISO date string otherwise.
    """
    if not ts:
        return "unknown"
    try:
        ts = int(ts)
    except (ValueError, TypeError):
        return str(ts)
    now = int(time.time())
    diff = now - ts
    if diff < 0:
        diff = 0
    if diff < 60:
        return "just now"
    if diff < 3600:
        mins = diff // 60
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if diff < 86400:
        hours = diff // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = diff // 86400
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _fmt_ts(ts: int | None) -> str:
    if not ts:
        return "unknown"
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


templates.env.filters["relative_time"] = _relative_time
templates.env.filters["fmt_ts"] = _fmt_ts

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

_SESSION_AGG_SQL = """
SELECT session_id,
       COUNT(*) AS chunk_count,
       MIN(ts)  AS started_ts,
       MAX(ts)  AS ended_ts,
       GROUP_CONCAT(DISTINCT kind) AS kinds,
       SUM(CASE WHEN superseded_by != '' THEN 1 ELSE 0 END) AS superseded_count,
       SUM(pinned) AS pinned_count
FROM chunks
WHERE session_id != ''
GROUP BY session_id
ORDER BY ended_ts DESC
LIMIT ?
"""

_SESSION_CHUNKS_SQL = """
SELECT id, kind, ts, agent_id, superseded_by, pinned, retrieval_count, decay_score, related_ids
FROM chunks
WHERE session_id = ?
ORDER BY ts ASC
"""

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/sessions")
async def sessions_index(
    request: Request,
    project: str = Query("", description="Project hash"),
    limit: int = Query(50, ge=1, le=500),
) -> HTMLResponse:
    from common import discover_projects  # noqa: PLC0415

    if not project:
        return RedirectResponse(url="/", status_code=303)
    validate_project(project)
    projects = discover_projects()

    rows: list[dict] = []
    conn = None
    try:
        conn = open_meta_ro(project)
        raw = conn.execute(_SESSION_AGG_SQL, (limit,)).fetchall()
        for r in raw:
            rows.append({
                "session_id":       r[0],
                "session_id_short": r[0][:8] if r[0] else "",
                "chunk_count":      r[1] or 0,
                "started_ts":       r[2],
                "ended_ts":         r[3],
                "kinds":            sorted((r[4] or "").split(",")) if r[4] else [],
                "superseded_count": r[5] or 0,
                "pinned_count":     r[6] or 0,
            })
    except Exception:
        # meta.sqlite might not exist yet for a brand-new project dir
        pass
    finally:
        if conn is not None:
            conn.close()

    ctx: dict[str, Any] = {
        "current_project": project,
        "projects":        projects,
        "sessions":        rows,
    }
    return templates.TemplateResponse(request, "sessions.html", ctx)


@router.get("/session/{sid}")
async def session_detail(
    sid: str,
    request: Request,
    project: str = Query(..., description="Project hash"),
) -> HTMLResponse:
    from common import discover_projects  # noqa: PLC0415

    validate_project(project)
    projects = discover_projects()

    conn = open_meta_ro(project)
    try:
        raw = conn.execute(_SESSION_CHUNKS_SQL, (sid,)).fetchall()
    finally:
        conn.close()

    if not raw:
        # Return an empty timeline page rather than 404 — session may be valid
        # but have been pruned.
        ctx: dict[str, Any] = {
            "current_project": project,
            "projects":        projects,
            "sid":             sid,
            "chunks":          [],
            "agent_ids":       [],
            "started_ts":      None,
            "ended_ts":        None,
            "chunk_count":     0,
        }
        return templates.TemplateResponse(request, "session.html", ctx)

    # Bulk-fetch text from LanceDB
    ids = [r[0] for r in raw]
    table = open_lance(project)
    texts = fetch_chunk_texts(table, ids)

    # Build an id-set for supersession resolution within this session
    session_id_set = set(ids)

    # Assign a lane colour/label per agent_id
    _lane_colours = [
        "#7c6af0",  # accent purple  (main / None)
        "#4caf7d",  # success green
        "#e09c5c",  # warm orange
        "#5cb8e0",  # sky blue
        "#e05c9c",  # pink
        "#c8e05c",  # lime
    ]
    agent_order: list[str | None] = []
    agent_colour: dict[str | None, str] = {}
    agent_label: dict[str | None, str] = {}

    chunks: list[dict] = []
    for r in raw:
        cid, kind, ts, agent_id, superseded_by, pinned, retrieval_count, decay_score, related_ids = r
        agent_key: str | None = agent_id if agent_id else None

        if agent_key not in agent_colour:
            idx = len(agent_colour)
            agent_colour[agent_key] = _lane_colours[idx % len(_lane_colours)]
            agent_label[agent_key] = "main" if agent_key is None else f"A{idx}"
            agent_order.append(agent_key)

        supersedes_in_session = (
            superseded_by and superseded_by in session_id_set
        ) if superseded_by else False

        chunks.append({
            "id":                      cid,
            "kind":                    kind or "",
            "ts":                      ts,
            "agent_id":                agent_id or "",
            "agent_key":               agent_key,
            "agent_colour":            agent_colour[agent_key],
            "agent_label":             agent_label[agent_key],
            "superseded_by":           superseded_by or "",
            "is_superseded":           bool(superseded_by),
            "supersedes_in_session":   supersedes_in_session,
            "pinned":                  pinned or 0,
            "retrieval_count":         retrieval_count or 0,
            "decay_score":             decay_score if decay_score is not None else 1.0,
            "related_ids":             related_ids or "",
            "text":                    texts.get(cid, ""),
        })

    ctx = {
        "current_project": project,
        "projects":        projects,
        "sid":             sid,
        "chunks":          chunks,
        "agent_ids":       [ak or "" for ak in agent_order],
        "started_ts":      raw[0][2],
        "ended_ts":        raw[-1][2],
        "chunk_count":     len(chunks),
    }
    return templates.TemplateResponse(request, "session.html", ctx)


@router.get("/api/sessions/recent")
async def api_sessions_recent(
    project: str = Query(..., description="Project hash"),
    limit: int = Query(10, ge=1, le=100),
) -> JSONResponse:
    validate_project(project)

    rows: list[dict] = []
    conn = None
    try:
        conn = open_meta_ro(project)
        raw = conn.execute(_SESSION_AGG_SQL, (limit,)).fetchall()
        for r in raw:
            rows.append({
                "session_id":  r[0],
                "chunk_count": r[1] or 0,
                "ended_ts":    r[3],
            })
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()

    return JSONResponse(rows)
