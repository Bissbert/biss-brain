"""Timeline panel router — Decay Timeline V2.

Endpoints:
  GET /timeline?project=<hash>&days=<int>&kind=<kind>
      Returns full HTML page (extends base.html). Data is embedded inline.

  GET /api/timeline.json?project=<hash>&days=<int>&kind=<kind>
      Returns JSON: {bins, summary, kinds_present}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ui.data import open_meta_ro, validate_project

# ---------------------------------------------------------------------------
# Path / template setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent.parent   # ui/
_TEMPLATES_DIR = _HERE / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _datetimeformat(value: int | None) -> str:
    import datetime as _dt
    if not value:
        return "unknown"
    try:
        return _dt.datetime.fromtimestamp(int(value), tz=_dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except Exception:
        return str(value)


templates.env.filters["datetimeformat"] = _datetimeformat

router = APIRouter()

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

_BINS_SQL = """
SELECT date(ts, 'unixepoch', 'localtime') AS day,
       kind,
       COUNT(*) AS count,
       AVG(COALESCE(decay_score, 1.0)) AS mean_decay,
       SUM(pinned) AS pinned,
       SUM(CASE WHEN superseded_by != '' THEN 1 ELSE 0 END) AS superseded
FROM chunks
WHERE ts >= ?
  AND (? = 'all' OR kind = ?)
GROUP BY day, kind
ORDER BY day ASC
"""

_SUMMARY_SQL = """
SELECT
  COUNT(*) AS total_chunks,
  SUM(CASE WHEN superseded_by = '' THEN 1 ELSE 0 END) AS active_chunks,
  SUM(CASE WHEN superseded_by != '' THEN 1 ELSE 0 END) AS superseded,
  SUM(pinned) AS pinned,
  AVG(COALESCE(decay_score, 1.0)) AS mean_decay,
  MIN(ts) AS earliest_ts,
  MAX(ts) AS latest_ts
FROM chunks
WHERE ts >= ?
  AND (? = 'all' OR kind = ?)
"""

_KINDS_SQL = "SELECT DISTINCT kind FROM chunks ORDER BY kind"


def _query_data(project: str, days: int, kind: str) -> dict[str, Any]:
    """Run all SQLite queries and return the combined payload dict."""
    since_ts = int(time.time()) - days * 86400

    conn = open_meta_ro(project)
    try:
        # Bins
        rows = conn.execute(_BINS_SQL, (since_ts, kind, kind)).fetchall()
        bins = [
            {
                "day": row[0],
                "kind": row[1],
                "count": row[2],
                "mean_decay": round(float(row[3]), 4) if row[3] is not None else None,
                "pinned": int(row[4]) if row[4] is not None else 0,
                "superseded": int(row[5]) if row[5] is not None else 0,
            }
            for row in rows
        ]

        # Summary
        srow = conn.execute(_SUMMARY_SQL, (since_ts, kind, kind)).fetchone()
        if srow and srow[0]:
            summary = {
                "total_chunks": int(srow[0]),
                "active_chunks": int(srow[1]) if srow[1] is not None else 0,
                "superseded": int(srow[2]) if srow[2] is not None else 0,
                "pinned": int(srow[3]) if srow[3] is not None else 0,
                "mean_decay": round(float(srow[4]), 4) if srow[4] is not None else None,
                "earliest_ts": int(srow[5]) if srow[5] is not None else None,
                "latest_ts": int(srow[6]) if srow[6] is not None else None,
            }
        else:
            summary = {
                "total_chunks": 0,
                "active_chunks": 0,
                "superseded": 0,
                "pinned": 0,
                "mean_decay": None,
                "earliest_ts": None,
                "latest_ts": None,
            }

        # Kinds present (all, not filtered)
        kinds_present = [r[0] for r in conn.execute(_KINDS_SQL).fetchall()]
    finally:
        conn.close()

    return {"bins": bins, "summary": summary, "kinds_present": kinds_present}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/timeline.json")
async def timeline_json(
    project: str = Query(..., description="Project hash"),
    days: int = Query(90, ge=1, description="Look-back window in days"),
    kind: str = Query("all", description="Filter by chunk kind, or 'all'"),
) -> JSONResponse:
    validate_project(project)
    data = _query_data(project, days, kind)
    return JSONResponse(data)


@router.get("/timeline", response_class=HTMLResponse)
async def timeline(
    request: Request,
    project: str = Query("", description="Project hash"),
    days: int = Query(90, ge=1, description="Look-back window in days"),
    kind: str = Query("all", description="Filter by chunk kind, or 'all'"),
) -> HTMLResponse:
    if not project:
        return RedirectResponse(url="/", status_code=303)
    validate_project(project)
    data = _query_data(project, days, kind)

    # Discover all projects for the nav selector (base.html needs it)
    _HERE_SRV = Path(__file__).resolve().parent.parent.parent
    _scripts = _HERE_SRV / "scripts"
    for _p in (str(_HERE_SRV), str(_scripts)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from common import discover_projects  # noqa: PLC0415
    projects = discover_projects()

    summary = data["summary"]
    # Starlette 1.0+ API: TemplateResponse(request, name, context)
    return templates.TemplateResponse(
        request,
        "timeline.html",
        {
            "current_project": project,
            "projects": projects,
            "days": days,
            "kind": kind,
            "data_json": json.dumps(data),
            # Flat summary fields — no nested dicts in context
            "kinds_present": data["kinds_present"],
            "summary_total_chunks": summary["total_chunks"],
            "summary_active_chunks": summary["active_chunks"],
            "summary_superseded": summary["superseded"],
            "summary_pinned": summary["pinned"],
            "summary_mean_decay": summary["mean_decay"],
            "summary_earliest_ts": summary["earliest_ts"],
            "summary_latest_ts": summary["latest_ts"],
        },
    )
