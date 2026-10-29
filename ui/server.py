"""FastAPI application skeleton for the biss-brain UI inspector.

Phase 0: foundation routes only. Panel logic (search, review, timeline,
sessions, status) lives in their own router modules under ui/routers/.
Phase 1/2 agents fill those in without touching this file.

Routing contract (stable for Phase 1/2):
  GET /                        -> index.html (overview; with ?project=, the next-session preview)
  GET /search?project=<h>&q=   -> search panel
  GET /review?project=<h>      -> review queue (proposals, fading, replaced)
  GET /status?project=<h>      -> brain status
  GET /sessions?project=<h>    -> sessions panel
  GET /session/<sid>?project=<h> -> session detail panel
  GET /timeline?project=<h>    -> timeline panel
  GET /chunk/<id>?project=<h>  -> chunk row as JSON
  GET /healthz                 -> {"ok": True}
  GET /static/*                -> StaticFiles
  POST /api/label              -> write project_path.txt (form body: hash, path)
"""
from __future__ import annotations

import sqlite3
import sys
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Path setup – allow `from scripts.common import …` when running from the
# project root, and also allow plain `from common import …` (scripts/ style).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # ui/
_ROOT = _HERE.parent                              # plugin root
_SCRIPTS = _ROOT / "scripts"
for _p in (_ROOT, _SCRIPTS):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from common import BRAIN_ROOT, discover_projects, lancedb_dir, project_dir  # noqa: E402
from ui.data import validate_project, open_meta_ro  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="biss-brain inspector", docs_url=None, redoc_url=None)

# Static files
_STATIC_DIR = _HERE / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Templates
import datetime as _dt

_TEMPLATES_DIR = _HERE / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _datetimeformat(value: int | None) -> str:
    if not value:
        return "unknown"
    try:
        return _dt.datetime.fromtimestamp(int(value), tz=_dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except Exception:
        return str(value)


templates.env.filters["datetimeformat"] = _datetimeformat

# Expose plugin version as a Jinja2 global (read once at startup)
_PLUGIN_JSON = _ROOT / ".claude-plugin" / "plugin.json"
try:
    import json as _json
    _plugin_version = _json.loads(_PLUGIN_JSON.read_text()).get("version", "")
except Exception:
    _plugin_version = ""
templates.env.globals["plugin_version"] = _plugin_version

# ---------------------------------------------------------------------------
# Panel routers – each agent owns its own file; no conflicts during parallel work
# ---------------------------------------------------------------------------
from ui.routers import review, search, sessions, timeline, status  # noqa: E402

app.include_router(search.router)
app.include_router(sessions.router)
app.include_router(timeline.router)
app.include_router(status.router)
app.include_router(review.router)

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _generic_error(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    # Sanitise – no traceback to client
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# Shared infrastructure routes (NOT panel work – stay in server.py)
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Any:
    return JSONResponse(status_code=204, content=None)


@app.get("/")
async def index(request: Request, project: str = Query("", description="Project hash")) -> Any:
    projects = discover_projects()
    preview = None
    if project and any(p["hash"] == project for p in projects):
        from ui.preview import next_session

        preview = next_session(project)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"projects": projects, "current_project": project or None, "preview": preview},
    )


@app.get("/chunk/{chunk_id}")
async def chunk_detail(
    chunk_id: str,
    request: Request,
    project: str = Query(..., description="Project hash"),
) -> dict:
    pdir = validate_project(project)
    conn = open_meta_ro(project)
    try:
        row = conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")
        return dict(row)
    finally:
        conn.close()


@app.post("/api/label")
async def api_label(
    hash: str = Form(..., description="Project hash to label"),
    path: str = Form(..., description="Filesystem path to associate"),
) -> dict:
    """Write project_path.txt for a project that lacks one.

    Reads hash and path from a form-encoded POST body so that the default
    HTMX hx-post behaviour (form encoding) works without any JS.
    Validates that the provided path actually exists before writing.
    """
    pdir = validate_project(hash)
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist: {path}",
        )
    label_file = pdir / "project_path.txt"
    label_file.write_text(str(resolved))
    return {"ok": True, "hash": hash, "path": str(resolved)}
