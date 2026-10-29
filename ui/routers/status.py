"""Brain Status panel router.

Endpoints:
  GET  /status?project=<hash>       -- HTML status page
  GET  /api/status.json?project=<hash> -- JSON status payload
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

_HERE = Path(__file__).resolve().parent.parent   # ui/
_TEMPLATES = Jinja2Templates(directory=str(_HERE / "templates"))

for _p in (_HERE.parent, _HERE.parent / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ui.data import fetch_brain_status, validate_project  # noqa: E402

router = APIRouter()


@router.get("/status", response_class=HTMLResponse)
async def status_page(
    request: Request,
    project: str = Query(..., description="Project hash"),
) -> HTMLResponse:
    """Render the brain status HTML page."""
    validate_project(project)
    from common import discover_projects  # noqa: PLC0415

    status = fetch_brain_status(project)
    projects = discover_projects()
    ctx = {
        "current_project": project,
        "status": status,
        "projects": projects,
    }
    return _TEMPLATES.TemplateResponse(request, "status.html", ctx)


@router.get("/api/status.json")
async def status_json(
    project: str = Query(..., description="Project hash"),
) -> JSONResponse:
    """Return brain status as JSON."""
    validate_project(project)
    data = fetch_brain_status(project)
    return JSONResponse(data)
