"""Shared pool: what every project can recall, and taking a row back out.

Endpoints:
  GET  /pool[?project=<hash>]          -- live pool rows, where each came from
  POST /api/pool/unshare?id=<pool id>  -- out of the pool; every writer keeps its copy (Undo offered)
  POST /api/pool/restore?id=<pool id>  -- undo that
"""
from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

import globalpool
from common import GLOBAL_PROJECT_HASH, discover_projects
from ui.data import fetch_chunk_texts, open_lance, safe_id
from ui.routers.search import templates

router = APIRouter()
LIMIT = 200


def _name(h: str) -> str:
    """Folder name of project *h* from its project_path.txt, else the hash."""
    try:
        path = (globalpool._root() / h / "project_path.txt").read_text().strip()
    except OSError:
        path = ""
    return path.rstrip("/").split("/")[-1] or h


def _rows(ids: list[str] | None = None) -> list[dict]:
    """Pool rows (live ones when *ids* is None), newest first, with text and origin."""
    path = globalpool.pool_dir() / "meta.sqlite"
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if ids is None:
            rows = conn.execute("SELECT id, kind, ts, superseded_by, source_project, source_projects FROM chunks "
                                "WHERE project = ? AND superseded_by = '' ORDER BY ts DESC LIMIT ?",
                                (GLOBAL_PROJECT_HASH, LIMIT)).fetchall()
        else:
            q = ",".join("?" * len(ids))
            rows = conn.execute("SELECT id, kind, ts, superseded_by, source_project, source_projects FROM chunks "
                                f"WHERE project = ? AND id IN ({q})", (GLOBAL_PROJECT_HASH, *ids)).fetchall()
    finally:
        conn.close()
    texts = fetch_chunk_texts(open_lance(GLOBAL_PROJECT_HASH), [r["id"] for r in rows])
    out = []
    for r in rows:
        try:
            joined = json.loads(r["source_projects"] or "[]")
        except ValueError:
            joined = []
        projects = list(dict.fromkeys(h for h in [r["source_project"], *joined] if h))
        out.append({
            "id": r["id"], "kind": r["kind"], "ts": r["ts"], "text": texts.get(r["id"], ""),
            "live": not r["superseded_by"],
            "projects": [_name(h) for h in projects],
        })
    return out


@router.get("/pool", response_class=HTMLResponse)
async def pool(request: Request, project: str = Query("", description="Project hash, for the nav")) -> HTMLResponse:
    projects = discover_projects()
    ctx = {
        "current_project": project if any(p["hash"] == project for p in projects) else "",
        "projects": projects,
        "rows": _rows(),
        "limit": LIMIT,
    }
    return templates.TemplateResponse(request, "pool.html", ctx)


def _row_response(request: Request, gid: str, **extra: object) -> HTMLResponse:
    rows = _rows([gid])
    if not rows:
        raise HTTPException(status_code=404, detail=f"Not in the pool: {gid}")
    return templates.TemplateResponse(request, "_pool_row.html", {"row": rows[0], **extra})


@router.post("/api/pool/{action}", response_class=HTMLResponse)
async def pool_action(request: Request, action: str, id: str = Query(...)) -> HTMLResponse:
    """Unshare a pool row (each project that wrote it keeps it local) or undo that."""
    if action not in ("unshare", "restore") or safe_id(id) is None:
        raise HTTPException(status_code=404)
    if action == "unshare":
        done = globalpool.unshare_pool(id)
        return _row_response(request, id, just_unshared=done)
    globalpool.restore_pool(id)
    return _row_response(request, id)
