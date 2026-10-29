"""Review queue: the few memories that want a decision.

Endpoints:
  GET  /review?project=<hash>                     -- fading, recently replaced, proposed core notes
  POST /api/proposal/{accept,reject}?project=&id= -- act on a core proposal
"""
from __future__ import annotations

import html
import sqlite3
import time

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

import provenance
from common import GLOBAL_PROJECT_HASH, discover_projects
from ui.data import open_meta_ro, safe_id, validate_project
from ui.routers.search import load_cards, templates

router = APIRouter()

# decay_score = exp(-age_days / 30), times five when pinned: an unpinned
# memory falls below this at about six weeks (hygiene.decay_usage_boost adds recalls).
FADING_BELOW = 0.25
REPLACED_WITHIN_S = 30 * 86400
LIMIT = 20


def _fading(conn: sqlite3.Connection, project: str) -> tuple[list[str], int]:
    where = ("FROM chunks WHERE project = ? AND superseded_by = '' AND pinned = 0 "
             "AND kind NOT IN ('hypothesis', 'quarantine') AND decay_score < ?")
    total = conn.execute(f"SELECT COUNT(*) {where}", (project, FADING_BELOW)).fetchone()[0]
    ids = [r[0] for r in conn.execute(
        f"SELECT id {where} ORDER BY decay_score, ts LIMIT ?", (project, FADING_BELOW, LIMIT))]
    return ids, total


def _replaced(conn: sqlite3.Connection, project: str) -> list[tuple[str, str]]:
    """(old, new) pairs whose replacement was written in the last 30 days, newest first."""
    return [(r[0], r[1]) for r in conn.execute(
        "SELECT c.id, n.id FROM chunks c JOIN chunks n ON n.id = c.superseded_by AND n.project = c.project "
        "WHERE c.project = ? AND n.ts >= ? ORDER BY n.ts DESC LIMIT ?",
        (project, int(time.time()) - REPLACED_WITHIN_S, LIMIT))]


def _proposals(project: str) -> list[dict]:
    """Open core proposals for the project and the global layer, read-only."""
    import core

    path = core.index_path()
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM core_proposals WHERE status = 'open' AND project IN (?, ?) ORDER BY ts",
                (project, GLOBAL_PROJECT_HASH)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [{**dict(r), "imperative": bool(r["text"]) and provenance.looks_imperative(r["text"])} for r in rows]


@router.get("/review", response_class=HTMLResponse)
async def review(request: Request, project: str = Query(..., description="Project hash")) -> HTMLResponse:
    validate_project(project)
    from ui.preview import project_cwd

    conn = open_meta_ro(project)
    try:
        fading_ids, fading_total = _fading(conn, project)
        pairs = _replaced(conn, project)
    finally:
        conn.close()
    cards = {c["id"]: c for c in load_cards(project, fading_ids + [i for p in pairs for i in p])}
    ctx = {
        "project": project,
        "current_project": project,
        "projects": discover_projects(),
        "fading": [cards[i] for i in fading_ids if i in cards],
        "fading_total": fading_total,
        "replaced": [(cards[o], cards[n]) for o, n in pairs if o in cards and n in cards],
        "proposals": _proposals(project),
        "can_act": project_cwd(project) is not None,
    }
    return templates.TemplateResponse(request, "review.html", ctx)


@router.post("/api/proposal/{action}", response_class=HTMLResponse)
async def proposal(action: str, project: str = Query(...), id: str = Query(...)) -> HTMLResponse:
    """Accept or reject one open core proposal; returns the row's replacement."""
    if action not in ("accept", "reject") or safe_id(id) is None:
        raise HTTPException(status_code=404)
    validate_project(project)
    from ui.preview import project_cwd

    cwd = project_cwd(project)
    if cwd is None:
        raise HTTPException(status_code=409, detail="Label the project with its path first")
    import core

    try:
        if action == "accept":
            msg = "Accepted: " + core.accept(cwd, id)
        else:
            core.reject(cwd, id)
            msg = "Rejected."
    except core.CoreError as e:
        msg = str(e)
    return HTMLResponse(f'<li class="proposal is-done" role="status">{html.escape(msg)}</li>')
