"""Scatter panel router — UMAP Embedding Map V1.

Endpoints:
  GET  /scatter?project=<hash>
      Returns full HTML page (extends base.html).  The page bootstraps via
      a JS fetch to /api/scatter.json on load.

  GET  /api/scatter.json?project=<hash>&force=<0|1>
      Returns JSON projection data with bbox field.

  POST /api/scatter/recompute?project=<hash>
      Force-recomputes the projection and returns the same shape as
      /api/scatter.json.

Note: umap-learn is imported lazily inside projection_cache — this module
never imports it directly at load time.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ui.data import fetch_chunk_v05, open_meta_ro, validate_project

# ---------------------------------------------------------------------------
# Path / template setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent   # ui/
_TEMPLATES_DIR = _HERE / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _add_bbox(result: dict) -> dict:
    """Augment the projection dict with a bounding-box field for client scaling."""
    points = result.get("points", [])
    if not points:
        result["bbox"] = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        return result

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    result["bbox"] = {
        "xmin": float(min(xs)),
        "xmax": float(max(xs)),
        "ymin": float(min(ys)),
        "ymax": float(max(ys)),
    }
    return result


def _reason_for_empty(result: dict) -> str:
    """Determine the human-readable reason when points == []."""
    if result.get("model_n", 0) == 0:
        return "not enough data"
    return "no vectors"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/scatter", response_class=HTMLResponse)
async def scatter(
    request: Request,
    project: str = Query("", description="Project hash"),
) -> HTMLResponse:
    """Full HTML page for the scatter panel."""
    if not project:
        return RedirectResponse(url="/", status_code=303)
    validate_project(project)

    # Discover projects for the nav selector
    _root = Path(__file__).resolve().parent.parent.parent
    _scripts = _root / "scripts"
    for _p in (str(_root), str(_scripts)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from common import discover_projects  # noqa: PLC0415

    projects = discover_projects()

    return templates.TemplateResponse(
        request,
        "scatter.html",
        {
            "current_project": project,
            "projects": projects,
        },
    )


def _enrich_points_v05(project: str, points: list[dict]) -> list[dict]:
    """Bulk-read v0.5 extra columns and merge into points in-place.

    Adds: salience, consolidation_round, decay_score (already present but
    double-checked), retrieval_count, co_retrieval_count to each point dict.
    """
    if not points:
        return points
    ids = [p["id"] for p in points]
    try:
        conn = open_meta_ro(project)
        try:
            extra = fetch_chunk_v05(conn, ids)
        finally:
            conn.close()
    except Exception:
        extra = {}

    for p in points:
        v05 = extra.get(p["id"], {})
        p["salience"] = int(v05.get("salience") or 0)
        p["consolidation_round"] = int(v05.get("consolidation_round") or 0)
        p["retrieval_count"] = int(v05.get("retrieval_count_local") or 0)
        p["co_retrieval_count"] = int(v05.get("co_retrieval_count") or 0)
        # decay_score already in the point; only backfill if missing
        if "decay_score" not in p:
            p["decay_score"] = float(v05.get("decay_score") or 1.0)
    return points


@router.get("/api/scatter.json")
async def scatter_json(
    project: str = Query(..., description="Project hash"),
    force: int = Query(0, description="Force recompute (1=yes)"),
    colour_by: str = Query("kind", description="Field to colour points by"),
) -> JSONResponse:
    """Return JSON projection data, computing or loading from cache as needed."""
    validate_project(project)

    # Lazy import — keeps server cold-start fast; umap-learn only loaded here
    from ui.projection_cache import get_or_compute  # noqa: PLC0415

    result = get_or_compute(project, force=bool(force))
    _add_bbox(result)

    if not result["points"]:
        result["reason"] = _reason_for_empty(result)
    else:
        # Enrich with v0.5 columns for client colour-by support
        result["points"] = _enrich_points_v05(project, result["points"])

    result["colour_by"] = colour_by
    return JSONResponse(result)


@router.post("/api/scatter/recompute")
async def scatter_recompute(
    project: str = Query(..., description="Project hash"),
) -> JSONResponse:
    """Force a full UMAP recompute and return the fresh projection."""
    validate_project(project)

    from ui.projection_cache import get_or_compute  # noqa: PLC0415

    result = get_or_compute(project, force=True)
    _add_bbox(result)

    if not result["points"]:
        result["reason"] = _reason_for_empty(result)
    else:
        result["points"] = _enrich_points_v05(project, result["points"])

    return JSONResponse(result)
