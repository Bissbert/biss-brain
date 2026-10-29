"""Backlink Graph panel router (V3).

Endpoints:
  GET  /graph                    -- HTML page (extends base.html)
  GET  /api/graph.json           -- full graph JSON
  POST /api/graph/expand         -- neighborhood around one node
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ui.data import (
    fetch_chunk_texts,
    open_lance,
    open_meta_ro,
    parse_cross_links,
    validate_project,
)

router = APIRouter()

_HERE = Path(__file__).resolve().parent.parent   # ui/
_templates = Jinja2Templates(directory=str(_HERE / "templates"))

# Ensure scripts/ is on the path for backlinks.parse_related_ids
for _p in (_HERE.parent, _HERE.parent / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backlinks import parse_related_ids  # noqa: E402


# ---------------------------------------------------------------------------
# Union-Find for connected-component counting
# ---------------------------------------------------------------------------

def _union_find_components(node_ids: list[str], edges: list[tuple[str, str]]) -> int:
    """Return the number of connected components for isolated nodes + edges."""
    parent: dict[str, str] = {n: n for n in node_ids}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            _union(a, b)

    return len({_find(n) for n in node_ids})


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------

def _build_graph(
    project: str,
    include_superseded: int = 0,
    min_degree: int = 0,
) -> dict[str, Any]:
    """Query SQLite and build the full node/edge graph dict."""
    conn = open_meta_ro(project)
    try:
        # Try to read cross_links column (v0.5+); fall back gracefully if absent
        try:
            if include_superseded:
                rows = conn.execute(
                    "SELECT id, kind, ts, pinned, decay_score, superseded_by, "
                    "related_ids, cross_links FROM chunks"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, kind, ts, pinned, decay_score, superseded_by, "
                    "related_ids, cross_links FROM chunks WHERE superseded_by = ''"
                ).fetchall()
        except Exception:
            # Older DB without cross_links column
            if include_superseded:
                rows = conn.execute(
                    "SELECT id, kind, ts, pinned, decay_score, superseded_by, "
                    "related_ids FROM chunks"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, kind, ts, pinned, decay_score, superseded_by, "
                    "related_ids FROM chunks WHERE superseded_by = ''"
                ).fetchall()
    finally:
        conn.close()

    # Build adjacency and typed edge map: (src, dst) -> type
    # Use a dict instead of set so we can store the type
    typed_edges: dict[tuple[str, str], str] = {}  # (min_id, max_id) -> type
    all_ids: set[str] = set()
    adjacency: dict[str, set[str]] = {}

    for row in rows:
        row_dict = dict(row)
        nid = row_dict["id"]
        all_ids.add(nid)
        if nid not in adjacency:
            adjacency[nid] = set()

        # Parse related_ids using v0.5 typed adapter
        related_raw = row_dict.get("related_ids") or ""
        typed_items = parse_related_ids(related_raw)
        for item in typed_items:
            rid = item.get("id", "")
            if not rid or rid == nid:
                continue
            edge_key = (min(nid, rid), max(nid, rid))
            edge_type = item.get("type", "co_occurs")
            # Prefer more specific type if already seen
            if edge_key not in typed_edges:
                typed_edges[edge_key] = edge_type
            adjacency[nid].add(rid)

        # Parse cross_links (v0.5 column)
        cross_raw = row_dict.get("cross_links", "[]")
        cross_items = parse_cross_links(cross_raw)
        for cl in cross_items:
            global_id = cl.get("global_id") or cl.get("id")
            if not global_id or global_id == nid:
                continue
            edge_key = (min(nid, global_id), max(nid, global_id))
            # cross_link edges always win the type contest
            typed_edges[edge_key] = "cross_link"
            adjacency[nid].add(global_id)
            all_ids.add(global_id)  # ensure global node appears

    # Degree per node
    degree: dict[str, int] = {nid: 0 for nid in all_ids}
    for a, b in typed_edges:
        if a in degree:
            degree[a] += 1
        if b in degree:
            degree[b] += 1

    # Apply min_degree filter
    if min_degree > 0:
        surviving = {nid for nid, d in degree.items() if d >= min_degree}
        typed_edges = {
            (a, b): t for (a, b), t in typed_edges.items()
            if a in surviving and b in surviving
        }
        degree = {nid: 0 for nid in surviving}
        for a, b in typed_edges:
            degree[a] += 1
            degree[b] += 1
        all_ids = surviving

    # Hub detection: degree > p90
    p90_degree = 0
    if degree:
        sorted_degrees = sorted(degree.values())
        p90_idx = max(0, int(len(sorted_degrees) * 0.9) - 1)
        p90_degree = sorted_degrees[p90_idx]

    # Fetch text from LanceDB for labels/tooltips
    tbl = open_lance(project)
    id_list = list(all_ids)
    texts = fetch_chunk_texts(tbl, id_list) if tbl and id_list else {}

    # Build row lookup for metadata (convert sqlite3.Row to plain dict)
    row_meta: dict[str, Any] = {}
    for row in rows:
        row_dict = dict(row)
        if row_dict["id"] in all_ids:
            row_meta[row_dict["id"]] = row_dict

    # Assemble nodes
    nodes: list[dict] = []
    for nid in all_ids:
        meta = row_meta.get(nid, {})
        text = texts.get(nid, "")
        label = text[:60] if text else nid[:20]
        tooltip = text[:200] if text else ""
        deg = degree.get(nid, 0)
        nodes.append({
            "id": nid,
            "label": label,
            "kind": meta.get("kind", "unknown") if meta else "unknown",
            "ts": meta.get("ts", 0) if meta else 0,
            "pinned": int(meta.get("pinned", 0)) if meta else 0,
            "decay_score": float(meta.get("decay_score", 1.0)) if meta else 1.0,
            "superseded": bool(meta.get("superseded_by", "")) if meta else False,
            "is_hub": deg > p90_degree if p90_degree > 0 else False,
            "degree": deg,
            "tooltip": tooltip,
        })

    # Assemble edges with type
    edges: list[dict] = [
        {"id": f"{a}--{b}", "source": a, "target": b, "type": t}
        for (a, b), t in typed_edges.items()
        if a in all_ids and b in all_ids
    ]

    edge_tuples = [(e["source"], e["target"]) for e in edges]
    components = _union_find_components(list(all_ids), edge_tuples) if all_ids else 0
    max_degree = max(degree.values()) if degree else 0

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "components": components,
            "max_degree": max_degree,
            "p90_degree": p90_degree,
        },
    }


# ---------------------------------------------------------------------------
# Neighborhood expander (BFS up to depth)
# ---------------------------------------------------------------------------

def _expand_neighborhood(
    project: str,
    node_id: str,
    depth: int = 1,
) -> dict[str, Any]:
    """Return the local neighborhood of node_id up to depth hops."""
    depth = max(1, min(depth, 3))

    conn = open_meta_ro(project)
    try:
        try:
            rows = conn.execute(
                "SELECT id, kind, ts, pinned, decay_score, superseded_by, "
                "related_ids, cross_links FROM chunks"
            ).fetchall()
        except Exception:
            rows = conn.execute(
                "SELECT id, kind, ts, pinned, decay_score, superseded_by, "
                "related_ids FROM chunks"
            ).fetchall()
    finally:
        conn.close()

    # Build full adjacency (bidirectional) with typed edges
    adjacency: dict[str, set[str]] = {}
    typed_adj: dict[tuple[str, str], str] = {}  # (min, max) -> type
    row_meta: dict[str, Any] = {}

    for row in rows:
        row_dict = dict(row)
        nid = row_dict["id"]
        row_meta[nid] = row_dict
        adjacency.setdefault(nid, set())

        related_raw = row_dict.get("related_ids") or ""
        typed_items = parse_related_ids(related_raw)
        for item in typed_items:
            rid = item.get("id", "")
            if not rid or rid == nid:
                continue
            adjacency[nid].add(rid)
            adjacency.setdefault(rid, set()).add(nid)
            ek = (min(nid, rid), max(nid, rid))
            if ek not in typed_adj:
                typed_adj[ek] = item.get("type", "co_occurs")

        cross_raw = row_dict.get("cross_links", "[]")
        cross_items = parse_cross_links(cross_raw)
        for cl in cross_items:
            global_id = cl.get("global_id") or cl.get("id")
            if not global_id or global_id == nid:
                continue
            adjacency[nid].add(global_id)
            adjacency.setdefault(global_id, set()).add(nid)
            ek = (min(nid, global_id), max(nid, global_id))
            typed_adj[ek] = "cross_link"

    if node_id not in adjacency and node_id not in row_meta:
        return {"nodes": [], "edges": [], "summary": {
            "node_count": 0, "edge_count": 0, "components": 0,
            "max_degree": 0, "p90_degree": 0,
        }}

    # BFS
    visited: set[str] = {node_id}
    frontier: set[str] = {node_id}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for nid in frontier:
            for neighbor in adjacency.get(nid, set()):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
        visited |= next_frontier
        frontier = next_frontier

    # Collect typed edges among visited nodes
    visit_edges: dict[tuple[str, str], str] = {
        (a, b): t for (a, b), t in typed_adj.items()
        if a in visited and b in visited
    }

    degree: dict[str, int] = {nid: 0 for nid in visited}
    for a, b in visit_edges:
        if a in degree:
            degree[a] += 1
        if b in degree:
            degree[b] += 1

    p90_degree = 0
    if degree:
        sorted_degrees = sorted(degree.values())
        p90_idx = max(0, int(len(sorted_degrees) * 0.9) - 1)
        p90_degree = sorted_degrees[p90_idx]

    tbl = open_lance(project)
    texts = fetch_chunk_texts(tbl, list(visited)) if tbl else {}

    nodes: list[dict] = []
    for nid in visited:
        meta = row_meta.get(nid, {})
        text = texts.get(nid, "")
        label = text[:60] if text else nid[:20]
        tooltip = text[:200] if text else ""
        deg = degree.get(nid, 0)
        nodes.append({
            "id": nid,
            "label": label,
            "kind": meta.get("kind", "unknown") if meta else "unknown",
            "ts": meta.get("ts", 0) if meta else 0,
            "pinned": int(meta.get("pinned", 0)) if meta else 0,
            "decay_score": float(meta.get("decay_score", 1.0)) if meta else 1.0,
            "superseded": bool(meta.get("superseded_by", "")) if meta else False,
            "is_hub": deg > p90_degree if p90_degree > 0 else False,
            "degree": deg,
            "tooltip": tooltip,
        })

    edges = [
        {"id": f"{a}--{b}", "source": a, "target": b, "type": t}
        for (a, b), t in visit_edges.items()
    ]
    edge_tuples = [(e["source"], e["target"]) for e in edges]
    components = _union_find_components(list(visited), edge_tuples) if visited else 0
    max_degree = max(degree.values()) if degree else 0

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "components": components,
            "max_degree": max_degree,
            "p90_degree": p90_degree,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/graph")
async def graph_page(
    request: Request,
    project: str = Query("", description="Project hash"),
) -> Any:
    if not project:
        return RedirectResponse(url="/", status_code=303)
    validate_project(project)
    # Fetch summary for header display
    try:
        data = _build_graph(project)
        summary = data["summary"]
    except Exception:
        summary = {"node_count": 0, "edge_count": 0, "components": 0,
                   "max_degree": 0, "p90_degree": 0}

    # Build project list for nav
    from common import discover_projects
    projects = discover_projects()

    # Starlette 1.0+ API: TemplateResponse(request, name, context)
    return _templates.TemplateResponse(
        request,
        "graph.html",
        {
            "current_project": project,
            "projects": projects,
            "summary": summary,
        },
    )


@router.get("/api/graph.json")
async def graph_json(
    project: str = Query(..., description="Project hash"),
    include_superseded: int = Query(0, ge=0, le=1),
    min_degree: int = Query(0, ge=0),
) -> JSONResponse:
    validate_project(project)
    data = _build_graph(project, include_superseded=include_superseded, min_degree=min_degree)
    return JSONResponse(data)


@router.post("/api/graph/expand")
async def graph_expand(
    project: str = Query(..., description="Project hash"),
    id: str = Query(..., description="Node chunk id to expand from"),
    depth: int = Query(1, ge=1, le=3),
) -> JSONResponse:
    validate_project(project)
    data = _expand_neighborhood(project, id, depth=depth)
    return JSONResponse(data)
