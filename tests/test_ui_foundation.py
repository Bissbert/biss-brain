"""Phase 0 smoke tests for the biss-brain UI foundation.

test_discover_projects_runs  – ensure discover_projects() returns a well-formed list
test_server_starts           – ensure the FastAPI app responds to /healthz in-process
test_router_modules          – each panel router exports an APIRouter instance
test_api_label_form          – POST /api/label with form-encoded body writes the file
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path so `ui` package resolves before
# `scripts/ui.py` (which would shadow it if scripts/ came first).
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
# Insert root first so ui/ package takes precedence over scripts/ui.py
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)

import pytest
import httpx


# ---------------------------------------------------------------------------
# Test 1: discover_projects
# ---------------------------------------------------------------------------

def test_discover_projects_runs() -> None:
    from common import discover_projects

    result = discover_projects()
    assert isinstance(result, list), "discover_projects() must return a list"

    expected_keys = {"hash", "path", "chunk_count", "last_active_ts"}
    for item in result:
        assert isinstance(item, dict), f"Each item must be a dict, got: {type(item)}"
        missing = expected_keys - item.keys()
        assert not missing, f"Dict missing keys: {missing}. Got: {set(item.keys())}"
        assert isinstance(item["hash"], str) and len(item["hash"]) == 12
        assert item["path"] is None or isinstance(item["path"], str)
        assert isinstance(item["chunk_count"], int)
        assert item["last_active_ts"] is None or isinstance(item["last_active_ts"], int)


# ---------------------------------------------------------------------------
# Test 2: server starts and /healthz responds
# ---------------------------------------------------------------------------

import anyio


def test_server_starts() -> None:
    from ui.server import app

    async def _run() -> dict:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/healthz")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            return response.json()

    data = anyio.run(_run)
    assert data == {"ok": True}, f"Expected {{\"ok\": True}}, got: {data}"


# ---------------------------------------------------------------------------
# Test 3: each panel router module exports an APIRouter
# ---------------------------------------------------------------------------

def test_router_modules() -> None:
    from fastapi import APIRouter

    from ui.routers import search, sessions, timeline

    for name, module in [
        ("search", search),
        ("sessions", sessions),
        ("timeline", timeline),
    ]:
        assert hasattr(module, "router"), f"ui.routers.{name} has no 'router' attribute"
        assert isinstance(module.router, APIRouter), (
            f"ui.routers.{name}.router is {type(module.router)}, expected APIRouter"
        )


# ---------------------------------------------------------------------------
# Test 4: POST /api/label with form-encoded body writes project_path.txt
# ---------------------------------------------------------------------------

def test_api_label_form(tmp_path: Path) -> None:
    """Verify that /api/label reads hash and path from the form body."""
    import os
    from common import BRAIN_ROOT
    from ui.server import app

    # Create a fake project directory under BRAIN_ROOT
    fake_hash = "aabbccddeeff"
    fake_project_dir = BRAIN_ROOT / fake_hash
    fake_project_dir.mkdir(parents=True, exist_ok=True)

    # Use tmp_path as the "project path" – it always exists
    label_file = fake_project_dir / "project_path.txt"
    # Remove any previous label so the test starts clean
    if label_file.exists():
        label_file.unlink()

    async def _run() -> dict:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/label",
                data={"hash": fake_hash, "path": str(tmp_path)},
            )
            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}: {response.text}"
            )
            return response.json()

    data = anyio.run(_run)
    assert data["ok"] is True
    assert data["hash"] == fake_hash

    # The label file must have been written
    assert label_file.exists(), "project_path.txt was not written"
    written = label_file.read_text().strip()
    assert written == str(tmp_path.resolve()), (
        f"Expected {tmp_path.resolve()}, got {written}"
    )

    # Cleanup
    label_file.unlink(missing_ok=True)
    try:
        fake_project_dir.rmdir()
    except OSError:
        pass  # not empty – that's fine, leave it
