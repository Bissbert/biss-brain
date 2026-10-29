"""v0.9 cards: forget can be undone, a replacement cannot, and each card says
what it replaced or what replaced it."""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

import anyio
import httpx

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "scripts", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("BISS_BRAIN_FAKE_EMBED", "1")

from tests.test_ui_search_filters import (  # noqa: E402
    _insert,
    _make_project,
    _patch_brain_root,
    _restore_brain_root,
)

_HASH = "cafe12345678"
_NOW = int(time.time())


def _project(tmp_path: Path) -> Path:
    pdir = _make_project(tmp_path / "brain", _HASH)
    _insert(pdir, "old-claim", ts=_NOW - 60, superseded_by="new-claim")
    _insert(pdir, "new-claim", ts=_NOW)
    _insert(pdir, "plain-note", ts=_NOW - 30)
    return pdir


def _sup(pdir: Path, cid: str) -> str:
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    try:
        return conn.execute("SELECT superseded_by FROM chunks WHERE id = ?", (cid,)).fetchone()[0]
    finally:
        conn.close()


def _call(tmp_path: Path, *requests: tuple[str, str]) -> list[httpx.Response]:
    from ui.server import app

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return [await client.request(m, url) for m, url in requests]

    patched = _patch_brain_root(tmp_path / "brain")
    try:
        return anyio.run(run)
    finally:
        _restore_brain_root(*patched)


def test_forget_offers_undo_and_undo_restores(tmp_path: Path) -> None:
    pdir = _project(tmp_path)
    q = f"?project={_HASH}&id=plain-note"
    forgot, = _call(tmp_path, ("POST", f"/api/forget{q}"))
    assert forgot.status_code == 200
    assert _sup(pdir, "plain-note") != ""
    assert "Undo" in forgot.text and 'role="status"' in forgot.text
    assert "forgotten" in forgot.text

    restored, = _call(tmp_path, ("POST", f"/api/unforget{q}"))
    assert restored.status_code == 200
    assert _sup(pdir, "plain-note") == ""
    assert "forgotten" not in restored.text and 'data-action="forget"' in restored.text


def test_unforget_leaves_a_replacement_alone(tmp_path: Path) -> None:
    pdir = _project(tmp_path)
    resp, = _call(tmp_path, ("POST", f"/api/unforget?project={_HASH}&id=old-claim"))
    assert resp.status_code == 200
    assert _sup(pdir, "old-claim") == "new-claim"


def test_forget_does_not_overwrite_a_replacement(tmp_path: Path) -> None:
    pdir = _project(tmp_path)
    resp, = _call(tmp_path, ("POST", f"/api/forget?project={_HASH}&id=old-claim"))
    assert _sup(pdir, "old-claim") == "new-claim"
    assert "Undo" not in resp.text


def test_cards_show_replaces_and_replaced_by(tmp_path: Path) -> None:
    _project(tmp_path)
    page, = _call(tmp_path, ("GET", f"/search?project={_HASH}&include_superseded=1&kind=note"))
    body = page.text
    assert "replaced by <code" in body and ">new-claim<" in body
    assert "replaces <code" in body and ">old-claim<" in body


def test_lineage_tells_tombstones_from_replacements(tmp_path: Path) -> None:
    pdir = _project(tmp_path)
    _insert(pdir, "gone", superseded_by="0000-tombstone")
    from ui.data import fetch_lineage

    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    try:
        got = fetch_lineage(conn, ["old-claim", "new-claim", "gone", "plain-note"])
    finally:
        conn.close()
    assert got["old-claim"]["replaced_by"] == "new-claim" and not got["old-claim"]["forgotten"]
    assert got["new-claim"]["replaces"] == ["old-claim"]
    assert got["gone"]["forgotten"] and got["gone"]["replaced_by"] == ""
    assert got["plain-note"] == {"superseded_by": "", "replaced_by": "", "forgotten": False, "replaces": []}
