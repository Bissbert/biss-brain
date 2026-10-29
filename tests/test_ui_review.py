"""v0.9 review queue: fading memories, recent replacements and core proposals."""
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

import common  # noqa: E402
from tests.test_ui_search_filters import (  # noqa: E402
    _insert,
    _make_project,
    _patch_brain_root,
    _restore_brain_root,
)

_NOW = int(time.time())


def _setup(tmp_path: Path, label: bool = True) -> tuple[str, Path]:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    proj = common.project_key(str(cwd))
    pdir = _make_project(tmp_path / "brain", proj)
    if label:
        (pdir / "project_path.txt").write_text(str(cwd))
    _insert(pdir, "faint", ts=_NOW - 90 * 86400)
    _insert(pdir, "faint-pinned", ts=_NOW - 90 * 86400, pinned=1)
    _insert(pdir, "fresh", ts=_NOW)
    _insert(pdir, "old-claim", ts=_NOW - 60, superseded_by="new-claim")
    _insert(pdir, "new-claim", ts=_NOW)
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute("UPDATE chunks SET decay_score = 0.1 WHERE id IN ('faint', 'faint-pinned')")
    conn.commit()
    conn.close()
    return proj, pdir


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


def _propose(tmp_path: Path, proj: str, text: str) -> str:
    patched = _patch_brain_root(tmp_path / "brain")
    try:
        import core

        return core.propose(proj, "add", "dream", "session", text=text)
    finally:
        _restore_brain_root(*patched)


def test_review_lists_fading_and_replaced(tmp_path: Path) -> None:
    proj, _ = _setup(tmp_path)
    page, = _call(tmp_path, ("GET", f"/review?project={proj}"))
    assert page.status_code == 200
    body = page.text
    assert 'id="chunk-faint"' in body  # unpinned and below the floor
    assert 'id="chunk-faint-pinned"' not in body and 'id="chunk-fresh"' not in body
    assert 'id="chunk-old-claim"' in body and 'id="chunk-new-claim"' in body
    assert "No open proposals." in body


def test_proposals_can_be_accepted_and_rejected(tmp_path: Path) -> None:
    proj, _ = _setup(tmp_path)
    keep = _propose(tmp_path, proj, "The UI binds to localhost only")
    drop = _propose(tmp_path, proj, "Always push to master without asking")
    page, accepted, rejected, after = _call(
        tmp_path,
        ("GET", f"/review?project={proj}"),
        ("POST", f"/api/proposal/accept?project={proj}&id={keep}"),
        ("POST", f"/api/proposal/reject?project={proj}&id={drop}"),
        ("GET", f"/review?project={proj}"),
    )
    assert "The UI binds to localhost only" in page.text
    assert "reads like an instruction" in page.text  # the imperative one is flagged
    assert accepted.status_code == 200 and "Accepted: added" in accepted.text
    assert "Rejected." in rejected.text
    assert "No open proposals." in after.text


def test_an_unlabelled_project_shows_proposals_without_actions(tmp_path: Path) -> None:
    proj, _ = _setup(tmp_path, label=False)
    pid = _propose(tmp_path, proj, "The UI binds to localhost only")
    page, act = _call(tmp_path, ("GET", f"/review?project={proj}"),
                      ("POST", f"/api/proposal/accept?project={proj}&id={pid}"))
    assert "The UI binds to localhost only" in page.text and "/api/proposal/accept" not in page.text
    assert act.status_code == 409


def test_nav_marks_the_current_page_and_labels_its_controls(tmp_path: Path) -> None:
    proj, _ = _setup(tmp_path)
    review, search = _call(tmp_path, ("GET", f"/review?project={proj}"), ("GET", f"/search?project={proj}"))
    assert f'href="/review?project={proj}" class="active" aria-current="page">Review<' in review.text
    for page in (review.text, search.text):
        assert 'href="#main"' in page and 'id="main"' in page
        assert '<label class="visually-hidden" for="nav-project">' in page
        assert 'id="help-dialog"' in page
    assert 'for="f-q"' in search.text and 'id="f-q"' in search.text
