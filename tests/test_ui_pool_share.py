"""Stage 4 UI: Share on a memory's card, and the Shared page, each with Undo."""
from __future__ import annotations

import sys
from pathlib import Path

import anyio
import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "scripts", _ROOT, _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dream  # noqa: E402
import test_v10_pool  # noqa: E402
from test_v10_pool import A, B, _pool_rows, _row, _store, _vec  # noqa: E402

pool = test_v10_pool.pool  # the fixture


@pytest.fixture
def ui(pool, monkeypatch):
    """The pool fixture with the UI pointed at it."""
    import ui.data as _data
    from ui.routers import search

    root, cfg = pool
    monkeypatch.setitem(sys.modules, "dream", dream)  # globalpool.share imports it lazily
    monkeypatch.setattr(_data, "BRAIN_ROOT", root)
    monkeypatch.setattr(_data, "_lance_cache", {})
    monkeypatch.setattr(search, "load_config", lambda: cfg)
    return root, cfg


def _call(*requests: tuple[str, str]) -> list[httpx.Response]:
    from ui.server import app

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return [await client.request(m, url) for m, url in requests]

    return anyio.run(run)


def _live(root: Path) -> list[str]:
    p = root / "_global" / "meta.sqlite"
    return sorted(i for i, r in _pool_rows(root).items() if not r["superseded_by"]) if p.exists() else []


def test_card_share_then_undo_puts_everything_back(ui):
    root, cfg = ui
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format replaces black", "vec": _vec(1),
                                  "scope": "project"}])
    q = f"?project={A}&id=a1"
    shared, = _call(("POST", f"/api/share{q}"))
    assert shared.status_code == 200
    assert _live(root) == ["a1"]
    assert "Shared with every project" in shared.text and 'role="status"' in shared.text
    assert "autofocus" in shared.text and ">shared<" in shared.text and ">Unshare<" in shared.text
    assert "pinned=0&scope=project" in shared.text  # the Undo carries what to put back

    undone, = _call(("POST", f"/api/unshare{q}&pinned=0&scope=project"))
    assert undone.status_code == 200 and 'role="status"' not in undone.text
    assert _live(root) == []
    r = _row(root, A, "a1")
    assert (r["pinned"], r["scope"], r["global_veto"]) == (0, "project", "user")
    assert ">Share<" in undone.text


def test_card_unshare_offers_undo_that_shares_again(ui):
    root, cfg = ui
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format replaces black", "vec": _vec(1)}])
    dream._stage_promote_to_global(root / A, cfg)
    q = f"?project={A}&id=a1"
    off, = _call(("POST", f"/api/unshare{q}"))
    assert _live(root) == [] and "Kept in this project only" in off.text and "autofocus" in off.text
    assert f'hx-post="/api/share?project={A}&id=a1"' in off.text
    _call(("POST", f"/api/share{q}"))
    assert _live(root) == ["a1"]


def test_card_says_why_a_share_was_refused(ui):
    root, cfg = ui
    _store(root, A, "/x/alpha", [{"id": "s1", "text": "the token is ghp_" + "a1B2" * 9, "vec": _vec(1)}])
    resp, = _call(("POST", f"/api/share?project={A}&id=s1"))
    assert resp.status_code == 200
    assert "looks like it holds a secret" in resp.text and "Undo" not in resp.text
    assert _live(root) == []


def test_share_button_only_where_sharing_can_happen(ui):
    root, cfg = ui
    _store(root, A, "/x/alpha", [
        {"id": "a1", "text": "ruff format replaces black", "vec": _vec(1)},
        {"id": "h1", "text": "maybe the cache is stale", "vec": _vec(2), "kind": "hypothesis"},
    ])
    from ui.routers.search import load_cards, templates

    def render(cid: str, project: str = A) -> str:
        card = load_cards(project, [cid])[0]
        return templates.get_template("_chunk_card.html").render(project=project, chunk=card)

    assert 'data-action="share"' in render("a1")
    assert 'data-action="share"' not in render("h1")
    import sqlite3
    c = sqlite3.connect(str(root / A / "meta.sqlite"))
    c.execute("UPDATE chunks SET taint = '[\"web\"]', source = 'web' WHERE id = 'a1'")
    c.commit()
    c.close()
    assert 'data-action="share"' not in render("a1")


def test_unshare_refuses_the_pool_store_itself(ui):
    root, cfg = ui
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format replaces black", "vec": _vec(1)}])
    dream._stage_promote_to_global(root / A, cfg)
    from common import GLOBAL_PROJECT_HASH

    a, b = _call(("POST", f"/api/share?project={GLOBAL_PROJECT_HASH}&id=a1"),
                 ("POST", f"/api/unshare?project={GLOBAL_PROJECT_HASH}&id=a1"))
    assert (a.status_code, b.status_code) == (400, 400)
    assert _live(root) == ["a1"]


def test_pool_page_lists_rows_with_their_projects_and_unshare_undoes(ui):
    root, cfg = ui
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format replaces black", "vec": _vec(1)}])
    _store(root, B, "/x/beta", [{"id": "b1", "text": "ruff format stands in for black", "vec": _vec(1)}])
    dream._stage_promote_to_global(root / A, cfg)
    dream._stage_promote_to_global(root / B, cfg)

    page, = _call(("GET", "/pool"))
    assert page.status_code == 200
    assert "ruff format replaces black" in page.text
    assert "from alpha, written in 2 projects" in page.text
    assert 'href="/pool"' in page.text and 'aria-current="page"' in page.text

    off, = _call(("POST", "/api/pool/unshare?id=a1"))
    assert _live(root) == [] and "Out of the pool" in off.text and "autofocus" in off.text
    assert _row(root, B, "b1")["global_veto"] == "user"
    assert "ruff format replaces black" not in _call(("GET", "/pool"))[0].text

    back, = _call(("POST", "/api/pool/restore?id=a1"))
    assert back.status_code == 200 and _live(root) == ["a1"]
    assert 'role="status"' not in back.text and ">Unshare<" in back.text
    assert _row(root, B, "b1")["cross_eligible"] == 1


def test_pool_page_empty_and_bad_ids(ui):
    page, = _call(("GET", "/pool"))
    assert page.status_code == 200 and "Nothing is shared yet" in page.text
    bad, nope, unknown = _call(("POST", "/api/pool/unshare?id=x' OR 1=1"),
                               ("POST", "/api/pool/delete?id=a1"),
                               ("POST", "/api/pool/unshare?id=a1"))
    assert (bad.status_code, nope.status_code, unknown.status_code) == (404, 404, 404)
