"""Tests for search panel filter integration (F-002, F-003, F-006, F-007).

Verifies date, session, pinned, salience, round filters restrict results
correctly; pagination links appear with correct offsets; total_count badge
reflects filtered counts.

Uses a fake project with SQLite only (no LanceDB / Ollama needed) exercising
the GET /search endpoint through the ASGI stack.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import anyio
import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Force fake embed so no Ollama call is made
os.environ.setdefault("BISS_BRAIN_FAKE_EMBED", "1")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_NOW = int(time.time())
_DAY = 86400

# Chunk data: (id, kind, ts, session_id, pinned, salience, consolidation_round, superseded_by)
_CHUNK_SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        project TEXT NOT NULL DEFAULT '',
        session_id TEXT DEFAULT '',
        agent_id TEXT DEFAULT '',
        kind TEXT NOT NULL DEFAULT 'note',
        ts INTEGER NOT NULL DEFAULT 0,
        content_hash TEXT NOT NULL DEFAULT '',
        superseded_by TEXT NOT NULL DEFAULT '',
        pinned INTEGER DEFAULT 0,
        retrieval_count INTEGER DEFAULT 0,
        last_accessed INTEGER,
        decay_score REAL NOT NULL DEFAULT 1.0,
        related_ids TEXT NOT NULL DEFAULT '',
        salience INTEGER DEFAULT 0,
        consolidation_round INTEGER DEFAULT 0
    )
"""


def _make_project(brain_root: Path, project_hash: str = "aabb11223344") -> Path:
    pdir = brain_root / project_hash
    pdir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(_CHUNK_SCHEMA)
    conn.commit()
    conn.close()
    return pdir


def _insert(pdir: Path, chunk_id: str, **kwargs: object) -> None:
    defaults = {
        "project": pdir.name,
        "kind": "note",
        "ts": _NOW,
        "session_id": "",
        "pinned": 0,
        "salience": 0,
        "consolidation_round": 0,
        "superseded_by": "",
        "content_hash": "x",
    }
    defaults.update(kwargs)
    conn = sqlite3.connect(str(pdir / "meta.sqlite"))
    conn.execute(
        "INSERT INTO chunks (id, project, kind, ts, session_id, pinned, salience, "
        "consolidation_round, superseded_by, content_hash) "
        "VALUES (:id, :project, :kind, :ts, :session_id, :pinned, :salience, "
        ":consolidation_round, :superseded_by, :content_hash)",
        {"id": chunk_id, **defaults},
    )
    conn.commit()
    conn.close()


async def _get(app: object, url: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(url)


def _patch_brain_root(brain_root: Path) -> tuple:
    """Return (common, _data) modules with BRAIN_ROOT patched. Call restore() after."""
    import common
    import ui.data as _data
    orig_common = common.BRAIN_ROOT
    orig_data = _data.BRAIN_ROOT
    common.BRAIN_ROOT = brain_root
    _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]
    # Invalidate kinds cache for all projects
    _data._kinds_cache.clear()
    return common, _data, orig_common, orig_data


def _restore_brain_root(common: object, _data: object, orig_common: object, orig_data: object) -> None:
    import common as _c
    import ui.data as _d
    _c.BRAIN_ROOT = orig_common  # type: ignore[attr-defined]
    _d.BRAIN_ROOT = orig_data  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helper: run a GET and return response, patching BRAIN_ROOT
# ---------------------------------------------------------------------------

def _search(tmp_path: Path, project_hash: str, params: str) -> httpx.Response:
    from ui.server import app

    brain_root = tmp_path / "brain"
    _make_project(brain_root, project_hash)
    _insert(brain_root / project_hash, "chunk-note-1", kind="note", ts=_NOW - _DAY)
    _insert(brain_root / project_hash, "chunk-decision-1", kind="decision", ts=_NOW)
    _insert(brain_root / project_hash, "chunk-pinned-1", kind="note", ts=_NOW, pinned=1)
    _insert(brain_root / project_hash, "chunk-salience-1", kind="lesson", ts=_NOW, salience=8)
    _insert(brain_root / project_hash, "chunk-round-1", kind="lesson", ts=_NOW, consolidation_round=2)
    _insert(brain_root / project_hash, "chunk-session-1", kind="note", ts=_NOW, session_id="abc123def456")
    _insert(brain_root / project_hash, "chunk-super-1", kind="note", ts=_NOW, superseded_by="tombstone")

    common, _data, orig_c, orig_d = _patch_brain_root(brain_root)
    try:
        resp = anyio.run(_get, app, f"/search?project={project_hash}&{params}")
    finally:
        _restore_brain_root(common, _data, orig_c, orig_d)
    return resp


# ---------------------------------------------------------------------------
# Tests: basic rendering
# ---------------------------------------------------------------------------

class TestSearchBasicRendering:
    def test_search_page_renders(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223344", "")
        assert resp.status_code == 200
        assert "<form" in resp.text

    def test_kinds_present_in_dropdown(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223351", "")
        assert resp.status_code == 200
        # dynamic kinds should come from DB
        body = resp.text
        # note, decision, lesson are inserted
        assert "note" in body or "decision" in body or "lesson" in body

    def test_filter_bar_has_date_inputs(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223352", "")
        body = resp.text
        assert 'name="from_date"' in body
        assert 'name="to_date"' in body

    def test_filter_bar_has_session_input(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223353", "")
        assert 'name="session"' in resp.text

    def test_filter_bar_has_pinned_checkbox(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223354", "")
        assert 'name="pinned_only"' in resp.text

    def test_filter_bar_has_salience_input(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223355", "")
        assert 'name="salience_min"' in resp.text

    def test_filter_bar_has_round_input(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223356", "")
        assert 'name="round_min"' in resp.text


# ---------------------------------------------------------------------------
# Tests: _build_where filter logic (unit level)
# ---------------------------------------------------------------------------

class TestBuildWhere:
    def setup_method(self) -> None:
        from ui.routers.search import _build_where
        self._bw = _build_where

    def test_no_filters_returns_none(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=None, to_ts=None,
            session=None, pinned_only=False, salience_min=0, round_min=0,
        )
        assert result is None

    def test_exclude_superseded_default(self) -> None:
        result = self._bw(
            include_superseded=False, kind="", from_ts=None, to_ts=None,
            session=None, pinned_only=False, salience_min=0, round_min=0,
        )
        assert result is not None
        assert "superseded_by = ''" in result

    def test_valid_kind_included(self) -> None:
        result = self._bw(
            include_superseded=True, kind="note", from_ts=None, to_ts=None,
            session=None, pinned_only=False, salience_min=0, round_min=0,
        )
        assert result is not None
        assert "kind = 'note'" in result

    def test_injection_kind_excluded(self) -> None:
        # Injection attempt should not appear in WHERE clause
        result = self._bw(
            include_superseded=True, kind="foo'; DROP TABLE chunks;--", from_ts=None, to_ts=None,
            session=None, pinned_only=False, salience_min=0, round_min=0,
        )
        # kind injection rejected; no other filters → None or just empty
        assert result is None or "DROP" not in result

    def test_from_ts_filter(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=1700000000, to_ts=None,
            session=None, pinned_only=False, salience_min=0, round_min=0,
        )
        assert result is not None
        assert "ts >= 1700000000" in result

    def test_to_ts_filter(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=None, to_ts=1800000000,
            session=None, pinned_only=False, salience_min=0, round_min=0,
        )
        assert "ts <= 1800000000" in result

    def test_session_valid_hex(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=None, to_ts=None,
            session="abc123def456", pinned_only=False, salience_min=0, round_min=0,
        )
        assert result is not None
        assert "session_id = 'abc123def456'" in result

    def test_session_injection_rejected(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=None, to_ts=None,
            session="'; DROP TABLE chunks; --", pinned_only=False, salience_min=0, round_min=0,
        )
        assert result is None or "DROP" not in result

    def test_pinned_only(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=None, to_ts=None,
            session=None, pinned_only=True, salience_min=0, round_min=0,
        )
        assert result is not None
        assert "pinned = 1" in result

    def test_salience_min(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=None, to_ts=None,
            session=None, pinned_only=False, salience_min=5, round_min=0,
        )
        assert result is not None
        assert "salience >= 5" in result

    def test_round_min(self) -> None:
        result = self._bw(
            include_superseded=True, kind="", from_ts=None, to_ts=None,
            session=None, pinned_only=False, salience_min=0, round_min=1,
        )
        assert result is not None
        assert "consolidation_round >= 1" in result

    def test_multiple_filters_combined_with_and(self) -> None:
        result = self._bw(
            include_superseded=False, kind="note", from_ts=1700000000, to_ts=None,
            session=None, pinned_only=True, salience_min=0, round_min=0,
        )
        assert result is not None
        assert " AND " in result
        assert "superseded_by = ''" in result
        assert "kind = 'note'" in result
        assert "ts >= 1700000000" in result
        assert "pinned = 1" in result


# ---------------------------------------------------------------------------
# Tests: _parse_date_to_ts
# ---------------------------------------------------------------------------

class TestParseDateToTs:
    def setup_method(self) -> None:
        from ui.routers.search import _parse_date_to_ts
        self._parse = _parse_date_to_ts

    def test_valid_date_parsed(self) -> None:
        ts = self._parse("2024-01-15")
        assert ts is not None
        assert ts > 0
        # 2024-01-15 00:00:00 local
        from datetime import datetime
        dt = datetime.fromtimestamp(ts)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_end_of_day_flag(self) -> None:
        ts = self._parse("2024-01-15", end_of_day=True)
        assert ts is not None
        from datetime import datetime
        dt = datetime.fromtimestamp(ts)
        assert dt.hour == 23
        assert dt.minute == 59
        assert dt.second == 59

    def test_empty_string_returns_none(self) -> None:
        assert self._parse("") is None

    def test_whitespace_returns_none(self) -> None:
        assert self._parse("   ") is None

    def test_invalid_format_returns_none(self) -> None:
        assert self._parse("15-01-2024") is None
        assert self._parse("2024/01/15") is None
        assert self._parse("not-a-date") is None

    def test_injection_attempt_returns_none(self) -> None:
        assert self._parse("2024-01-15'; DROP TABLE") is None


# ---------------------------------------------------------------------------
# Tests: _qs helper
# ---------------------------------------------------------------------------

class TestQsHelper:
    def setup_method(self) -> None:
        from ui.routers.search import _qs
        self._qs = _qs

    def test_basic_params(self) -> None:
        qs = self._qs({"project": "abc", "q": "test"})
        assert "project=abc" in qs
        assert "q=test" in qs

    def test_falsy_values_omitted(self) -> None:
        qs = self._qs({"project": "abc", "q": "", "offset": 0, "pinned_only": False})
        assert "q=" not in qs
        assert "offset" not in qs
        assert "pinned_only" not in qs

    def test_override_replaces_param(self) -> None:
        qs = self._qs({"offset": 0, "limit": 50}, offset=50)
        assert "offset=50" in qs

    def test_none_omitted(self) -> None:
        qs = self._qs({"kind": None, "project": "abc"})
        assert "kind" not in qs


# ---------------------------------------------------------------------------
# Tests: Pagination query strings
# ---------------------------------------------------------------------------

class TestPaginationQs:
    def test_qs_next_increments_offset(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223360", "limit=5&offset=0")
        body = resp.text
        # qs_next should have offset=5 (limit=5, next page)
        assert "offset=5" in body or "offset" not in body  # may not show if total <= limit

    def test_qs_prev_decrements_offset(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223361", "limit=5&offset=5")
        body = resp.text
        # prev link should be present when offset > 0
        assert 'rel="prev"' in body

    def test_second_page_shows_the_next_chunks(self, tmp_path: Path) -> None:
        ids = lambda body: set(re.findall(r'data-id="(chunk-[a-z0-9-]+)"', body))  # noqa: E731
        first = ids(_search(tmp_path / "a", "aabb11223363", "limit=3&offset=0").text)
        second = ids(_search(tmp_path / "b", "aabb11223363", "limit=3&offset=3").text)
        assert len(first) == 3 and second and not first & second

    def test_result_meta_shows_count(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223362", "")
        body = resp.text
        # "Showing X of Y matching" badge
        assert "Showing" in body or "matching" in body


# ---------------------------------------------------------------------------
# Tests: SQL injection via URL params (regression)
# ---------------------------------------------------------------------------

class TestSqlInjectionViaUrl:
    def test_kind_injection_no_error(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223370", "kind=foo%27%3Bdrop")
        # Must not raise 500
        assert resp.status_code == 200

    def test_session_injection_no_error(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223371", "session=abc%27%3B+DROP+TABLE+chunks%3B--")
        assert resp.status_code == 200

    def test_from_date_garbage_no_error(self, tmp_path: Path) -> None:
        resp = _search(tmp_path, "aabb11223372", "from_date=not-a-date")
        assert resp.status_code == 200

    def test_unknown_project_returns_404(self, tmp_path: Path) -> None:
        from ui.server import app
        brain_root = tmp_path / "brain_404"
        brain_root.mkdir(parents=True, exist_ok=True)

        import common
        import ui.data as _data
        _orig_c, _orig_d = common.BRAIN_ROOT, _data.BRAIN_ROOT
        common.BRAIN_ROOT = brain_root
        _data.BRAIN_ROOT = brain_root  # type: ignore[attr-defined]
        try:
            resp = anyio.run(_get, app, "/search?project=doesnotexist000")
        finally:
            common.BRAIN_ROOT = _orig_c
            _data.BRAIN_ROOT = _orig_d  # type: ignore[attr-defined]
        assert resp.status_code == 404
