"""Tests for safe_kind() (F-003) and dynamic kinds list (F-002).

Verifies SQL injection attempts are rejected, valid kinds pass through,
and the dynamic kinds list comes from the DB not a hardcoded set.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (_SCRIPTS, _ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from ui.data import fetch_kinds_present, safe_kind


# ---------------------------------------------------------------------------
# safe_kind tests
# ---------------------------------------------------------------------------

class TestSafeKindValidInputs:
    @pytest.mark.parametrize("kind", [
        "note",
        "decision",
        "stop",
        "subagent_stop",
        "pre_compact",
        "lesson",
        "hypothesis",
        "schema",
        "my-kind",
        "kind123",
        "UPPER",
        "a" * 32,  # exactly 32 chars — allowed
    ])
    def test_valid_kind_returned(self, kind: str) -> None:
        assert safe_kind(kind) == kind

    def test_empty_string_returns_none(self) -> None:
        assert safe_kind("") is None


class TestSafeKindRejectedInputs:
    @pytest.mark.parametrize("malicious", [
        "foo'; DROP TABLE chunks; --",  # classic injection
        "foo'",                          # single quote
        'foo"',                          # double quote
        "foo;bar",                       # semicolon
        "foo--bar",                      # SQL comment
        "foo bar",                       # whitespace
        "foo\tbar",                      # tab
        "foo\nbar",                      # newline
        "foo\x00bar",                    # null byte
        "a" * 33,                        # too long (33 chars)
        "foo (bar)",                     # parentheses
        "foo=bar",                       # equals
        "foo%20bar",                     # percent-encoded space (raw % char)
        "kind OR 1=1",                   # OR injection
        "note UNION SELECT",             # UNION injection
        "'; SELECT * FROM chunks--",     # full injection attempt
    ])
    def test_injection_rejected(self, malicious: str) -> None:
        result = safe_kind(malicious)
        assert result is None, f"Expected None for {malicious!r}, got {result!r}"

    def test_33_char_rejected(self) -> None:
        assert safe_kind("a" * 33) is None

    def test_32_char_accepted(self) -> None:
        assert safe_kind("a" * 32) == "a" * 32


# ---------------------------------------------------------------------------
# fetch_kinds_present tests
# ---------------------------------------------------------------------------

def _make_test_db(tmp_path: Path, kinds: list[tuple[str, str]]) -> sqlite3.Connection:
    """Create an in-memory-style SQLite DB with the given (id, kind, superseded_by) rows."""
    db_path = tmp_path / "meta.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT 'note',
            superseded_by TEXT NOT NULL DEFAULT '',
            ts INTEGER NOT NULL DEFAULT 0
        )
    """)
    for chunk_id, kind in kinds:
        conn.execute(
            "INSERT INTO chunks (id, kind) VALUES (?, ?)",
            (chunk_id, kind),
        )
    conn.commit()
    return conn


class TestFetchKindsPresent:
    def test_returns_kinds_from_db(self, tmp_path: Path) -> None:
        conn = _make_test_db(tmp_path, [
            ("id1", "note"),
            ("id2", "decision"),
            ("id3", "note"),  # duplicate kind — should be DISTINCT
        ])
        kinds = fetch_kinds_present(conn, "test_project_abc")
        conn.close()
        assert "note" in kinds
        assert "decision" in kinds
        assert kinds.count("note") == 1  # DISTINCT

    def test_excludes_superseded_kinds(self, tmp_path: Path) -> None:
        db_path = tmp_path / "meta2.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'note',
                superseded_by TEXT NOT NULL DEFAULT '',
                ts INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("INSERT INTO chunks (id, kind, superseded_by) VALUES ('a', 'note', '')")
        conn.execute("INSERT INTO chunks (id, kind, superseded_by) VALUES ('b', 'ghost_kind', 'tombstone123')")
        conn.commit()
        kinds = fetch_kinds_present(conn, "test_project_def")
        conn.close()
        assert "note" in kinds
        assert "ghost_kind" not in kinds

    def test_uses_cache_on_second_call(self, tmp_path: Path) -> None:
        conn = _make_test_db(tmp_path, [("id1", "note")])
        # Invalidate any existing cache for this project
        from ui.data import _kinds_cache
        _kinds_cache.pop("test_project_cache", None)

        kinds1 = fetch_kinds_present(conn, "test_project_cache")
        # Insert a new kind — cache should prevent it from appearing immediately
        conn.execute("INSERT INTO chunks (id, kind) VALUES ('id_new', 'lessonXYZ')")
        conn.commit()
        kinds2 = fetch_kinds_present(conn, "test_project_cache")
        conn.close()

        # Second call should be cached (lessonXYZ not yet visible)
        assert "lessonXYZ" not in kinds2

    def test_empty_table_returns_empty_list(self, tmp_path: Path) -> None:
        conn = _make_test_db(tmp_path, [])
        kinds = fetch_kinds_present(conn, "test_project_empty_zzz")
        conn.close()
        assert kinds == []

    def test_not_hardcoded_set(self, tmp_path: Path) -> None:
        """Dynamic kinds must come from DB, not a hardcoded VALID_KINDS constant."""
        # Insert a kind that was NOT in the old hardcoded VALID_KINDS set
        conn = _make_test_db(tmp_path, [("x1", "custom_kind_xyz")])
        kinds = fetch_kinds_present(conn, "test_project_custom_xyz")
        conn.close()
        assert "custom_kind_xyz" in kinds

    def test_hardcoded_kinds_no_longer_authoritative(self) -> None:
        """Verify the old VALID_KINDS set is gone from search router."""
        import ui.routers.search as search_mod
        assert not hasattr(search_mod, "VALID_KINDS"), (
            "VALID_KINDS should be removed from search.py (replaced by fetch_kinds_present)"
        )

    def test_sorted_alphabetically(self, tmp_path: Path) -> None:
        conn = _make_test_db(tmp_path, [
            ("a", "zebra"),
            ("b", "apple"),
            ("c", "mango"),
        ])
        kinds = fetch_kinds_present(conn, "test_project_sort")
        conn.close()
        assert kinds == sorted(kinds)
