"""Validated LanceDB `where` clauses.

LanceDB filters are SQL strings with no bound parameters, so every value that
reaches one goes through this module. Values are checked against the shape of
their column (ids and project keys are short word tokens, kinds come from a
fixed set, timestamps are ints), so a quote or an `OR 1=1` can never get in:
anything that does not match raises FilterError instead of being escaped.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

KINDS = frozenset({
    "stop", "subagent_stop", "pre_compact",           # captured events
    "decision", "note", "code", "task",               # explicit /remember
    "arc", "daily_summary", "lesson", "schema",       # consolidation products
    "hypothesis", "core",
    "quarantine",                                     # low-trust imperatives
})

# Production ids are uuid4 hex and project keys 12 hex chars; the check is a
# safe charset rather than the exact shape so fixtures can use short ids.
_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class FilterError(ValueError):
    """A value does not have the shape its column requires."""


def _id(v: str) -> str:
    if not isinstance(v, str) or not _KEY.match(v):
        raise FilterError(f"bad chunk id: {v!r:.40}")
    return v


def _superseded(v: str) -> str:
    return v if v == "" else _id(v)


def _project(v: str) -> str:
    if not isinstance(v, str) or not _KEY.match(v):
        raise FilterError(f"bad project key: {v!r:.40}")
    return v


def _kind(v: str) -> str:
    if v not in KINDS:
        raise FilterError(f"unknown kind: {v!r:.40}")
    return v


def _token(v: str) -> str:
    if not isinstance(v, str) or not _TOKEN.match(v):
        raise FilterError(f"bad token: {v!r:.40}")
    return v


_STRING_COLS = {
    "id": _id,
    "superseded_by": _superseded,
    "project": _project,
    "kind": _kind,
    "session_id": _token,
    "agent_id": _token,
    "content_hash": _token,
}
_INT_COLS = {"ts"}


def _lit(col: str, v: object) -> str:
    if col in _INT_COLS:
        if isinstance(v, bool) or not isinstance(v, int):
            raise FilterError(f"{col} needs an int, got {type(v).__name__}")
        return str(v)
    check = _STRING_COLS.get(col)
    if check is None:
        raise FilterError(f"unknown column: {col!r}")
    return f"'{check(v)}'"


def eq(col: str, v: object) -> str:
    return f"{col} = {_lit(col, v)}"


def ne(col: str, v: object) -> str:
    return f"{col} != {_lit(col, v)}"


def ge(col: str, v: int) -> str:
    if col not in _INT_COLS:
        raise FilterError(f"{col} is not numeric")
    return f"{col} >= {_lit(col, v)}"


def isin(col: str, vs: Iterable[object]) -> str:
    vals = [_lit(col, v) for v in vs]
    if not vals:
        raise FilterError(f"empty IN list for {col}")
    return f"{col} IN ({', '.join(vals)})"


def notin(col: str, vs: Iterable[object]) -> str:
    return "NOT " + isin(col, vs)


def and_(*clauses: str | None) -> str:
    """Join non-empty clauses with AND."""
    return " AND ".join(c for c in clauses if c)


LIVE = "superseded_by = ''"
NOT_HYPOTHESIS = "kind != 'hypothesis'"
NOT_QUARANTINE = "kind != 'quarantine'"


def live_in(proj: str, *extra: str | None) -> str:
    """Rows of `proj` that are not superseded, plus any extra clauses."""
    return and_(eq("project", proj), LIVE, *extra)


def in_project(proj: str, *extra: str | None) -> str:
    """All rows of `proj`, superseded ones included (temporal queries)."""
    return and_(eq("project", proj), *extra)
