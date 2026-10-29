"""Temporal validity (bi-temporal, Zep-style): when a memory was true.

Two times per chunk:
  ts          record time — when the brain stored it (unchanged)
  valid_from  event time — when it became true; NULL means "same as ts"
  invalid_at  when a later memory contradicted it; NULL while still valid

Contradiction never deletes: the judge sets old.invalid_at to new.valid_from
and links old to new through superseded_by. A superseded_by link that names
no chunk is a tombstone (forget / UI delete); such a row is gone for every
query, historical ones included.

Read side:
  now (default)         live rows only — same as v0.6 (superseded_by = '')
  as_of=T               rows valid at T: valid_from <= T < invalid_at
  include_history=True  each result carries the chain of memories it replaced
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

VALID_FROM_SQL = "COALESCE(valid_from, ts)"
_MAX_CHAIN = 20


@dataclass(frozen=True)
class Validity:
    valid_from: int
    invalid_at: int | None
    superseded_by: str
    # superseded_by names an existing chunk (a contradiction, not a tombstone)
    has_successor: bool

    @property
    def live(self) -> bool:
        return not self.superseded_by

    @property
    def forgotten(self) -> bool:
        return bool(self.superseded_by) and not (self.has_successor and self.invalid_at is not None)

    def valid_at(self, t: int) -> bool:
        if self.forgotten or self.valid_from > t:
            return False
        return self.invalid_at is None or t < self.invalid_at


def from_lance(row: dict) -> Validity:
    """Fallback for a row with no meta row: record time is event time."""
    sup = row.get("superseded_by") or ""
    return Validity(int(row.get("ts") or 0), None, sup, False)


def validity(meta: Any, proj: str, ids: list[str]) -> dict[str, Validity]:
    """Validity per id from the project's meta db; missing ids are absent."""
    if meta is None or not ids:
        return {}
    marks = ",".join("?" * len(ids))
    try:
        rows = meta.execute(
            f"SELECT c.id, {VALID_FROM_SQL}, c.invalid_at, c.superseded_by, "
            "EXISTS (SELECT 1 FROM chunks s WHERE s.id = c.superseded_by AND s.project = c.project) "
            f"FROM chunks c WHERE c.project = ? AND c.id IN ({marks})",
            (proj, *ids),
        ).fetchall()
    except Exception:  # noqa: BLE001 — pre-v0.7 meta (read-only) without the columns
        return {}
    return {r[0]: Validity(int(r[1] or 0), r[2], r[3] or "", bool(r[4])) for r in rows}


def invalidate(meta: Any, proj: str, old_id: str, new_id: str) -> int | None:
    """Record that *new_id* contradicts *old_id*. Returns old.invalid_at.

    invalid_at is the new memory's event time, never earlier than the old
    one's (so an interval is never negative). Caller commits.
    """
    row = meta.execute(
        f"SELECT (SELECT {VALID_FROM_SQL} FROM chunks WHERE id = ? AND project = ?), "
        f"{VALID_FROM_SQL} FROM chunks WHERE id = ? AND project = ?",
        (new_id, proj, old_id, proj),
    ).fetchone()
    if row is None:
        return None
    new_vf, old_vf = row
    at = max(int(new_vf if new_vf is not None else time.time()), int(old_vf or 0))
    meta.execute(
        "UPDATE chunks SET superseded_by = ?, invalid_at = ? WHERE id = ? AND project = ?",
        (new_id, at, old_id, proj),
    )
    return at


def head_of(meta: Any, proj: str, cid: str) -> str | None:
    """The live memory at the end of cid's contradiction chain, or None."""
    seen = set()
    cur = cid
    while len(seen) < _MAX_CHAIN:
        seen.add(cur)
        row = meta.execute(
            "SELECT superseded_by, invalid_at FROM chunks WHERE id = ? AND project = ?", (cur, proj)
        ).fetchone()
        if row is None:
            return None
        sup, inv = row
        if not sup:
            return cur
        if inv is None or sup in seen:
            return None  # tombstone or a cycle
        cur = sup
    return None


def predecessors(meta: Any, proj: str, cid: str) -> list[tuple[str, int, int]]:
    """(id, valid_from, invalid_at) of the memories cid replaced, newest first."""
    out: list[tuple[str, int, int]] = []
    seen = {cid}
    frontier = [cid]
    while frontier and len(out) < _MAX_CHAIN:
        marks = ",".join("?" * len(frontier))
        rows = meta.execute(
            f"SELECT id, {VALID_FROM_SQL}, invalid_at FROM chunks "
            f"WHERE project = ? AND invalid_at IS NOT NULL AND superseded_by IN ({marks})",
            (proj, *frontier),
        ).fetchall()
        frontier = [r[0] for r in rows if r[0] not in seen]
        seen.update(frontier)
        out.extend((r[0], int(r[1]), int(r[2])) for r in rows if r[0] in frontier)
    return sorted(out, key=lambda r: r[2], reverse=True)[:_MAX_CHAIN]


# ── parsing and display ──────────────────────────────────────────────────

_REL = re.compile(r"^(\d+)([dhw])$")


def parse_as_of(s: str | int | None, now: float | None = None) -> int | None:
    """None/'now' → None; unix ts; '30d'/'12h'/'2w' ago; 'YYYY-MM-DD' (end of
    that local day); or an ISO datetime. Raises ValueError otherwise."""
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = s.strip()
    if s in ("", "now"):
        return None
    now = time.time() if now is None else now
    if s.isdigit():
        return int(s)
    m = _REL.match(s)
    if m:
        return int(now) - int(m.group(1)) * {"h": 3600, "d": 86400, "w": 7 * 86400}[m.group(2)]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return int(datetime.strptime(s, "%Y-%m-%d").timestamp()) + 86399
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except ValueError:
        raise ValueError(f"unrecognised as_of {s!r}; use 'now', a unix ts, 30d, or YYYY-MM-DD") from None


def iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    try:
        return time.strftime("%Y-%m-%d", time.localtime(int(ts)))
    except (TypeError, ValueError, OverflowError):
        return None
