"""Compaction restore: a raw-tail digest now, the distilled summary when it lands.

PreCompact no longer waits for the distill. capture.py writes a digest of the
conversation tail (`sessions/<sid>.compact.json`) and returns; the pre_compact
segment is distilled in the background.

SessionStart(compact) restores this session's pre_compact summary if it has
landed, and otherwise the digest. While only the digest was restored, the
first UserPromptSubmit after the summary lands injects it once.

"Landed" means the PreCompact event's queue file is gone: flush deletes a
segment's events only after all of its chunks are stored, so a restore never
sees half a summary. A summary can be several chunks (windows, trust groups);
they share the segment's valid_from.

The digest holds only `[user]` and `[claude]` segments: no file, tool or web
text. It is stripped of `<private>`, defused, scrubbed, and escaped again when
framed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from inject_state import SESSION_RE

DIGEST_TTL_S = 86400
DIGEST_READ_BYTES = 262_144
SUMMARY_MAX_CHUNKS = 6


def _path(pdir: Path, sid: str) -> Path | None:
    if not SESSION_RE.match(sid or ""):
        return None
    return pdir / "sessions" / f"{sid}.compact.json"


def write_digest(pdir: Path, sid: str, event: str, path: str, offset: int, last_message: str, ts: int,
                 cfg: dict) -> None:
    """Digest the transcript tail up to *offset* (or *last_message* without one).

    *event* is the PreCompact event's queue file name; see landed().
    """
    p = _path(pdir, sid)
    if p is None:
        return
    import segment
    import transcript
    from scrub import scrub

    segs: list[transcript.Segment] = []
    if path and offset:
        try:
            raw, _ = segment.read_range(path, max(0, offset - DIGEST_READ_BYTES), offset, DIGEST_READ_BYTES)
            segs = [s for s in transcript.segments(raw) if s.source in ("user", "assistant")]
        except OSError:
            segs = []
    if not segs and last_message.strip():
        segs = [transcript.Segment("assistant", transcript.defuse(last_message))]
    text = transcript.render(segs, int(cfg["capture"].get("compact_digest_chars", 6000)))
    if not text.strip():
        return
    if cfg["capture"].get("scrub_secrets", True):
        text = scrub(text)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"ts": ts, "text": text, "event": event}))
    os.chmod(tmp, 0o600)
    tmp.replace(p)


def read_digest(pdir: Path, sid: str, now: float | None = None) -> dict | None:
    """The session's digest, or None (none, unreadable, or older than a day: removed)."""
    p = _path(pdir, sid)
    if p is None or not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
        if (now or time.time()) - int(d["ts"]) <= DIGEST_TTL_S and isinstance(d.get("text"), str):
            return d
    except (OSError, ValueError, KeyError, TypeError):
        pass
    drop_digest(pdir, sid)
    return None


def landed(pdir: Path, d: dict) -> bool:
    """True once flush has stored (or dropped) the segment the digest belongs to."""
    ev = d.get("event")
    if not isinstance(ev, str) or not ev or Path(ev).name != ev:
        return True
    return not (pdir / "queue" / ev).exists()


def drop_digest(pdir: Path, sid: str) -> None:
    p = _path(pdir, sid)
    if p is not None:
        p.unlink(missing_ok=True)


def summary_rows(meta, proj: str, sid: str, since: int | None, injectable_sql: str) -> list[tuple[str, int]]:
    """[(id, valid_from)] of this session's newest pre_compact summary from *since* on.

    The chunks of one segment share its valid_from; only the newest one is returned.
    """
    import temporal

    vf = temporal.VALID_FROM_SQL
    sql = (f"SELECT id, {vf} FROM chunks WHERE project = ? AND kind = 'pre_compact' "
           f"AND session_id = ? AND superseded_by = '' AND {injectable_sql}")
    args: list = [proj, sid]
    if since is not None:
        sql += f" AND {vf} >= ?"
        args.append(since)
    rows = meta.execute(sql + f" ORDER BY {vf} DESC, ts ASC LIMIT ?", [*args, SUMMARY_MAX_CHUNKS * 4]).fetchall()
    return [(r[0], r[1]) for r in rows if r[1] == rows[0][1]][:SUMMARY_MAX_CHUNKS]
