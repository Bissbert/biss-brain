"""Session segments: queued Stop events → one distill per segment.

A Stop hook only records where the transcript ended (a byte offset); it does
not parse anything. The flush daemon groups queued events by transcript and
distills each group once, when the segment closes:

  - SessionEnd or PreCompact (a boundary event),
  - no new event for `segment_idle_seconds` (default 10 min),
  - `segment_max_events` events queued (default 20).

The transcript is then read from its cursor (`BRAIN_ROOT/cursors/`, keyed by
transcript path, so a `cd` mid-session does not re-read it), tagged, scrubbed
and packed into windows of whole JSONL lines (plan()). A pass ends at the
first boundary event, so post-compaction turns never join a pre_compact
summary, and covers at most `raw_tail_bytes` and `segment_max_windows`
windows; the rest stays queued for the next pass. Nothing is skipped.

flush stores one window at a time and then moves the cursor past it, so a
failed distill re-reads only from the window that failed. An event is
finished, and its queue file deleted, once the cursor has passed its offset.

Events without an offset (pre-5.4 queue files) keep their inline excerpt and
the old per-kind debounce.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass

import common
import transcript
from common import Event, now_unix
from scrub import scrub

log = logging.getLogger("biss-brain.segment")

BOUNDARY = ("session_end", "pre_compact")
CURSOR_TTL_S = 30 * 86400
# Ollama distills with num_ctx 16384 and silently drops input past it.
OLLAMA_WINDOW_CHARS = 24000


def _own_subagent(ev: Event) -> bool:
    """A SubagentStop that points at the subagent's own transcript."""
    return ev.kind == "subagent_stop" and bool(ev.payload.get("own_transcript"))


def group_key(ev: Event) -> str:
    if ev.offset is None:
        return f"legacy:{ev.kind}"
    return f"seg:{ev.transcript_path or ev.session_id}"


def kind(events: list[tuple[Path, Event]]) -> str:
    """Distill and storage kind for a segment."""
    kinds = {e.kind for _, e in events}
    if "pre_compact" in kinds:
        return "pre_compact"
    if all(_own_subagent(e) for _, e in events):
        return "subagent_stop"
    return "stop"


def ready(events: list[tuple[Path, Event]], cfg: dict, now: int | None = None) -> bool:
    if not events:
        return False
    if any(e.payload.get("__immediate") or e.kind in BOUNDARY for _, e in events):
        return True
    deb = cfg["debounce"]
    age = (now or now_unix()) - max(e.enqueued_ts for _, e in events)
    k = events[0][1].kind
    if events[0][1].offset is None:  # legacy: per-kind debounce
        wait = deb["subagent_stop_seconds"] if k == "subagent_stop" else deb["stop_seconds"]
        return age >= wait or len(events) >= deb["max_events_per_batch"]
    if all(_own_subagent(e) for _, e in events):
        return age >= deb["subagent_stop_seconds"]
    return age >= deb["segment_idle_seconds"] or len(events) >= deb["segment_max_events"]


# ── cursors ──────────────────────────────────────────────────────────────

def _cursor_file(path: str) -> Path:
    return common.BRAIN_ROOT / "cursors" / (hashlib.sha256(path.encode()).hexdigest()[:16] + ".json")


def cursor(path: str) -> int:
    try:
        return int(json.loads(_cursor_file(path).read_text())["offset"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def set_cursor(path: str, offset: int) -> None:
    f = _cursor_file(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"path": path, "offset": offset}))
    tmp.replace(f)
    now = time.time()
    for p in f.parent.glob("*.json"):
        try:
            if now - p.stat().st_mtime > CURSOR_TTL_S:
                p.unlink()
        except OSError:
            pass


def read_range(path: str, start: int, end: int, max_bytes: int) -> tuple[str, int]:
    """Complete JSONL lines in [start, end) and the offset just past the last one.

    Over *max_bytes*, the oldest bytes are skipped (from a line boundary). A
    file shorter than *start* (rewritten) is read from 0; an *end* at or
    before *start* (an event already distilled) reads nothing.
    """
    size = os.path.getsize(path)
    if start > size:
        start = 0
    end = min(end, size)
    if end <= start:
        return "", start
    skip = end - start > max_bytes
    if skip:
        start = end - max_bytes
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    if skip:
        nl = data.find(b"\n")
        data = data[nl + 1:] if nl >= 0 else b""
        start = end - len(data)
    last = data.rfind(b"\n")
    if last < 0:
        return "", start
    return data[: last + 1].decode("utf-8", errors="replace"), start + last + 1


# ── segment text ─────────────────────────────────────────────────────────

def read_lines(path: str, start: int, end: int, max_bytes: int) -> tuple[list[tuple[int, str]], int, bool]:
    """Complete JSONL lines in [start, end), oldest first, the offset past the last one,
    and whether the read reached *end* (or the end of a shorter file).

    At most *max_bytes* are read, but always at least one whole line. Each
    line comes with the byte offset just past it. A file shorter than
    *start* (rewritten) is read from 0.
    """
    size = os.path.getsize(path)
    if start > size:
        start = 0
    end = min(end, size)
    if end <= start:
        return [], start, True
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(min(end - start, max_bytes))
        while b"\n" not in data and start + len(data) < end:
            more = f.read(min(end - start - len(data), 1 << 20))
            if not more:
                break
            data += more
    if len(data) > max_bytes and (nl := data.find(b"\n")) >= 0:
        data = data[: nl + 1]  # one line over the cap: that line only
    reached = start + len(data) >= end
    out: list[tuple[int, str]] = []
    pos = 0
    while (nl := data.find(b"\n", pos)) >= 0:
        out.append((start + nl + 1, data[pos:nl].decode("utf-8", errors="replace")))
        pos = nl + 1
    return out, start + pos, reached


def width(cfg: dict) -> int:
    """Window size in chars: Ollama's smaller context if it may distill.

    Windows are planned before the distill call, so an Ollama fallback sizes
    them too (and Claude then gets about twice as many, smaller calls).
    """
    w = int(cfg["capture"]["segment_window_chars"])
    may = [cfg["distill"]["backend"], *(cfg["distill"].get("fallback") or [])]
    return min(w, OLLAMA_WINDOW_CHARS) if "ollama" in may else w


@dataclass
class Window:
    text: str
    present: list[str]  # raw sources in the text
    end: int | None     # cursor after this window; None without a transcript


@dataclass
class Pass:
    """One distill pass over a segment: its events, windows and transcript."""

    events: list[tuple[Path, Event]]  # the events this pass may finish
    windows: list[Window]
    path: str | None    # transcript whose cursor moves; None: no cursor
    end: int | None     # cursor once every window is stored
    kind: str = "stop"
    start: int = 0      # cursor before the pass
    complete: bool = True  # the pass covers everything readable up to its events

    def commit(self, offset: int | None, final: bool = False) -> list[tuple[Path, Event]]:
        """Move the cursor to *offset*; the events now finished.

        Without a cursor, and on the *final* commit of a complete pass, that
        is every event: an offset past the last whole line (a line still
        being written, or a transcript rewritten shorter) is never reached.
        """
        if self.path is None or offset is None:
            return list(self.events)
        set_cursor(self.path, offset)
        if final and self.complete:
            return list(self.events)
        return [(f, e) for f, e in self.events if (e.offset or 0) <= offset]


def plan(events: list[tuple[Path, Event]], cfg: dict) -> Pass:
    """The next distill pass for a ready segment (see the module docstring)."""
    cap = cfg["capture"]
    scrub_on = cap.get("scrub_secrets", True)
    evs = [e for _, e in events]
    if evs[0].offset is None:
        # Pre-v0.7 events (sources=None) carry untagged raw JSONL: treat as tool output.
        joined = "\n\n---\n\n".join(e.transcript_excerpt for e in evs if e.transcript_excerpt)
        present = sorted({s for e in evs for s in (e.sources if e.sources is not None else ["tool"])})
        wins = [Window(joined, present, None)] if joined.strip() else []
        return Pass(list(events), wins, None, None, kind(events))

    # The pass ends at the first boundary: later events start the next segment.
    # Events of the same second sort by kind, so membership goes by offset.
    upto = next((i for i, e in enumerate(evs) if e.kind in BOUNDARY), len(evs) - 1)
    limit = max(e.offset or 0 for e in evs[: upto + 1])
    in_pass = [(f, e) for f, e in events if (e.offset or 0) <= limit]
    k = kind(in_pass)
    path = evs[upto].transcript_path
    lines = None
    if path and os.path.isfile(path):
        try:
            start = cursor(path)
            lines, read_end, reached = read_lines(path, start, limit, int(cap.get("raw_tail_bytes", 2_000_000)))
        except OSError as e:
            log.warning("segment: cannot read transcript (%s); using last messages", e)
    if lines is None:
        # Transcript gone or unreadable: the last assistant messages, scrubbed at capture.
        segs = [transcript.Segment("assistant", transcript.defuse(e.transcript_excerpt))
                for _, e in in_pass if e.transcript_excerpt]
        text = transcript.render(segs, width(cfg))
        wins = [Window(text, transcript.sources_in(text), None)] if text.strip() else []
        return Pass(in_pass, wins, None, None, k)

    per_line = transcript.line_segments([ln for _, ln in lines])
    packed = transcript.line_windows([(off, segs) for (off, _), segs in zip(lines, per_line)],
                                     width(cfg), int(cap["segment_max_windows"]))
    capped = bool(packed) and packed[-1][1] < read_end and _more(per_line, lines, packed[-1][1])
    end = packed[-1][1] if capped else read_end
    wins = []
    for text, off in packed:
        text = scrub(text) if scrub_on else text
        wins.append(Window(text, transcript.sources_in(text), off))
    if end < limit:
        log.info("segment: pass ends at %d of %d; the rest is distilled next", end, limit)
    return Pass(in_pass, wins, path, end, k, start=start, complete=reached and not capped)


def _more(per_line: list, lines: list[tuple[int, str]], off: int) -> bool:
    """True if a line after *off* has text: the windows stopped at their cap."""
    return any(segs for (o, _), segs in zip(lines, per_line) if o > off)
