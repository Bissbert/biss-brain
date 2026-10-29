"""Retry state for segments whose distill or embed failed.

A failed segment keeps its queue events and its transcript cursor; what was
stored before the failure stays stored (segment.Pass.commit). This module
only decides when to try again: `<project>/pending/<key>.json` holds the
attempt count and the next try, backing off from `retry_base_seconds` to
`retry_max_seconds`. Transient failures (no reply, embed errors) have no
last attempt; doctor reports the backlog. A reply that stays invalid for the
same window (a prompt injection or refusal reproduces) is skipped after
`skip_invalid_after` tries by flush and recorded in `skipped.jsonl`, so the
rest of the session is not blocked behind it.

The state holds no transcript text, only counts, times and a short error.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

DIR = "pending"


def _file(pdir: Path, gk: str) -> Path:
    return Path(pdir) / DIR / (hashlib.sha256(gk.encode()).hexdigest()[:16] + ".json")


def state(pdir: Path, gk: str) -> dict | None:
    try:
        d = json.loads(_file(pdir, gk).read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def waiting(pdir: Path, gk: str, now: float | None = None) -> bool:
    """True while the segment is backing off after a failure."""
    d = state(pdir, gk)
    return d is not None and (now or time.time()) < float(d.get("next_try", 0))


def fail(pdir: Path, gk: str, kind: str, error: str, cfg: dict, now: float | None = None,
         at: int | None = None) -> dict:
    """Record a failed attempt at cursor *at* and schedule the next one.

    `stuck` counts the failures in a row with the same error at the same cursor.
    """
    now = now or time.time()
    d = state(pdir, gk) or {"group": gk, "attempts": 0, "first_failure": int(now)}
    d["attempts"] = int(d.get("attempts", 0)) + 1
    same = d.get("cursor") == at and d.get("error") == error[:200]
    d["stuck"] = int(d.get("stuck", 0)) + 1 if same else 1
    d["cursor"] = at
    dc = cfg.get("distill", {})
    base = float(dc.get("retry_base_seconds", 60))
    delay = min(base * 2 ** (d["attempts"] - 1), float(dc.get("retry_max_seconds", 3600)))
    d.update(kind=kind, error=error[:200], last_failure=int(now), next_try=int(now + delay))
    f = _file(pdir, gk)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    os.chmod(tmp, 0o600)
    tmp.replace(f)
    return d


def clear(pdir: Path, gk: str) -> None:
    _file(pdir, gk).unlink(missing_ok=True)


def skip(pdir: Path, gk: str, path: str | None, start: int | None, end: int | None, error: str) -> None:
    """Record a window flush gave up on: where it is, not what it says."""
    f = Path(pdir) / DIR / "skipped.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = {"group": gk, "path": path, "start": start, "end": end, "error": error[:200], "ts": int(time.time())}
    fd = os.open(f, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def skipped(root: Path) -> list[dict]:
    """Every project's skipped windows, oldest first."""
    out = []
    for f in Path(root).glob(f"*/{DIR}/skipped.jsonl"):
        try:
            out += [{**json.loads(ln), "project_dir": str(f.parent.parent)} for ln in f.read_text().splitlines() if ln]
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda d: d.get("ts", 0))


EMBED_MARK = "embed.mark"


def mark_embed(pdir: Path, cwd: str) -> None:
    """Note that the project has chunks stored without a vector (flush._backfill)."""
    f = Path(pdir) / DIR / EMBED_MARK
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(cwd)


def embed_marks(root: Path) -> list[tuple[Path, str]]:
    """[(project dir, cwd)] of projects with chunks waiting for a vector."""
    out = []
    for f in sorted(Path(root).glob(f"*/{DIR}/{EMBED_MARK}")):
        try:
            out.append((f.parent.parent, f.read_text().strip()))
        except OSError:
            continue
    return out


def clear_embed(pdir: Path) -> None:
    (Path(pdir) / DIR / EMBED_MARK).unlink(missing_ok=True)


EMBED_SKIP = "embed_skip.json"


def embed_skips(pdir: Path) -> list[str]:
    """Chunk ids the embedder rejects while it is up; they keep a NULL vector."""
    try:
        d = json.loads((Path(pdir) / DIR / EMBED_SKIP).read_text())
        return [str(x) for x in d] if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def skip_embed(pdir: Path, chunk_id: str) -> None:
    f = Path(pdir) / DIR / EMBED_SKIP
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sorted(set(embed_skips(pdir)) | {chunk_id})))


def backlog(root: Path) -> list[dict]:
    """Every project's retry states, with the project dir, oldest failure first."""
    out = []
    for f in Path(root).glob(f"*/{DIR}/*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            out.append({**d, "project_dir": str(f.parent.parent)})
    return sorted(out, key=lambda d: d.get("first_failure", 0))
