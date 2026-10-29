"""Per-project state for UserPromptSubmit injection: session ledger, kill switch, metrics.

  <project>/sessions/<sid>.injected   ids already in this session's context, one per line
  <project>/inject.off                present = injection off for this project
  <project>/inject_metrics.jsonl      one line per prompt hook run (no prompt text)

All of it is best effort: a failure here never blocks a prompt.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LEDGER_TTL_S = 7 * 86400
METRICS_MAX_BYTES = 512 * 1024


def _ledger(pdir: Path, sid: str) -> Path | None:
    if not SESSION_RE.match(sid or ""):
        return None
    return pdir / "sessions" / f"{sid}.injected"


def ledger_ids(pdir: Path, sid: str) -> set[str]:
    p = _ledger(pdir, sid)
    try:
        return {line.strip() for line in p.read_text().splitlines() if line.strip()} if p else set()
    except OSError:
        return set()


def ledger_add(pdir: Path, sid: str, ids: list[str], reset: bool = False) -> None:
    """Record ids as injected. reset=True starts the ledger over (startup, clear, compact:
    whatever was in context before is gone)."""
    p = _ledger(pdir, sid)
    if p is None:
        return
    try:
        p.parent.mkdir(exist_ok=True)
        with p.open("w" if reset else "a") as f:
            f.writelines(f"{i}\n" for i in ids)
    except OSError:
        pass


def prune_ledgers(pdir: Path, now: float | None = None) -> None:
    now = now or time.time()
    for p in [*(pdir / "sessions").glob("*.injected"), *(pdir / "sessions").glob("*.compact.json")]:
        try:
            if now - p.stat().st_mtime > LEDGER_TTL_S:
                p.unlink()
        except OSError:
            pass


def disabled(pdir: Path) -> bool:
    return (pdir / "inject.off").exists()


def set_disabled(pdir: Path, off: bool) -> None:
    flag = pdir / "inject.off"
    if off:
        flag.write_text("prompt injection disabled for this project; remove this file to re-enable\n")
    else:
        flag.unlink(missing_ok=True)


def record(pdir: Path, **fields) -> None:
    """Append one metrics line; halve the file when it outgrows METRICS_MAX_BYTES."""
    p = pdir / "inject_metrics.jsonl"
    try:
        with p.open("a") as f:
            f.write(json.dumps({"ts": int(time.time()), **fields}) + "\n")
        if p.stat().st_size > METRICS_MAX_BYTES:
            lines = p.read_text().splitlines(keepends=True)
            p.write_text("".join(lines[len(lines) // 2:]))
    except OSError:
        pass


def metrics(pdir: Path, last: int = 500) -> list[dict]:
    try:
        lines = (pdir / "inject_metrics.jsonl").read_text().splitlines()[-last:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def summary(pdir: Path, last: int = 500) -> dict:
    """Latency and outcome counts over runs that reached retrieval (not skipped early)."""
    rows = metrics(pdir, last)
    ran = [r["ms"] for r in rows if r.get("outcome") in ("injected", "abstain", "deadline")]
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r.get("outcome", "?")] = outcomes.get(r.get("outcome", "?"), 0) + 1
    return {"n": len(rows), "n_retrieved": len(ran), "p50_ms": percentile(ran, 0.5),
            "p95_ms": percentile(ran, 0.95), "outcomes": outcomes,
            "embed_fallbacks": sum(1 for r in rows if r.get("vector") is False)}
