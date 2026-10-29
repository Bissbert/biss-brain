#!/usr/bin/env python3
"""DMN (Default Mode Network) idle daemon — STDP edge drain and backlink refresh.

Runs as a long-lived background process. Wakes every
`dmn.idle_seconds_before_run` seconds (default 300 s). If the system and
all projects appear idle (no recent Stop events), it performs cheap,
no-LLM maintenance for each project:

  1. _drain_edge_bumps — drain co-retrieval edge_bumps into typed related_ids
     edges (STDP). Uses BEGIN IMMEDIATE to avoid write-upgrade deadlocks when
     dream.py is also running.
  2. _refresh_recent_backlinks — recompute related_ids for chunks added in the
     last hour by re-scanning their ANN neighbourhood.
  3. _update_dream_state — record DMN_TICK in the dream_state table.

No LLM calls. Pure indexing.

Usage:
    python -m scripts.dmn          # foreground, Ctrl-C to stop
    python scripts/dmn.py          # same

Install via launchd:
    /biss-brain:install --with-dmn writes and loads ch.bissbert.biss-brain.dmn
    (opt-in; the plain install leaves it out).

Signal handling:
    SIGTERM  — finish current project, then exit cleanly.
    SIGINT   — exit immediately (all work is idempotent; safe to drop).
"""
from __future__ import annotations

import logging
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Make scripts/ importable when run as __main__ or via python -m scripts.dmn
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lance_filter as F  # noqa: E402
from backlinks import SIMILARITY_THRESHOLD, _add_typed_edge, _add_typed_edge_multi  # noqa: E402
from common import BRAIN_ROOT, ensure_schema, load_config, now_unix, vector_search  # noqa: E402

log = logging.getLogger("biss-brain.dmn")

# ── logging setup ─────────────────────────────────────────────────────────


def _setup_logging() -> None:
    BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    fh = logging.FileHandler(str(BRAIN_ROOT / "dmn.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(sh)


# ── signal handling ───────────────────────────────────────────────────────

_shutdown = False


def _handle_sigterm(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _shutdown
    _shutdown = True
    log.info("dmn: SIGTERM received; will exit after current project")


# ── project enumeration ───────────────────────────────────────────────────


def _list_active_projects() -> list[Path]:
    """Enumerate all ~/.claude/brain/<12-hex>/ directories that have a meta.sqlite."""
    if not BRAIN_ROOT.exists():
        return []
    out: list[Path] = []
    for d in sorted(BRAIN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if len(name) != 12 or not all(c in "0123456789abcdef" for c in name):
            continue
        if not (d / "meta.sqlite").exists():
            continue
        out.append(d)
    return out


# ── idle detection ────────────────────────────────────────────────────────


def _most_recent_stop_ts() -> int:
    """Return the Unix timestamp of the most recent Stop event across all projects.

    Primary signal: the filename prefix of queue/*.json files encodes the
    enqueue timestamp (``<ts>-stop-<uuid>.json``).  We scan all project queue
    directories and take the maximum.

    Returns 0 if no Stop events have ever been queued.
    """
    latest = 0
    if not BRAIN_ROOT.exists():
        return latest
    for d in BRAIN_ROOT.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if len(name) != 12 or not all(c in "0123456789abcdef" for c in name):
            continue
        qdir = d / "queue"
        if not qdir.exists():
            continue
        for f in qdir.iterdir():
            if not f.is_file():
                continue
            fname = f.name
            # Filename format: <ts>-stop-<uuid>.json or <ts>-subagent_stop-<uuid>.json
            if not fname.endswith(".json"):
                continue
            if "-stop-" not in fname and "-subagent_stop-" not in fname:
                continue
            try:
                ts = int(fname.split("-")[0])
                if ts > latest:
                    latest = ts
            except (ValueError, IndexError):
                continue
    return latest


def _system_idle_seconds() -> int | None:
    """Return macOS system idle time in seconds, or None if unavailable.

    Uses ``ioreg -c IOHIDSystem`` which is macOS-specific and may be slow
    (~50 ms).  Treated as a supplementary signal only; never raises.
    """
    try:
        result = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "HIDIdleTime" in line:
                # Value is in nanoseconds
                parts = line.split()
                ns = int(parts[-1])
                return int(ns / 1_000_000_000)
    except Exception:  # noqa: BLE001
        pass
    return None


def _is_idle(cfg: dict) -> bool:
    """Return True if the system appears idle enough to run a DMN tick.

    Primary signal: no Stop event in any project queue within the last
    ``idle_seconds_before_run`` seconds.

    Secondary (macOS) signal: system HID idle time also exceeds threshold.
    The secondary signal is checked only when the primary says idle; if ioreg
    is unavailable the primary alone decides.
    """
    threshold = int(cfg["dmn"]["idle_seconds_before_run"])
    cutoff = now_unix() - threshold

    recent_stop_ts = _most_recent_stop_ts()
    if recent_stop_ts > cutoff:
        log.debug("dmn: not idle — recent Stop at %d (cutoff %d)", recent_stop_ts, cutoff)
        return False

    # Optional supplementary: macOS system idle
    sys_idle = _system_idle_seconds()
    if sys_idle is not None and sys_idle < threshold:
        log.debug("dmn: not idle — system HID idle only %d s (threshold %d)", sys_idle, threshold)
        return False

    return True


# ── SQLite helpers ────────────────────────────────────────────────────────


def _open_meta_direct(pdir: Path) -> sqlite3.Connection:
    """Open the project's meta.sqlite by absolute path, applying migrations."""
    conn = sqlite3.connect(str(pdir / "meta.sqlite"), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


# ── STDP edge drain ───────────────────────────────────────────────────────


def _drain_edge_bumps(pdir: Path, cfg: dict, deadline: float) -> int:
    """Drain undrained edge_bumps into co_occurs typed edges.

    Opens the DB with BEGIN IMMEDIATE to prevent write-upgrade deadlocks when
    dream.py is concurrently running its own drainer.

    Only bumps with COUNT(*) >= co_retrieval_min_count are promoted to edges.
    Only the bumps belonging to promoted pairs are marked drained=1 — bumps
    below threshold are left undrained so they can accumulate further and
    promote in a future tick.  This matches dream.py:_stage_nrem2_edge_drain
    semantics exactly.

    Returns the number of edges written.
    """
    if not cfg.get("stdp", {}).get("enabled", True):
        return 0

    min_count = int(cfg["stdp"]["co_retrieval_min_count"])
    proj = pdir.name
    edges_written = 0

    conn = _open_meta_direct(pdir)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                """
                SELECT chunk_a, chunk_b, COUNT(*) AS cnt, MAX(bumped_at) AS last_ts
                FROM edge_bumps
                WHERE project = ? AND drained = 0
                GROUP BY chunk_a, chunk_b
                HAVING cnt >= ?
                """,
                (proj, min_count),
            )
            rows = cur.fetchall()

            for row in rows:
                if time.monotonic() >= deadline:
                    log.info("dmn: budget exhausted mid-drain for project %s", proj)
                    break
                chunk_a = row["chunk_a"]
                chunk_b = row["chunk_b"]
                cnt = row["cnt"]
                last_ts = row["last_ts"]
                weight = min(1.0, cnt / 10.0)
                # Bidirectional edge — one call per direction (matches backlinks API)
                _add_typed_edge(conn, chunk_a, chunk_b, "co_occurs", weight, last_ts, project=proj)
                _add_typed_edge(conn, chunk_b, chunk_a, "co_occurs", weight, last_ts, project=proj)
                # Mark only the bumps that were consumed in this promoted pair
                conn.execute(
                    "UPDATE edge_bumps SET drained = 1"
                    " WHERE project = ? AND chunk_a = ? AND chunk_b = ? AND drained = 0",
                    (proj, chunk_a, chunk_b),
                )
                edges_written += 1

            conn.execute("COMMIT")
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    if edges_written:
        log.info("dmn: drained %d edge(s) for project %s", edges_written, proj)
    return edges_written


# ── backlink refresh ──────────────────────────────────────────────────────


def _refresh_recent_backlinks(pdir: Path, cfg: dict, deadline: float) -> int:
    """Re-scan chunks added in the last hour and recompute their related_ids.

    Pure dense-vector ANN search — no LLM calls.  Returns count of chunks
    refreshed.  Silently skips if LanceDB is unavailable for this project.
    """
    try:
        import lancedb  # noqa: PLC0415
    except ImportError:
        return 0

    lance_path = pdir / "lancedb"
    if not lance_path.exists():
        return 0

    db = lancedb.connect(str(lance_path))
    if "chunks" not in db.list_tables().tables:
        return 0
    tbl = db.open_table("chunks")

    proj = pdir.name
    cutoff = now_unix() - 3600  # last hour

    conn = _open_meta_direct(pdir)
    try:
        rows = conn.execute(
            "SELECT id FROM chunks "
            "WHERE project = ? AND superseded_by = '' AND ts >= ? "
            "ORDER BY ts DESC",
            (proj, cutoff),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0

    refreshed = 0
    for r in rows:
        if time.monotonic() >= deadline:
            log.info("dmn: budget exhausted mid-backlink-refresh for project %s", proj)
            break

        cid = r["id"]
        try:
            lance_row = tbl.search().where(F.eq("id", cid)).limit(1).to_list()
        except Exception:  # noqa: BLE001
            continue
        if not lance_row:
            continue
        vec = lance_row[0].get("vector")
        if vec is None:
            continue

        try:
            candidates = vector_search(tbl, list(vec), F.live_in(proj, F.ne("id", cid)), 5)
        except Exception:  # noqa: BLE001
            continue
        neighbour_ids = [c["id"] for c in candidates if c["_cosine"] >= SIMILARITY_THRESHOLD]
        if not neighbour_ids:
            continue

        conn = _open_meta_direct(pdir)
        try:
            _add_typed_edge_multi(conn, cid, neighbour_ids, "co_occurs", 0.5, now_unix(), project=proj)
            conn.commit()
        finally:
            conn.close()

        refreshed += 1

    if refreshed:
        log.info("dmn: refreshed backlinks for %d chunk(s) in project %s", refreshed, proj)
    return refreshed


# ── dream_state update ────────────────────────────────────────────────────


def _update_dream_state(pdir: Path, stage: str) -> None:
    """Record a DMN tick in the dream_state table for this project."""
    proj = pdir.name
    conn = _open_meta_direct(pdir)
    try:
        conn.execute(
            """
            INSERT INTO dream_state (project, last_stage, last_complete_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(project) DO UPDATE SET
                last_stage = excluded.last_stage,
                last_complete_ts = excluded.last_complete_ts
            """,
            (proj, stage, now_unix()),
        )
        conn.commit()
    finally:
        conn.close()


# ── main loop ─────────────────────────────────────────────────────────────


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)

    cfg = load_config()
    if not cfg.get("dmn", {}).get("enabled", True):
        # Exit before setting up logging so we don't create log files for a
        # disabled daemon.
        return 0

    _setup_logging()

    poll_interval = int(cfg["dmn"]["idle_seconds_before_run"])
    max_runtime = int(cfg["dmn"]["max_runtime_seconds"])

    log.info(
        "dmn: starting (poll=%ds, max_runtime=%ds)",
        poll_interval,
        max_runtime,
    )

    while not _shutdown:
        time.sleep(poll_interval)

        if _shutdown:
            break

        if not _is_idle(cfg):
            log.debug("dmn: not idle; skipping tick")
            continue

        deadline = time.monotonic() + max_runtime
        log.info("dmn: idle tick starting (budget=%ds)", max_runtime)

        for pdir in _list_active_projects():
            if _shutdown:
                break
            if time.monotonic() >= deadline:
                log.info("dmn: runtime budget exhausted; skipping remaining projects")
                break

            proj = pdir.name
            try:
                _drain_edge_bumps(pdir, cfg, deadline)
                if time.monotonic() < deadline:
                    _refresh_recent_backlinks(pdir, cfg, deadline)
                _update_dream_state(pdir, "DMN_TICK")
            except Exception as exc:  # noqa: BLE001
                log.warning("dmn: error in project %s: %s", proj, exc)

        log.info("dmn: tick complete")

    log.info("dmn: shutdown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
