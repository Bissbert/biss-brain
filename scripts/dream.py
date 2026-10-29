#!/usr/bin/env python3
"""Nightly dream pass — multi-stage offline consolidation of the brain.

Stage 1 (cheap, no LLM by default beyond conflict judge):
  1a backlink refresh    — re-run A-MEM-style top-K linking on chunks the
                           pass hasn't visited yet (synonym-edge effect)
  1b conflict sweep      — for each new chunk in the candidate window,
                           judge top-similar neighbour; on CONTRADICTS the
                           older (by event time) becomes invalid from the
                           newer's valid_from; an extends edge on EXTENDS
  1c decay               — recompute decay_score (same logic as decay.py)

Stage 2 (LLM, additive, no destructive edits):
  2a session arcs        — group raw chunks by session_id, distill into
                           kind='arc' chunks at consolidation_round=1
  2b daily summaries     — opt-in roll-up of arcs by date, kind='daily_summary'
                           at consolidation_round=2
  2c retrieval prefs     — handled inside recall.py / inject_context.py
                           via the [dream].prefer_summaries_in_recall flag

CLI:
  dream.py                       # all stages, all projects (nightly default)
  dream.py --stage 1             # only Stage 1
  dream.py --stage 2             # only Stage 2
  dream.py --project <hash>      # single project
  dream.py --once                # alias for nightly default; no-op flag for clarity
  dream.py --dry-run             # log what would change, write nothing

Idempotent. Safe to interrupt — every step commits per chunk / per session.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Make scripts/ importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lancedb  # noqa: E402

import common  # noqa: E402
from common import (  # noqa: E402
    BRAIN_ROOT,
    GLOBAL_PROJECT_HASH,
    content_hash,
    ensure_schema,
    load_config,
    now_unix,
    embed,
    fetch_by_ids,
    open_or_create_chunks,
    refresh_fts,
    vector_search,
)
import lance_filter as F  # noqa: E402
import provenance  # noqa: E402
import temporal  # noqa: E402
from cli import run_claude  # noqa: E402
from distill import render  # noqa: E402
from backlinks import (  # noqa: E402
    SIMILARITY_THRESHOLD,
    _add_typed_edge,
    _add_typed_edge_multi,
)
from decay import _compute_decay  # noqa: E402

log = logging.getLogger("biss-brain.dream")

# ── log setup (file + stderr) ───────────────────────────────────────────


def _setup_logging() -> None:
    BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    fh = logging.FileHandler(str(BRAIN_ROOT / "dream.log"))
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


# ── project enumeration ──────────────────────────────────────────────────


def _project_dirs(only: str | None = None) -> list[Path]:
    if not BRAIN_ROOT.exists():
        return []
    out: list[Path] = []
    for d in sorted(BRAIN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if len(name) != 12 or not all(c in "0123456789abcdef" for c in name):
            continue
        if only is not None and name != only:
            continue
        if not (d / "meta.sqlite").exists():
            continue
        out.append(d)
    return out


def _cwd_for_project(pdir: Path) -> str:
    """Return a path that hashes to this project's hash.

    Prefers project_path.txt when present; otherwise returns the project dir
    itself, which works because every helper that takes ``cwd`` only uses it
    via ``project_key(cwd)`` — and we sidestep that by injecting a known
    cwd through ``BISS_BRAIN_PROJECT_DIR_OVERRIDE``-style monkey patching is
    not needed: helpers like ``open_meta`` accept any cwd that hashes to the
    target. For dream, we hold the project hash directly and call SQLite/LanceDB
    via absolute paths, never via cwd-derived helpers.
    """
    label = pdir / "project_path.txt"
    if label.exists():
        try:
            return label.read_text().strip() or str(pdir)
        except OSError:
            return str(pdir)
    return str(pdir)


def _open_meta_direct(pdir: Path) -> sqlite3.Connection:
    """Open the project's meta.sqlite by absolute path, applying migrations."""
    conn = sqlite3.connect(str(pdir / "meta.sqlite"), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _open_lance(pdir: Path):
    db_path = pdir / "lancedb"
    if not db_path.exists():
        return None, None
    db = lancedb.connect(str(db_path))
    if "chunks" not in db.list_tables().tables:
        return db, None
    return db, db.open_table("chunks")


# ── embedding ─────────────────────────────────────────────────────────────


def _embed(text: str, cfg: dict) -> list[float]:
    return embed(text, cfg, purpose="document", timeout=60)


# ── Stage 1a: backlink refresh ────────────────────────────────────────────


def _write_backlinks_direct(
    pdir: Path, tbl: Any, proj: str, chunk_id: str, vec: list[float]
) -> bool:
    """A-MEM-style top-K backlink write using the project's absolute paths.

    Mirrors backlinks.write_backlinks but skips cwd-rehashing so it works on
    project dirs that may not have a project_path.txt label yet.
    Returns True on success, False on no-op.
    """
    try:
        candidates = vector_search(tbl, vec, F.live_in(proj, F.ne("id", chunk_id)), 3)
    except Exception as e:  # noqa: BLE001
        log.warning("backlink: candidate search failed for %s: %s", chunk_id[:8], e)
        return False
    neighbour_ids = [c["id"] for c in candidates if c["_cosine"] >= SIMILARITY_THRESHOLD]
    if not neighbour_ids:
        return False

    ts = now_unix()
    meta = _open_meta_direct(pdir)
    try:
        _add_typed_edge_multi(meta, chunk_id, neighbour_ids, "co_occurs", 1.0, ts, project=proj)
        for nid in neighbour_ids:
            _add_typed_edge(meta, nid, chunk_id, "co_occurs", 1.0, ts, project=proj)
        meta.commit()
    finally:
        meta.close()
    return True


def _stage1a_backlinks(pdir: Path, proj: str, cfg: dict, dry_run: bool) -> int:
    """Recompute related_ids for chunks whose last_dream_ts < ts. Returns count."""
    if not cfg["dream"]["stage1"].get("backlink_refresh", True):
        return 0

    meta = _open_meta_direct(pdir)
    try:
        rows = meta.execute(
            "SELECT id FROM chunks "
            "WHERE project = ? AND superseded_by = '' "
            "AND last_dream_ts < ts "
            "AND consolidation_round = 0 "
            "ORDER BY ts ASC",
            (proj,),
        ).fetchall()
    finally:
        meta.close()

    if not rows:
        return 0

    _, tbl = _open_lance(pdir)
    if tbl is None:
        return 0

    refreshed = 0
    for r in rows:
        cid = r["id"]
        try:
            lance_row = tbl.search().where(F.eq("id", cid)).limit(1).to_list()
        except Exception as e:  # noqa: BLE001
            log.warning("backlink: lance lookup failed for %s: %s", cid[:8], e)
            continue
        if not lance_row:
            continue
        vec = lance_row[0].get("vector")
        if not vec:
            continue
        if dry_run:
            refreshed += 1
            continue
        if _write_backlinks_direct(pdir, tbl, proj, cid, list(vec)):
            refreshed += 1

    return refreshed


# ── Stage 1b: conflict sweep ──────────────────────────────────────────────


def _stage1b_conflicts(pdir: Path, proj: str, cfg: dict, dry_run: bool) -> dict:
    """Run the conflict judge against recently-added chunks.

    Returns a small dict of counters: {checked, contradicts, extends, unrelated, judge_failed}.
    A failed judge call (None) changes nothing, so the pair is re-judged on the next
    sweep while the new chunk is inside the candidate window.
    """
    if not cfg["dream"]["stage1"].get("conflict_sweep", True):
        return {"checked": 0, "contradicts": 0, "extends": 0, "unrelated": 0, "judge_failed": 0, "deferred": 0}

    sim_threshold = float(cfg["dream"]["stage1"].get("conflict_sim_threshold", 0.85))
    window_h = int(cfg["dream"]["stage1"].get("candidate_window_hours", 24))
    cutoff = now_unix() - window_h * 3600

    db, tbl = _open_lance(pdir)
    if tbl is None:
        return {"checked": 0, "contradicts": 0, "extends": 0, "unrelated": 0, "judge_failed": 0, "deferred": 0}

    meta = _open_meta_direct(pdir)
    try:
        rows = meta.execute(
            "SELECT id, kind FROM chunks "
            "WHERE project = ? AND superseded_by = '' "
            "AND ts >= ? "
            "AND consolidation_round = 0 "
            "ORDER BY ts ASC",
            (proj, cutoff),
        ).fetchall()
    finally:
        meta.close()

    if not rows:
        return {"checked": 0, "contradicts": 0, "extends": 0, "unrelated": 0, "judge_failed": 0, "deferred": 0}

    # Lazy import — judge.py shells out and is the most expensive thing here.
    from judge import judge_conflict  # noqa: PLC0415

    judge_model = cfg["distill"]["models"].get("conflict_judge", "haiku")
    counters = {"checked": 0, "contradicts": 0, "extends": 0, "unrelated": 0, "judge_failed": 0, "deferred": 0}

    for r in rows:
        cid = r["id"]
        kind = r["kind"]
        try:
            new_lance = tbl.search().where(F.eq("id", cid)).limit(1).to_list()
        except Exception:  # noqa: BLE001
            continue
        if not new_lance:
            continue
        new_text = new_lance[0].get("text", "")
        new_vec = new_lance[0].get("vector")
        if not new_text or not new_vec:
            continue

        try:
            cand = vector_search(
                tbl, list(new_vec), F.live_in(proj, F.eq("kind", kind), F.ne("id", cid)), 3
            )
        except Exception:  # noqa: BLE001
            continue
        if not cand:
            continue

        best = cand[0]
        if best["_cosine"] < sim_threshold:
            continue

        old_id = best["id"]
        old_text = best.get("text", "")
        if not old_text:
            continue

        # Web/tool text never supersedes a more trusted memory: DEFER unjudged.
        meta = _open_meta_direct(pdir)
        try:
            provs = provenance.lookup(meta, proj, [cid, old_id])
            times = temporal.validity(meta, proj, [cid, old_id])
        finally:
            meta.close()
        # The neighbour may be the newer memory: the newer one is "new".
        if cid in times and old_id in times and times[old_id].valid_from > times[cid].valid_from:
            cid, old_id, new_text, old_text = old_id, cid, old_text, new_text
        if not provenance.may_supersede(provs.get(cid), provs.get(old_id)):
            counters["deferred"] += 1
            continue

        counters["checked"] += 1
        if dry_run:
            continue

        verdict = judge_conflict(new_text, old_text, model=judge_model)
        if verdict == "CONTRADICTS":
            counters["contradicts"] += 1
            meta = _open_meta_direct(pdir)
            try:
                temporal.invalidate(meta, proj, old_id, cid)
                meta.commit()
            finally:
                meta.close()
            try:
                tbl.update(where=F.eq("id", old_id), values={"superseded_by": cid})
            except Exception as e:  # noqa: BLE001
                log.warning("lance update superseded_by failed: %s", e)
        elif verdict == "EXTENDS":
            counters["extends"] += 1
            meta = _open_meta_direct(pdir)
            try:
                _add_typed_edge(meta, cid, old_id, "extends", 1.0, now_unix(), project=proj)
                meta.commit()
            finally:
                meta.close()
        elif verdict is None:
            counters["judge_failed"] += 1
        else:
            counters["unrelated"] += 1

    return counters


# ── Stage 1c: decay ───────────────────────────────────────────────────────


def _stage1c_decay(pdir: Path, cfg: dict, dry_run: bool) -> int:
    half_life = float(cfg.get("hygiene", {}).get("decay_half_life_days", 30))
    usage_boost = float(cfg.get("hygiene", {}).get("decay_usage_boost", 0.0))
    meta = _open_meta_direct(pdir)
    try:
        rows = meta.execute(
            "SELECT id, ts, retrieval_count, pinned FROM chunks"
        ).fetchall()
        if not rows:
            return 0
        if dry_run:
            return len(rows)
        updates = [
            (
                _compute_decay(r["ts"], r["retrieval_count"] or 0, r["pinned"] or 0, half_life, usage_boost),
                r["id"],
            )
            for r in rows
        ]
        meta.executemany("UPDATE chunks SET decay_score = ? WHERE id = ?", updates)
        meta.commit()
        return len(updates)
    finally:
        meta.close()


# ── Stage 1 mark visited ─────────────────────────────────────────────────


def _stage1_mark_visited(pdir: Path, proj: str, dry_run: bool) -> None:
    if dry_run:
        return
    now = now_unix()
    meta = _open_meta_direct(pdir)
    try:
        meta.execute(
            "UPDATE chunks SET last_dream_ts = ? "
            "WHERE project = ? AND superseded_by = '' AND last_dream_ts < ts",
            (now, proj),
        )
        meta.commit()
    finally:
        meta.close()


# ── Stage 2: arc distillation ─────────────────────────────────────────────


def _distill_arc(text: str, model: str, kind: str = "arc") -> str:
    """Distill an arc/daily summary via cli.run_claude. Returns "" on any failure."""
    if os.environ.get("BISS_BRAIN_FAKE_DISTILL"):
        # Test hook — return the first line of the input prefixed
        first = text.strip().splitlines()[0] if text.strip() else "empty"
        return f"- ARC: {first[:80]}\n- ARC: synthetic"
    try:
        prompt = render("arc" if kind == "arc" else "daily", text=text)
    except (OSError, KeyError) as e:
        log.warning("dream prompt unusable: %s", e)
        return ""
    return run_claude(prompt, model, timeout=180) or ""


def _open_or_create_lance_table(pdir: Path, dim: int):
    return open_or_create_chunks(pdir / "lancedb", dim)


def _stage2a_arcs(pdir: Path, proj: str, cfg: dict, dry_run: bool) -> int:
    """Distill one summary chunk per session whose chunks haven't been consolidated."""
    cooldown_h = int(cfg["dream"].get("session_cooldown_hours", 6))
    min_chunks = int(cfg["dream"].get("min_chunks_per_arc", 3))
    cutoff = now_unix() - cooldown_h * 3600
    arc_model = cfg["dream"]["models"].get("arc", "haiku")

    meta = _open_meta_direct(pdir)
    try:
        # Group only chunks not yet attached to a parent arc — keeps Stage 2a
        # idempotent across re-runs.
        sessions = meta.execute(
            "SELECT session_id, COUNT(*) AS n, MIN(ts) AS t0, MAX(ts) AS t1 "
            "FROM chunks "
            "WHERE project = ? AND consolidation_round = 0 "
            "AND superseded_by = '' "
            "AND session_id != '' "
            "AND parent_id = '' "
            f"AND {provenance.trusted_sql()} "
            "GROUP BY session_id "
            "HAVING n >= ? AND t1 < ?",
            (proj, min_chunks, cutoff),
        ).fetchall()
    finally:
        meta.close()

    if not sessions:
        return 0

    tbl = _open_or_create_lance_table(pdir, cfg["embed"]["dim"])
    arcs_built = 0

    for sess in sessions:
        sid = sess["session_id"]
        meta = _open_meta_direct(pdir)
        try:
            members = meta.execute(
                f"SELECT id, kind, ts, {temporal.VALID_FROM_SQL} AS vf FROM chunks "
                "WHERE project = ? AND session_id = ? "
                "AND consolidation_round = 0 "
                "AND superseded_by = '' "
                "AND parent_id = '' "
                # Arcs summarise first-party memories only; tool/web-tainted
                # chunks stay standalone so their taint can't launder.
                f"AND {provenance.trusted_sql()} "
                "ORDER BY ts ASC",
                (proj, sid),
            ).fetchall()
            arc_prov = provenance.combine(
                provenance.lookup(meta, proj, [m["id"] for m in members]).values(), cfg)
        finally:
            meta.close()

        if len(members) < min_chunks:
            continue

        member_ids = [m["id"] for m in members]
        try:
            lance_rows = (
                tbl.search().where(F.isin("id", member_ids)).limit(len(member_ids)).to_list()
            )
        except Exception as e:  # noqa: BLE001
            log.warning("dream stage2: lance fetch failed for session %s: %s", sid[:8], e)
            continue

        text_by_id = {r["id"]: r.get("text", "") for r in lance_rows}
        # Order by ts ASC matches members
        joined_lines: list[str] = []
        for m in members:
            t = text_by_id.get(m["id"], "")
            if not t:
                continue
            ts_str = time.strftime("%H:%M", time.localtime(m["ts"]))
            joined_lines.append(f"[{ts_str} {m['kind']}] {t}")
        joined = "\n\n".join(joined_lines)
        if not joined:
            continue

        if dry_run:
            log.info("dream stage2 dry-run: would build arc for session %s (%d chunks)",
                     sid[:8], len(members))
            arcs_built += 1
            continue

        summary = _distill_arc(joined, model=arc_model, kind="arc")
        if not summary:
            log.warning("dream stage2: empty distill output for session %s", sid[:8])
            continue

        # Insert as new chunk
        try:
            vec = _embed(summary, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("dream stage2: embed failed for session %s: %s", sid[:8], e)
            continue

        arc_id = uuid.uuid4().hex
        ts = now_unix()
        # An arc is true from its earliest member's event time.
        arc_valid_from = min(m["vf"] for m in members)
        ch = content_hash(summary)
        try:
            tbl.add(
                [
                    {
                        "id": arc_id,
                        "project": proj,
                        "session_id": sid,
                        "agent_id": "",
                        "kind": "arc",
                        "ts": ts,
                        "text": summary,
                        "content_hash": ch,
                        "superseded_by": "",
                        "vector": vec,
                    }
                ]
            )
            refresh_fts(tbl)
        except Exception as e:  # noqa: BLE001
            log.warning("dream stage2: lance insert failed for session %s: %s", sid[:8], e)
            continue

        # Update SQLite: insert arc row + set parent_id on members
        meta = _open_meta_direct(pdir)
        try:
            evidence_csv = ",".join(member_ids)
            meta.execute(
                "INSERT INTO chunks "
                "(id, project, session_id, agent_id, kind, ts, content_hash, "
                " superseded_by, pinned, last_accessed, "
                " evidence_ids, consolidation_round, last_dream_ts, source, trust, taint, valid_from) "
                "VALUES (?, ?, ?, '', 'arc', ?, ?, '', 0, ?, ?, 1, ?, ?, ?, ?, ?)",
                (arc_id, proj, sid, ts, ch, ts, evidence_csv, ts, *arc_prov.row(), arc_valid_from),
            )
            meta.executemany(
                "UPDATE chunks SET parent_id = ? WHERE id = ? AND project = ?",
                [(arc_id, mid, proj) for mid in member_ids],
            )
            meta.commit()
        finally:
            meta.close()

        arcs_built += 1
        log.info("dream stage2: built arc %s for session %s (%d members)",
                 arc_id[:8], sid[:8], len(member_ids))

    return arcs_built


def _stage2b_daily(pdir: Path, proj: str, cfg: dict, dry_run: bool) -> int:
    """Roll up arcs by date into kind='daily_summary'. Opt-in."""
    if not cfg["dream"].get("daily_summary_enabled", False):
        return 0

    daily_model = cfg["dream"]["models"].get("daily", "sonnet")

    meta = _open_meta_direct(pdir)
    try:
        # Group arcs by local event date (YYYY-MM-DD), only those without a
        # parent (so we don't roll a daily into a daily).
        rows = meta.execute(
            f"SELECT id, {temporal.VALID_FROM_SQL} AS ts FROM chunks "
            "WHERE project = ? AND kind = 'arc' "
            "AND parent_id = '' AND superseded_by = '' "
            "AND consolidation_round = 1 "
            f"AND {provenance.trusted_sql()} "
            "ORDER BY ts ASC",
            (proj,),
        ).fetchall()
        arc_provs = provenance.lookup(meta, proj, [r["id"] for r in rows])
    finally:
        meta.close()

    if not rows:
        return 0

    by_date: dict[str, list[str]] = {}
    first_ts: dict[str, int] = {}
    for r in rows:
        d = time.strftime("%Y-%m-%d", time.localtime(r["ts"]))
        by_date.setdefault(d, []).append(r["id"])
        first_ts.setdefault(d, r["ts"])

    today = time.strftime("%Y-%m-%d", time.localtime(now_unix()))
    days_built = 0
    tbl = _open_or_create_lance_table(pdir, cfg["embed"]["dim"])

    for date, arc_ids in by_date.items():
        if date == today:
            continue  # don't roll up the in-progress day
        if len(arc_ids) < 2:
            continue  # one arc is already a session summary; no need to wrap

        try:
            lance_rows = tbl.search().where(F.isin("id", arc_ids)).limit(len(arc_ids)).to_list()
        except Exception:  # noqa: BLE001
            continue
        joined = "\n\n".join(r.get("text", "") for r in lance_rows if r.get("text"))
        if not joined:
            continue

        if dry_run:
            days_built += 1
            continue

        summary = _distill_arc(joined, model=daily_model, kind="daily")
        if not summary:
            continue

        try:
            vec = _embed(summary, cfg)
        except Exception:  # noqa: BLE001
            continue

        did = uuid.uuid4().hex
        ts = now_unix()
        ch = content_hash(summary)
        try:
            tbl.add(
                [
                    {
                        "id": did,
                        "project": proj,
                        "session_id": "",
                        "agent_id": "",
                        "kind": "daily_summary",
                        "ts": ts,
                        "text": summary,
                        "content_hash": ch,
                        "superseded_by": "",
                        "vector": vec,
                    }
                ]
            )
            refresh_fts(tbl)
        except Exception:  # noqa: BLE001
            continue

        meta = _open_meta_direct(pdir)
        try:
            evidence_csv = ",".join(arc_ids)
            meta.execute(
                "INSERT INTO chunks "
                "(id, project, session_id, agent_id, kind, ts, content_hash, "
                " superseded_by, pinned, last_accessed, "
                " evidence_ids, consolidation_round, last_dream_ts, source, trust, taint, valid_from) "
                "VALUES (?, ?, '', '', 'daily_summary', ?, ?, '', 0, ?, ?, 2, ?, ?, ?, ?, ?)",
                (did, proj, ts, ch, ts, evidence_csv, ts,
                 *provenance.combine((arc_provs.get(a) or provenance.IMPORTED for a in arc_ids), cfg).row(),
                 first_ts[date]),
            )
            meta.executemany(
                "UPDATE chunks SET parent_id = ? WHERE id = ? AND project = ?",
                [(did, aid, proj) for aid in arc_ids],
            )
            meta.commit()
        finally:
            meta.close()
        days_built += 1
        log.info("dream stage2: built daily %s for %s (%d arcs)", did[:8], date, len(arc_ids))

    return days_built


# ── Global project init ───────────────────────────────────────────────────


def _ensure_global_project(cfg: dict) -> Path:
    """Initialise ~/.claude/brain/_global/ if it does not yet exist.

    Idempotent. Creates:
      - meta.sqlite  (standard chunks schema via open_meta)
      - lancedb/     (directory, table created lazily on first write)
      - project_path.txt  containing the literal string ``__global__``

    Uses GLOBAL_PROJECT_HASH as the project identifier, consistent with
    the constant pinned in common.py.
    """
    global_pdir = BRAIN_ROOT / "_global"
    global_pdir.mkdir(parents=True, exist_ok=True)

    # Write project_path.txt sentinel
    label_file = global_pdir / "project_path.txt"
    if not label_file.exists():
        label_file.write_text("__global__")

    # Initialise meta.sqlite via open_meta (applies all migrations idempotently).
    # open_meta takes a cwd that hashes to the project dir. We pass the global_pdir
    # directly via _open_meta_direct which reads the absolute path.
    db_path = global_pdir / "meta.sqlite"
    if not db_path.exists():
        # Bootstrap by calling open_meta on the sentinel path string.
        # open_meta hashes the cwd to derive the DB path — we sidestep this by
        # using _open_meta_direct which accepts an absolute project directory.
        conn = _open_meta_direct(global_pdir)
        conn.close()

    # Ensure lancedb/ directory exists (table created on first vector write)
    lance_dir = global_pdir / "lancedb"
    lance_dir.mkdir(exist_ok=True)

    return global_pdir


def _open_global_lance(cfg: dict):
    """Open (or create) the _global LanceDB table. Returns (db, tbl) or (db, None)."""
    global_pdir = BRAIN_ROOT / "_global"
    global_pdir.mkdir(parents=True, exist_ok=True)
    lance_dir = global_pdir / "lancedb"
    lance_dir.mkdir(exist_ok=True)
    db = lancedb.connect(str(lance_dir))
    if "chunks" not in db.list_tables().tables:
        return db, None
    return db, db.open_table("chunks")


def _eligible_for_global(chunk: sqlite3.Row, cfg: dict) -> bool:
    """Return True if the chunk passes the _global admission gate.

    Gates (all must pass):
      1. kind ∈ cfg.cross_project.admission.eligible_kinds
      2. consolidation_round >= cfg.cross_project.admission.min_consolidation_round
      3. pinned == 1  OR  retrieval_count_local >= min_retrievals

    This is conservative by default (Path A from orchestration spec §4.1):
    under-admits rather than over-admits.  Distinct-projects count is deferred
    to v0.7 (Path B).
    """
    admission = cfg["cross_project"]["admission"]
    eligible_kinds = admission["eligible_kinds"]
    min_round = admission["min_consolidation_round"]
    min_retrievals = admission["frequency"]["min_retrievals"]

    kind = chunk["kind"] if isinstance(chunk, sqlite3.Row) else chunk.get("kind", "")
    consolidation_round = (
        chunk["consolidation_round"]
        if isinstance(chunk, sqlite3.Row)
        else chunk.get("consolidation_round", 0)
    ) or 0
    pinned = (
        chunk["pinned"] if isinstance(chunk, sqlite3.Row) else chunk.get("pinned", 0)
    ) or 0
    retrieval_count_local = (
        chunk["retrieval_count_local"]
        if isinstance(chunk, sqlite3.Row)
        else chunk.get("retrieval_count_local", 0)
    ) or 0

    if kind not in eligible_kinds:
        return False
    if consolidation_round < min_round:
        return False
    if not (pinned == 1 or retrieval_count_local >= min_retrievals):
        return False
    return True


def _select_global_candidates(pdir: Path, proj: str, cfg: dict) -> list[sqlite3.Row]:
    """Return chunks that are not yet cross_eligible and could qualify for _global."""
    admission = cfg["cross_project"]["admission"]
    eligible_kinds = admission["eligible_kinds"]
    min_round = admission["min_consolidation_round"]

    kinds_placeholder = ",".join("?" * len(eligible_kinds))
    meta = _open_meta_direct(pdir)
    try:
        rows = meta.execute(
            f"SELECT * FROM chunks "
            f"WHERE project = ? "
            f"AND superseded_by = '' "
            f"AND cross_eligible = 0 "
            f"AND kind IN ({kinds_placeholder}) "
            f"AND consolidation_round >= ? "
            # tool/web-tainted memories stay in their project
            f"AND {provenance.trusted_sql()}",
            (proj, *eligible_kinds, min_round),
        ).fetchall()
    finally:
        meta.close()
    return rows


def _two_phase_promote(
    chunk: sqlite3.Row,
    source_proj: str,
    pdir: Path,
    cfg: dict,
    dry_run: bool,
) -> bool:
    """Atomically promote a chunk to _global.

    Phase 1: write to _global meta.sqlite + lancedb.
    Phase 2: only on success, set cross_eligible=1 on the source chunk.

    If Phase 1 fails, the source chunk is untouched (cross_eligible stays 0).
    The global write is INSERT OR IGNORE so re-runs are no-ops (idempotent).

    NEVER writes superseded_by — R2 invariant.
    """
    if dry_run:
        log.info(
            "global promotion dry-run: would promote chunk %s (kind=%s) from project %s",
            chunk["id"][:8],
            chunk["kind"],
            source_proj,
        )
        return True

    global_pdir = _ensure_global_project(cfg)
    chunk_id = chunk["id"]

    # ── Phase 1: write to _global ─────────────────────────────────────────
    try:
        g_conn = _open_meta_direct(global_pdir)
        try:
            g_conn.execute(
                "INSERT OR IGNORE INTO chunks "
                "(id, project, session_id, agent_id, kind, ts, content_hash, "
                " superseded_by, pinned, retrieval_count, last_accessed, "
                " consolidation_round, last_dream_ts, cross_eligible, "
                " retrieval_count_local, source, trust, taint) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, '', 0, 0, NULL, ?, 0, 1, 0, ?, ?, ?)",
                (
                    chunk_id,
                    GLOBAL_PROJECT_HASH,
                    chunk["session_id"] or "",
                    chunk["agent_id"] or "",
                    chunk["kind"],
                    chunk["ts"],
                    chunk["content_hash"],
                    chunk["consolidation_round"],
                    *provenance.from_row(chunk["source"], chunk["trust"], chunk["taint"]).row(),
                ),
            )
            g_conn.commit()
        finally:
            g_conn.close()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "global promotion phase 1 failed for chunk %s: %s", chunk_id[:8], e
        )
        return False

    # Phase 1: write the vector to global LanceDB
    try:
        _, src_tbl = _open_lance(pdir)
        if src_tbl is not None:
            src_rows = src_tbl.search().where(F.eq("id", chunk_id)).limit(1).to_list()
            if src_rows:
                src_vec = src_rows[0].get("vector")
                src_text = src_rows[0].get("text", "")
                if src_vec is not None:
                    g_tbl = _open_or_create_lance_table(global_pdir, cfg["embed"]["dim"])
                    try:
                        g_tbl.add(
                            [
                                {
                                    "id": chunk_id,
                                    "project": GLOBAL_PROJECT_HASH,
                                    "session_id": chunk["session_id"] or "",
                                    "agent_id": chunk["agent_id"] or "",
                                    "kind": chunk["kind"],
                                    "ts": chunk["ts"],
                                    "text": src_text,
                                    "content_hash": chunk["content_hash"],
                                    "superseded_by": "",
                                    "vector": src_vec,
                                }
                            ]
                        )
                        refresh_fts(g_tbl)
                    except Exception as e:  # noqa: BLE001
                        log.warning(
                            "global promotion LanceDB write failed for %s: %s",
                            chunk_id[:8],
                            e,
                        )
                        # Phase 1 partial failure — still safe to proceed since
                        # SQLite row is INSERT OR IGNORE; next run will retry vector.
    except Exception as e:  # noqa: BLE001
        log.warning(
            "global promotion LanceDB phase failed for chunk %s: %s",
            chunk_id[:8],
            e,
        )

    # ── Phase 2: mark source cross_eligible only after global write confirmed ──
    try:
        src_conn = _open_meta_direct(pdir)
        try:
            src_conn.execute(
                "UPDATE chunks SET cross_eligible = 1 WHERE id = ? AND project = ?",
                (chunk_id, source_proj),
            )
            src_conn.commit()
        finally:
            src_conn.close()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "global promotion phase 2 failed for chunk %s: %s", chunk_id[:8], e
        )
        # Source chunk stays untouched (cross_eligible=0); next run will retry.
        return False

    log.info(
        "global: promoted chunk %s (kind=%s) from project %s",
        chunk_id[:8],
        chunk["kind"],
        source_proj,
    )
    return True


def _mark_cross_eligible(chunk_id: str, source_proj: str, pdir: Path) -> None:
    """Set cross_eligible=1 on a source chunk (idempotent)."""
    src_conn = _open_meta_direct(pdir)
    try:
        src_conn.execute(
            "UPDATE chunks SET cross_eligible = 1 WHERE id = ? AND project = ?",
            (chunk_id, source_proj),
        )
        src_conn.commit()
    finally:
        src_conn.close()


def _stage_nrem2_edge_drain(pdir: Path, proj: str, cfg: dict, dry_run: bool) -> int:
    """Drain accumulated edge_bumps into typed co_occurs edges (NREM-2 sub-stage).

    Weight: min(1.0, n / 10.0) where n is the co-retrieval count — fresh
    edges at threshold 3 get weight 0.3, saturating at 10+ bumps.
    BEGIN IMMEDIATE prevents write-write contention with the DMN daemon.
    Idempotent via WHERE drained=0 filter.
    """
    if not cfg.get("stdp", {}).get("enabled", True):
        return 0

    threshold: int = int(cfg.get("stdp", {}).get("co_retrieval_min_count", 3))
    promoted = 0

    conn = _open_meta_direct(pdir)
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT chunk_a, chunk_b, COUNT(*) AS n, MAX(bumped_at) AS last_ts
              FROM edge_bumps
             WHERE project = ? AND drained = 0
             GROUP BY chunk_a, chunk_b
            HAVING n >= ?
            """,
            (proj, threshold),
        ).fetchall()

        for row in rows:
            chunk_a = row["chunk_a"]
            chunk_b = row["chunk_b"]
            n = row["n"]
            last_ts = row["last_ts"]
            weight = min(1.0, n / 10.0)
            if not dry_run:
                _add_typed_edge(conn, chunk_a, chunk_b, "co_occurs", weight, last_ts, project=proj)
                _add_typed_edge(conn, chunk_b, chunk_a, "co_occurs", weight, last_ts, project=proj)
                conn.execute(
                    "UPDATE edge_bumps SET drained = 1"
                    " WHERE project = ? AND chunk_a = ? AND chunk_b = ? AND drained = 0",
                    (proj, chunk_a, chunk_b),
                )
            promoted += 1

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    if promoted:
        log.info(
            "nrem2_edge_drain [%s] promoted=%d pairs (dry_run=%s)",
            proj, promoted, dry_run,
        )
    return promoted


def _stage_promote_to_global(pdir: Path, cfg: dict, dry_run: bool = False) -> int:
    """NREM-3 sub-stage: promote eligible chunks to the _global project.

    Called from run_stage2 (post arc-distillation). Gated permanently on
    cfg["cross_project"]["enabled"] — when False (the permanent default) this
    function is a no-op.

    Eligibility requirements (all must pass):
      - kind ∈ eligible_kinds  (default: schema, lesson, hypothesis)
      - consolidation_round >= min_consolidation_round  (default: 1)
      - pinned == 1  OR  retrieval_count_local >= min_retrievals  (default: 5)

    Promotion is two-phase atomic:
      Phase 1: write chunk to _global (INSERT OR IGNORE — idempotent)
      Phase 2: set cross_eligible=1 on source only after Phase 1 succeeds

    NEVER writes superseded_by — R2 invariant enforced structurally.
    """
    # PERMANENT default-off guard — must be the very first check.
    if not cfg["cross_project"]["enabled"]:
        return 0

    proj = pdir.name
    candidates = _select_global_candidates(pdir, proj, cfg)
    promoted = 0

    for chunk in candidates:
        if not _eligible_for_global(chunk, cfg):
            continue
        success = _two_phase_promote(chunk, proj, pdir, cfg, dry_run)
        if success:
            promoted += 1

    if promoted:
        log.info("global: promoted %d chunk(s) from project %s", promoted, proj)

    return promoted


# ── orchestration ─────────────────────────────────────────────────────────


# ── core proposals ───────────────────────────────────────────────────────

_CORE_SECTION = {"decision": "Decisions", "lesson": "Lessons", "schema": "Conventions"}
_CORE_MAX_CHARS = 300


def _stage2c_core_proposals(pdir: Path, proj: str, cfg: dict, dry_run: bool) -> int:
    """Suggest durable, often-recalled memories for core. Never writes core.

    Candidates: live, trusted (no tool/web taint), short decisions, lessons,
    schemas or pinned chunks recalled at least core.propose_min_retrievals
    times. Each lands in the proposal queue for /biss-brain:core review; a
    rejected proposal has the same id and is never queued again.
    """
    import core

    ccfg = cfg.get("core", {})
    limit = int(ccfg.get("proposals_per_dream", 5))
    if limit <= 0 or proj == GLOBAL_PROJECT_HASH:
        return 0
    conn = _open_meta_direct(pdir)
    try:
        rows = conn.execute(
            "SELECT id, kind, pinned, source, trust, taint FROM chunks "
            "WHERE project = ? AND superseded_by = '' AND retrieval_count >= ? "
            "AND (kind IN ('decision', 'lesson', 'schema') OR pinned = 1) "
            f"AND kind NOT IN ('quarantine', 'hypothesis') AND {provenance.trusted_sql()} "
            "ORDER BY pinned DESC, retrieval_count DESC LIMIT ?",
            (proj, int(ccfg.get("propose_min_retrievals", 3)), limit * 4),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return 0
    _, tbl = _open_lance(pdir)
    if tbl is None:
        return 0
    text_of = {r["id"]: r.get("text", "") for r in fetch_by_ids(tbl, [r["id"] for r in rows], limit=len(rows))}
    cwd = _cwd_for_project(pdir)
    try:
        existing = {" ".join(it.text.split()).lower() for _, path in core.layers_for(cwd)
                    for it in core.load(path).items()}
    except Exception:  # noqa: BLE001 — a moved repo or missing path: dedup by proposal id only
        existing = set()
    queued = 0
    for r in rows:
        text = " ".join((text_of.get(r["id"]) or "").split())
        if not text or len(text) > _CORE_MAX_CHARS or text.lower() in existing:
            continue
        if dry_run:
            queued += 1
        else:
            prov = provenance.from_row(r["source"], r["trust"], r["taint"])
            section = _CORE_SECTION.get(r["kind"], "Pinned")
            if core.propose(proj, "add", "dream", provenance.label(prov),
                            section=section, text=text, chunk_id=r["id"]):
                queued += 1
        if queued >= limit:
            break
    return queued


def run_stage1(pdir: Path, cfg: dict, dry_run: bool) -> dict:
    proj = pdir.name
    backlinks = _stage1a_backlinks(pdir, proj, cfg, dry_run)
    conflicts = _stage1b_conflicts(pdir, proj, cfg, dry_run)
    decayed = _stage1c_decay(pdir, cfg, dry_run)
    _stage1_mark_visited(pdir, proj, dry_run)
    log.info(
        "stage1 [%s] backlinks=%d conflicts=%s decayed=%d (dry_run=%s)",
        proj, backlinks, conflicts, decayed, dry_run,
    )
    return {"backlinks": backlinks, "conflicts": conflicts, "decayed": decayed}


def run_stage2(pdir: Path, cfg: dict, dry_run: bool) -> dict:
    proj = pdir.name
    # NREM-2 first: cheap STDP edge drain before heavy LLM stages
    edges_drained = _stage_nrem2_edge_drain(pdir, proj, cfg, dry_run)
    arcs = _stage2a_arcs(pdir, proj, cfg, dry_run)
    daily = _stage2b_daily(pdir, proj, cfg, dry_run)
    # NREM-3 sub-stage: cross-project promotion (gated on cross_project.enabled)
    promoted = _stage_promote_to_global(pdir, cfg, dry_run)
    try:
        core_proposed = _stage2c_core_proposals(pdir, proj, cfg, dry_run)
    except Exception:  # noqa: BLE001 — proposals are optional; never fail the pass
        log.exception("core proposals failed for %s", proj)
        core_proposed = 0
    log.info(
        "stage2 [%s] edges_drained=%d arcs=%d daily=%d global_promoted=%d core_proposed=%d (dry_run=%s)",
        proj, edges_drained, arcs, daily, promoted, core_proposed, dry_run,
    )
    return {"edges_drained": edges_drained, "arcs": arcs, "daily": daily, "global_promoted": promoted,
            "core_proposed": core_proposed}


def _record_state(pdir: Path, stage: str, error: str | None = None) -> None:
    """Note the pass in dream_state, which the UI's status page reads."""
    conn = _open_meta_direct(pdir)
    try:
        if error is None:
            conn.execute(
                "INSERT INTO dream_state (project, last_stage, last_complete_ts) VALUES (?, ?, ?) "
                "ON CONFLICT(project) DO UPDATE SET last_stage = excluded.last_stage, "
                "last_complete_ts = excluded.last_complete_ts, last_error = '', last_error_ts = 0",
                (pdir.name, stage, now_unix()),
            )
        else:
            conn.execute(
                "INSERT INTO dream_state (project, last_stage, last_error, last_error_ts) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(project) DO UPDATE SET last_stage = excluded.last_stage, "
                "last_error = excluded.last_error, last_error_ts = excluded.last_error_ts",
                (pdir.name, stage, error[:500], now_unix()),
            )
        conn.commit()
    except sqlite3.Error:
        log.warning("could not record dream_state for %s", pdir.name)
    finally:
        conn.close()


def _kick_flush() -> None:
    """Start the flush daemon when segments or embeds wait for a retry.

    The daemon exits while only retries are pending; without a new capture
    nothing would start it again.
    """
    import pending  # noqa: PLC0415

    if not (pending.backlog(BRAIN_ROOT) or pending.embed_marks(BRAIN_ROOT)):
        return
    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    root = Path(env_root).resolve() if env_root else Path(__file__).resolve().parent.parent
    if (root / "scripts" / "_run.sh").is_file():
        from capture import _ensure_daemon  # noqa: PLC0415

        log.info("retries pending; starting the flush daemon")
        _ensure_daemon(root)


def main() -> int:
    p = argparse.ArgumentParser(description="biss-brain nightly dream pass")
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--project", default=None, help="single project hash to process")
    p.add_argument("--once", action="store_true", help="explicit: run once and exit (default)")
    p.add_argument("--dry-run", action="store_true", help="log what would change, don't write")
    args = p.parse_args()

    _setup_logging()
    cfg = load_config()
    try:
        return _run(args, cfg)
    finally:
        # After the pass, so the daemon's writes do not race dream's.
        if not args.dry_run:
            _kick_flush()


def _run(args: argparse.Namespace, cfg: dict) -> int:

    bad = common.store_mismatch(cfg)
    if bad:
        log.error("embed settings differ from the store (%s); running decay only", bad)
    if bad or not cfg.get("dream", {}).get("enabled", True):
        if not bad:
            log.info("dream disabled in config; running decay only")
        # Still recompute decay so the nightly job stays useful.
        for pdir in _project_dirs(args.project):
            _stage1c_decay(pdir, cfg, args.dry_run)
        return 0

    summary: dict[str, Any] = {}
    for pdir in _project_dirs(args.project):
        proj = pdir.name
        s: dict[str, Any] = {}
        try:
            if args.stage in ("1", "all"):
                s["stage1"] = run_stage1(pdir, cfg, args.dry_run)
            if args.stage in ("2", "all"):
                s["stage2"] = run_stage2(pdir, cfg, args.dry_run)
        except Exception as e:
            if not args.dry_run:
                _record_state(pdir, f"STAGE{args.stage.upper()}", error=f"{type(e).__name__}: {e}")
            raise
        if not args.dry_run:
            _record_state(pdir, "COMPLETE" if args.stage == "all" else f"STAGE{args.stage}")
        summary[proj] = s

    log.info("dream pass complete: %s", json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
