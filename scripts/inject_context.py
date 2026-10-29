#!/usr/bin/env python3
"""Retrieval injection. Runs at SessionStart (startup, resume, clear, compact) and UserPromptSubmit.
Prints `{ "hookSpecificOutput": { "hookEventName": "<event>", "additionalContext": "..." } }`
to stdout so Claude Code injects the text into the model's next context.

Invoked as: inject_context.py <mode>
where <mode> in {session_start, post_compact, user_prompt_submit}.

After compaction, this session's pre_compact summary is restored if it is
stored, else the raw conversation-tail digest PreCompact wrote (compact.py);
the summary then follows once, on the first prompt after it lands.

Injected memories are framed as data, never as instructions: each one is a
<brain-memory source=… trust=… date=… kind=…> block (date = when it became
true, temporal.valid_from) whose text is
entity-escaped (so it cannot close the tag or open another), under a one-line
factual preamble. Attribute values are validated, not escaped.
"""
from __future__ import annotations

import time

T0 = time.perf_counter()  # prompt-hook latency and deadline count from here, imports included

import html  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

# lancedb costs ~0.8 s to import; it loads only on paths that read the store,
# so disabled, killed and trivial prompts exit in tens of ms.
import common  # noqa: E402
from common import (  # noqa: E402
    embed,
    fetch_by_ids,
    lancedb_dir,
    load_config,
    open_meta,
    project_dir,
    project_key,
)
import compact  # noqa: E402
import inject_state as S  # noqa: E402
import lance_filter as F  # noqa: E402
import provenance  # noqa: E402
import scope  # noqa: E402
import temporal  # noqa: E402
from effects import touch_injected  # noqa: E402
from retrieve import prompt_inject, trivial_prompt  # noqa: E402


def _embed_with_timeout(text: str, cfg: dict, timeout_s: float) -> list[float] | None:
    """Embed the prompt as a query, returning None if it takes longer than timeout_s."""
    import threading

    result: list[list[float]] = []

    def _worker() -> None:
        try:
            result.append(embed(text, cfg, purpose="query", timeout=timeout_s))
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0] if result else None


def _read_hook_input() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _table_or_none(cwd: str):
    ld = lancedb_dir(cwd)
    if not ld.exists():
        return None
    import lancedb

    db = lancedb.connect(str(ld))
    if "chunks" not in db.list_tables().tables:
        return None
    return db.open_table("chunks")


def _recent_decisions(cwd: str, top_k: int, cfg: dict | None = None) -> list[dict]:
    """recent_rows for cwd's store; [] when it has no chunks table."""
    meta = open_meta(cwd)
    try:
        tbl = _table_or_none(cwd)
        return recent_rows(meta, tbl, project_key(cwd), top_k, cfg) if tbl is not None else []
    finally:
        meta.close()


def recent_rows(meta, tbl, proj: str, top_k: int, cfg: dict | None = None) -> list[dict]:
    """Pull recently-accessed top-tier chunks for the project.

    Reads only: the UI's next-session preview calls this on a read-only
    connection, so what it shows is what the hook injects.

    Drives listing from SQLite metadata (fast index scans), then fetches
    the full text for matched ids from LanceDB using a by-id filter.

    Prefers consolidated chunks (kind='arc' / 'daily_summary') over raw
    distillations when both exist for the same session — arcs are the
    higher-signal product of the dream pass and feed the model better.
    Tool/web-tainted and quarantined chunks are never injected.
    """
    # Prefer summaries: arc/daily_summary outrank stop/subagent_stop, then pinned, then recency.
    rows = meta.execute(
        f"SELECT id, kind, ts, session_id, {temporal.VALID_FROM_SQL} FROM chunks "
        "WHERE project = ? AND superseded_by = '' AND kind != 'hypothesis' AND kind != 'quarantine' "
        f"AND {provenance.injectable_sql(cfg)} "
        "ORDER BY "
        "  CASE WHEN kind IN ('daily_summary', 'arc') THEN 0 ELSE 1 END, "
        "  pinned DESC, ts DESC "
        "LIMIT ?",
        (proj, top_k * 3),
    ).fetchall()
    if not rows:
        return []

    id_to_meta = {r[0]: {"id": r[0], "kind": r[1], "ts": r[2], "session_id": r[3], "_valid_from": r[4]}
                  for r in rows}
    ids = list(id_to_meta.keys())
    for cid, prov in provenance.lookup(meta, proj, ids).items():
        id_to_meta[cid]["_prov"] = prov

    lance_rows = fetch_by_ids(tbl, ids, limit=len(ids))
    # Merge lance text into meta rows
    results: list[dict] = []
    for lr in lance_rows:
        entry = dict(id_to_meta.get(lr["id"], {}))
        if not entry:
            continue
        entry["text"] = lr.get("text", "")
        results.append(entry)

    # Sort: summaries (arc/daily_summary) first, then by ts descending.
    # Pinned already at top via SQLite ORDER BY in caller.
    def _kind_priority(kind: str) -> int:
        if kind in ("daily_summary", "arc"):
            return 0
        return 1
    results.sort(key=lambda r: (_kind_priority(r.get("kind", "")), -r["ts"]))
    return results[:top_k]


def _format_chunks(chunks: list[dict], header: str) -> str:
    if not chunks:
        return ""
    lines = [f"## {header}", ""]
    for c in chunks:
        lines.append(f"### [{c['kind']}] {c.get('session_id', '')[:8]}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines).strip()


PREAMBLE = "Recalled notes from earlier sessions; reference only."
_SESSION = S.SESSION_RE


def _attrs(c: dict) -> str:
    p = c.get("_prov") or provenance.IMPORTED
    kind = c.get("kind") if c.get("kind") in F.KINDS else "note"
    try:
        when = c["_valid_from"] if c.get("_valid_from") is not None else c["ts"]
        date = time.strftime("%Y-%m-%d", time.localtime(int(when)))
    except (KeyError, TypeError, ValueError, OverflowError):
        date = "unknown"
    sid = str(c.get("session_id") or "")[:8]
    attrs = [
        f'source="{provenance.label(p)}"',
        f'trust="{p.trust:.1f}"',
        f'date="{date}"',
        f'kind="{kind}"',
    ]
    if _SESSION.match(sid):
        attrs.append(f'session="{sid}"')
    return " ".join(attrs)


def _frame_chunks(chunks: list[dict], header: str) -> str:
    """Hook context: memories as escaped <brain-memory> data blocks."""
    if not chunks:
        return ""
    lines = [f"## {header}", "", PREAMBLE, ""]
    for c in chunks:
        lines.append(f"<brain-memory {_attrs(c)}>")
        lines.append(html.escape(c.get("text", ""), quote=False))
        lines.append("</brain-memory>")
        lines.append("")
    return "\n".join(lines).strip()


def _compact_summary(cwd: str, cfg: dict, pdir, sid: str, since: int | None) -> tuple[str, list[str]]:
    """This session's framed pre_compact summary from *since* on, and its ids; ("", []) if not stored."""
    if not sid or not (pdir / "lancedb").exists():
        return "", []
    proj = project_key(cwd)
    meta = open_meta(cwd)
    try:
        hits = compact.summary_rows(meta, proj, sid, since, provenance.injectable_sql(cfg))
        provs = provenance.lookup(meta, proj, [h[0] for h in hits]) if hits else {}
    finally:
        meta.close()
    tbl = _table_or_none(cwd) if hits else None
    by_id = {r["id"]: r for r in fetch_by_ids(tbl, [h[0] for h in hits])} if tbl is not None else {}
    rows = [{**by_id[i], "_prov": provs.get(i), "_valid_from": vf} for i, vf in hits if i in by_id]
    if not rows:
        return "", []
    return _frame_chunks(rows, "Restored from pre-compaction memory"), [r["id"] for r in rows]


def _frame_digest(d: dict) -> str:
    """The raw conversation tail, framed and escaped like a memory."""
    return "\n".join([
        "## Conversation before compaction (biss-brain; raw, not yet distilled)",
        "",
        "The last turns before this session was compacted, kept for continuity; reference only.",
        "",
        '<brain-memory source="transcript" kind="pre_compact">',
        html.escape(d["text"], quote=False),
        "</brain-memory>",
    ])


def _late_compact_summary(cwd: str, cfg: dict, pdir, sid: str) -> str:
    """The summary of a compaction that was restored from the digest, once it is stored."""
    if not cfg.get("retrieve", {}).get("compact_late_restore", True):
        return ""
    d = compact.read_digest(pdir, sid)
    if d is None or not compact.landed(pdir, d):
        return ""
    try:
        framed, ids = _compact_summary(cwd, cfg, pdir, sid, int(d["ts"]))
    except Exception:  # noqa: BLE001
        return ""
    # Landed without a chunk (no findings, or merged into an older one): nothing follows.
    compact.drop_digest(pdir, sid)
    if ids:
        S.ledger_add(pdir, sid, ids)
    return framed


def _core_context(cwd: str, cfg: dict) -> str:
    """Core memory, injected in full (core.frame caps and announces drops).

    Best effort: a broken core file or index never blocks the session. No
    network and no embedding on this path.
    """
    if not cfg.get("core", {}).get("inject", True):
        return ""
    try:
        import core

        core.sync(cwd)
        return core.frame(cwd, cfg)[0]
    except Exception:  # noqa: BLE001
        return ""


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def top_k(cfg: dict) -> int:
    return cfg["retrieve"]["session_start_top_k"]


def session_start_context(core_ctx: str, chunks: list[dict]) -> str:
    """What a new session starts with: core memory, then the recent memories."""
    return _join(core_ctx, _frame_chunks(chunks, "Recent project memory (biss-brain)"))


def _emit(event_name: str, context: str) -> None:
    out = {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": context,
        }
    }
    print(json.dumps(out))


def _core_texts(cwd: str) -> set[str]:
    """Normalised core item texts; a memory that repeats one is already in context."""
    try:
        import core

        return {" ".join(it.text.lower().split()) for _, path in core.layers_for(cwd)
                for it in core.load(path).items()}
    except Exception:  # noqa: BLE001
        return set()


def _select(cwd: str, cfg: dict, prompt: str, sid: str, pdir, t0: float) -> tuple[list[dict], dict]:
    """Embed (within budget), retrieve, gate. Returns (rows, metric fields)."""
    r = cfg["retrieve"]
    qvec = _embed_with_timeout(prompt, cfg, timeout_s=int(r.get("inject_embed_ms", 300)) / 1000)
    t_embed = time.perf_counter()
    info = {"embed_ms": round((t_embed - t0) * 1000, 1), "vector": qvec is not None}
    tbl = _table_or_none(cwd)
    if tbl is None:
        return [], {**info, "outcome": "no_store"}
    proj_key = project_key(cwd)
    seen = S.ledger_ids(pdir, sid)
    meta = open_meta(cwd)
    try:
        rows = prompt_inject(tbl, meta, proj_key, prompt, qvec, cfg, exclude=seen)
        core_texts = _core_texts(cwd) if rows else set()
        rows = [x for x in rows if " ".join(x.get("text", "").lower().split()) not in core_texts]
        if rows:
            touch_injected(meta, proj_key, [x["id"] for x in rows])
    finally:
        meta.close()
    return rows, {**info, "outcome": "injected" if rows else "abstain"}


def _user_prompt_submit(cwd: str, cfg: dict, hook: dict, pdir) -> int:
    """Proactive injection under a hard deadline.

    The work runs on a daemon thread; if it misses inject_deadline_ms the
    hook prints nothing and exits at once, so a slow store or embed server
    never holds the prompt longer than the deadline.
    """
    import threading

    t0 = T0
    r = cfg.get("retrieve", {})
    prompt = hook.get("prompt") or hook.get("message") or ""
    sid = str(hook.get("session_id") or "")
    # Part of the compaction restore, not retrieval: not gated by the inject switches.
    late = _late_compact_summary(cwd, cfg, pdir, sid)

    def emit(ctx: str = "") -> None:
        ctx = _join(late, ctx)
        if ctx:
            _emit("UserPromptSubmit", ctx)

    def done(outcome: str, **f) -> int:
        S.record(pdir, mode="user_prompt_submit", outcome=outcome,
                 ms=round((time.perf_counter() - t0) * 1000, 1), **f)
        return 0

    if not r.get("user_prompt_inject", False):
        emit()
        return 0
    if S.disabled(pdir) or os.environ.get("BISS_BRAIN_NO_INJECT") == "1":
        emit()
        return done("off")
    if trivial_prompt(prompt, cfg):
        emit()
        return done("trivial")

    box: dict = {}

    def work() -> None:
        try:
            box["rows"], box["info"] = _select(cwd, cfg, prompt, sid, pdir, t0)
        except Exception as e:  # noqa: BLE001
            box["error"] = type(e).__name__

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout=max(0.0, int(r.get("inject_deadline_ms", 2500)) / 1000 - (time.perf_counter() - t0)))
    if t.is_alive():
        emit()
        done("deadline")
        sys.stdout.flush()
        os._exit(0)  # don't wait for the stuck worker at interpreter exit
    if "error" in box:
        emit()
        return done("error", error=box["error"])
    rows, info = box["rows"], box["info"]
    if rows:
        S.ledger_add(pdir, sid, [x["id"] for x in rows])
    emit(_frame_chunks(rows, "Relevant project memory (biss-brain)"))
    return done(info.pop("outcome"), n=len(rows), **info)


def _post_compact(cwd: str, cfg: dict, pdir, sid: str, core_ctx: str) -> int:
    """SessionStart(source=compact): core memory plus this session's compaction restore.

    The pre_compact summary of this compaction if it has landed; otherwise
    the raw-tail digest, and the summary follows on a later prompt.
    """
    d = compact.read_digest(pdir, sid)
    restored, ids = "", []
    if d is None or compact.landed(pdir, d):
        try:
            restored, ids = _compact_summary(cwd, cfg, pdir, sid, int(d["ts"]) if d else None)
        except Exception:  # noqa: BLE001
            restored, ids = "", []
        if d:
            compact.drop_digest(pdir, sid)
    if d and not ids:
        restored = _frame_digest(d)
    # compaction dropped whatever was injected before
    S.ledger_add(pdir, sid, ids, reset=True)
    ctx = _join(core_ctx, restored)
    if ctx:
        _emit("SessionStart", ctx)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    mode = sys.argv[1]
    cfg = load_config()
    hook = _read_hook_input()
    cwd = hook.get("cwd") or os.getcwd()
    if not scope.check(cwd, cfg):
        return 0
    if common.store_mismatch(cfg):
        return 0  # vectors from another model would rank noise; doctor reports it

    if mode in ("session_start", "post_compact"):
        # Core memory first: it exists even for a project with no store yet,
        # and compaction drops it from context, so post_compact restores it.
        core_ctx = _core_context(cwd, cfg)
    else:
        core_ctx = ""

    proj = project_dir(cwd)
    sid = str(hook.get("session_id") or "")
    if mode == "post_compact":
        return _post_compact(cwd, cfg, proj, sid, core_ctx)
    if not (proj / "lancedb").exists():
        if core_ctx:
            _emit("SessionStart", core_ctx)
        return 0  # cold project, no recalled memory

    if mode == "session_start":
        chunks = _recent_decisions(cwd, top_k(cfg), cfg)
        # resume keeps the earlier context; startup/clear start from nothing
        S.ledger_add(proj, sid, [c["id"] for c in chunks], reset=hook.get("source") != "resume")
        S.prune_ledgers(proj)
        ctx = session_start_context(core_ctx, chunks)
        if ctx:
            _emit("SessionStart", ctx)
        return 0

    if mode == "user_prompt_submit":
        return _user_prompt_submit(cwd, cfg, hook, proj)

    return 0


if __name__ == "__main__":
    sys.exit(main())
