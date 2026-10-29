#!/usr/bin/env python3
"""Explicit user memory insertion.

Usage:
  remember.py [--kind decision|note|code|task] [--pin] [--source user] <text>

Provenance: only the user-typed /remember command passes --source user (the
command sets disable-model-invocation, so the model cannot run it through the
Skill tool). Any other caller — the model via Bash, a script — is recorded as
source=assistant, so a memory written by tool calls never gets user trust.

If --kind is not given, classifies the text via `claude -p` Haiku (cli.run_claude).
Inserts directly into LanceDB + SQLite (no distillation step).
Prints the new chunk_id on success.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path


# Allow running as script directly
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import common  # noqa: E402
from common import (
    content_hash,
    embed,
    lancedb_dir,
    load_config,
    now_unix,
    open_meta,
    open_or_create_chunks,
    project_key,
    refresh_fts,
)
import provenance  # noqa: E402
from cli import run_claude  # noqa: E402
from distill import render  # noqa: E402
from scrub import scrub
from transcript import strip_private  # noqa: E402

VALID_KINDS = {"decision", "note", "code", "task"}


# ── embedding and table ──────────────────────────────────────────────────

def _embed(text: str, cfg: dict) -> list[float]:
    return embed(text, cfg, purpose="document", timeout=60)


def _open_table(cwd: str, dim: int):
    return open_or_create_chunks(lancedb_dir(cwd), dim)


# ── kind classifier ───────────────────────────────────────────────────────

def _classify_kind(text: str) -> str:
    """Classify *text* into one of VALID_KINDS via Haiku. Falls back to 'note'."""
    try:
        prompt = render("classify", text=text)
    except (OSError, KeyError):
        return "note"
    raw = (run_claude(prompt, "haiku", timeout=30) or "").strip().lower()
    for kind in VALID_KINDS:
        if raw.startswith(kind):
            return kind
    return "note"


# ── backlink update (imported after insert to avoid circular) ─────────────

def _update_backlinks(cwd: str, proj: str, chunk_id: str, vec: list[float], cfg: dict) -> None:
    """Find top-3 similar chunks and write bidirectional related_ids.

    Silently skips if backlink helpers are not available (e.g. schema not migrated).
    """
    try:
        from backlinks import write_backlinks  # noqa: PLC0415
        write_backlinks(cwd, proj, chunk_id, vec, cfg)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass


# ── main ─────────────────────────────────────────────────────────────────

class StoreMismatch(RuntimeError):
    """The store's vectors were made with other embed settings."""


def insert_chunk(
    text: str,
    kind: str,
    pinned: bool,
    cwd: str | None = None,
    session_id: str = "",
    agent_id: str = "",
    source: str = "assistant",
) -> str:
    """Insert text into LanceDB + SQLite. Returns chunk_id."""
    cfg = load_config()
    common.claim_store(cfg)
    bad = common.store_mismatch(cfg)
    if bad:
        raise StoreMismatch(f"embed settings differ from the store: {bad}. Fix: {common.MISMATCH_HINT}")
    if cwd is None:
        cwd = os.getcwd()
    proj = project_key(cwd)
    text = scrub(strip_private(text)).strip()

    # Content-hash dedup
    h = content_hash(text)
    meta = open_meta(cwd)
    cur = meta.execute("SELECT id FROM chunks WHERE content_hash = ?", (h,)).fetchone()
    if cur:
        meta.close()
        return cur[0]

    vec = _embed(text, cfg)

    chunk_id = uuid.uuid4().hex
    ts = now_unix()
    tbl = _open_table(cwd, cfg["embed"]["dim"])
    tbl.add([{
        "id": chunk_id,
        "project": proj,
        "session_id": session_id,
        "agent_id": agent_id,
        "kind": kind,
        "ts": ts,
        "text": text,
        "content_hash": h,
        "superseded_by": "",
        "vector": vec,
    }])
    refresh_fts(tbl)

    pin_val = 1 if (kind == "decision" or pinned) else 0
    meta.execute(
        "INSERT INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, "
        "superseded_by, pinned, last_accessed, source, trust, taint) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?)",
        (chunk_id, proj, session_id, agent_id, kind, ts, h, pin_val, ts, *provenance.make(source, cfg).row()),
    )
    meta.commit()
    meta.close()

    # Backlinks (4.2 — no-op if module not yet available)
    _update_backlinks(cwd, proj, chunk_id, vec, cfg)

    return chunk_id


def _remember_core(text: str, args) -> int:
    """Live core write for the user-typed command; anything else becomes a proposal."""
    import core

    cwd = os.getcwd()
    layer = "shared" if args.shared else "global" if args.global_ else "private"
    section = (args.section or {"decision": "decisions", "note": "lessons"}.get(args.kind or "", "decisions"))
    section = section.capitalize()
    try:
        if args.source != "user":
            pid = core.propose(project_key(cwd), "add", "remember", args.source,
                               layer="private" if layer == "shared" else layer, section=section, text=text)
            print(f"proposed {pid or '(already proposed)'}: review with /biss-brain:core review")
            return 0
        it = core.add(cwd, text, section, layer, source="user")
    except core.CoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"core {layer}/{section}: m:{it.id}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Insert an explicit user memory into biss-brain.")
    p.add_argument("text", nargs="+", help="Text to remember")
    p.add_argument(
        "--kind",
        choices=list(VALID_KINDS),
        default=None,
        help="Memory kind. If omitted, classified automatically via Haiku.",
    )
    p.add_argument(
        "--pin",
        action="store_true",
        help="Pin the chunk (immune to decay, ranks higher in retrieval).",
    )
    p.add_argument(
        "--source",
        choices=["user", "assistant"],
        default="assistant",
        help="Provenance. Only the user-typed /remember command passes 'user'.",
    )
    p.add_argument(
        "--core",
        action="store_true",
        help="Write to core memory (injected in full every session) instead of the store.",
    )
    layer = p.add_mutually_exclusive_group()
    layer.add_argument("--shared", action="store_true",
                       help="With --core: the repo's committed .claude/brain/CORE.md (enables it).")
    layer.add_argument("--global", dest="global_", action="store_true",
                       help="With --core: _global.md, injected in every project.")
    p.add_argument(
        "--section",
        choices=["decisions", "conventions", "lessons", "pinned"],
        default=None,
        help="With --core: the section (default: from --kind, else decisions).",
    )
    args = p.parse_args()

    # Scrub before anything else sees the text — including the classifier call.
    text = scrub(strip_private(" ".join(args.text))).strip()
    if not text:
        print("error: text must not be empty", file=sys.stderr)
        return 1

    if args.core or args.shared or args.global_:
        return _remember_core(text, args)

    kind = args.kind
    if kind is None:
        kind = _classify_kind(text)

    try:
        chunk_id = insert_chunk(text=text, kind=kind, pinned=args.pin, source=args.source)
    except StoreMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(chunk_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
