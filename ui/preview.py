"""What the next session in a project starts with, read without side effects.

The SessionStart hook (scripts/inject_context.py) syncs core memory, opens
the store read-write and records what it injected. This builds the same
text from the same functions on a read-only connection, so the preview is
what the hook would inject and looking at it changes nothing.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import common
import inject_context as I
import provenance
import scope

from ui.data import GLOBAL_PROJECT_HASH, _project_dir, open_lance, open_meta_ro


def project_cwd(project_hash: str) -> str | None:
    """The labelled path of a project, if it still hashes to the project."""
    label = _project_dir(project_hash) / "project_path.txt"
    try:
        path = label.read_text().strip()
    except OSError:
        return None
    if not path or not Path(path).is_dir() or common.project_key(path) != project_hash:
        return None
    return path


def _core(cwd: str, cfg: dict) -> tuple[str, int, int]:
    if not cfg.get("core", {}).get("inject", True):
        return "", 0, 0
    try:
        import core

        return core.frame(cwd, cfg)
    except Exception:  # noqa: BLE001  (the hook drops a broken core file the same way)
        return "", 0, 0


def next_session(project_hash: str) -> dict:
    """{status, cwd, core_shown, core_total, memories, text} for the project.

    status is "ok", "global" (the cross-project pool starts no session),
    "out_of_scope" or "store_mismatch" (the hook injects nothing), or
    "unreadable" (a store from before the current schema; the next hook
    run migrates it).
    Without a valid label core memory is unknown and only the store is read.
    """
    out = {"status": "ok", "cwd": None, "core_shown": 0, "core_total": 0, "memories": [], "text": ""}
    if project_hash == GLOBAL_PROJECT_HASH:
        return {**out, "status": "global"}
    cfg = common.load_config()
    cwd = project_cwd(project_hash)
    out["cwd"] = cwd
    if cwd and not scope.in_scope(cwd, cfg):
        return {**out, "status": "out_of_scope"}
    if common.store_mismatch(cfg):
        return {**out, "status": "store_mismatch"}

    core_ctx, out["core_shown"], out["core_total"] = _core(cwd, cfg) if cwd else ("", 0, 0)
    tbl = open_lance(project_hash)
    if tbl is not None and (_project_dir(project_hash) / "meta.sqlite").exists():
        meta = open_meta_ro(project_hash)
        try:
            out["memories"] = I.recent_rows(meta, tbl, project_hash, I.top_k(cfg), cfg)
        except sqlite3.Error:
            return {**out, "status": "unreadable"}
        finally:
            meta.close()
    out["text"] = I.session_start_context(core_ctx, out["memories"])
    for m in out["memories"]:
        m["source_label"] = provenance.label(m.get("_prov") or provenance.IMPORTED)
    return out
