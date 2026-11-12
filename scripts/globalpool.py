"""The shared pool (`_global`): what may enter it and keeping each copy true to its source.

dream copies a fact into the pool (dream._stage_promote_to_global); this module
holds everything around that copy:

  veto(text, cfg, tokens)      deterministic gate: secrets, paths, addresses,
                               project names, length. A vetoed fact stays local.
  find_duplicate / join        a fact another project already pooled joins that
                               row (source_projects) instead of a second copy.
  retract / retract_sources    tombstone a pool copy (forget, or its source left).
  reconcile(cfg)               nightly pull: every live copy is checked against
                               its source row; gone or replaced -> retracted,
                               reworded -> re-copied (text and vector).
  sweep(cfg)                   cross-project conflict pass over new pool rows:
                               a disagreement becomes a `diverges` edge, never a
                               supersede (R2).
  maintain(cfg)                reconcile + sweep + compaction, once per dream.
  share / unshare              the user shares one memory now, or takes it back
                               (UI card, /biss-brain:share); both undo each other.
  unshare_pool / restore_pool  the same from the pool's side (UI pool page).
  source_labels(ids)           where each pool row came from, in one query.

A pool row keeps the id of the row it was copied from. On a pool row
source_project / source_id name that row and source_projects lists every
project that wrote the same fact; on a project row global_id names the pool
row it fed and global_veto why it stays local.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

import common
import lance_filter as F
import provenance
from common import (
    GLOBAL_PROJECT_HASH, compact_store, ensure_schema, fetch_by_ids, now_unix,
    open_or_create_chunks, refresh_fts, tombstone_id, vector_search,
)
from scrub import scrub

log = logging.getLogger("biss-brain.globalpool")

# kinds that never leave their project: unvetted, unconfirmed or per-project digests
NEVER_POOLED = ("quarantine", "hypothesis", "arc", "daily_summary")

_HOME_PATH = re.compile(r"(?:^|[\s\"'`(=:])(?:~/|/Users/|/home/|/root/|/private/|/Volumes/)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_URL = re.compile(r"\b(?:https?|ssh|git)://", re.IGNORECASE)
# folder names that say nothing about which project wrote a fact
_GENERIC_PARTS = frozenset({
    "users", "home", "sources", "source", "projects", "project", "experiments", "documents",
    "desktop", "downloads", "developer", "code", "repos", "work", "src", "app", "apps", "lib",
    "docs", "scripts", "tests", "test", "content", "lessons", "data", "notes", "music", "fonts",
    "pack", "site", "web", "api", "demo", "demos", "pipeline", "marketing", "outreach",
    "plugins", "tools", "server", "client", "config", "assets", "public", "build",
})


def _root() -> Path:
    return common.BRAIN_ROOT  # read at call time: tests point it elsewhere


def pool_dir() -> Path:
    return _root() / "_global"


def _meta(pdir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(pdir / "meta.sqlite"), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _pool_table():
    import lancedb  # noqa: PLC0415

    d = pool_dir() / "lancedb"
    if not d.is_dir():
        return None
    db = lancedb.connect(str(d))
    return db.open_table("chunks") if "chunks" in db.list_tables().tables else None


def _project_table(h: str):
    import lancedb  # noqa: PLC0415

    d = _root() / h / "lancedb"
    if not d.is_dir():
        return None
    db = lancedb.connect(str(d))
    return db.open_table("chunks") if "chunks" in db.list_tables().tables else None


# ── the veto ─────────────────────────────────────────────────────────────

def project_tokens(root: Path | None = None) -> frozenset[str]:
    """Names that tie a text to one of the user's projects.

    The folder name of every store's project_path.txt and the home folder
    name, lowercased, at least 4 characters, minus generic folder names. A
    compound name also contributes its first part ("atlas" for atlas-api):
    notes call a project by its stem.
    """
    root = root or _root()
    out: set[str] = set()
    names = [Path.home().name]
    try:
        for d in root.iterdir():
            label = d / "project_path.txt"
            if d.name != "_global" and label.is_file():
                names.append(Path(label.read_text().strip()).name)
    except OSError:
        pass
    for n in names:
        n = n.strip().lower()
        if n.startswith("__"):
            continue
        for t in (n, re.split(r"[-_.]", n, maxsplit=1)[0]):
            if len(t) >= 4 and t not in _GENERIC_PARTS:
                out.add(t)
    return frozenset(out)


def _token_hit(text: str, tokens: frozenset[str]) -> bool:
    if not tokens:
        return False
    pat = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(t) for t in sorted(tokens)) + r")(?![\w-])")
    return bool(pat.search(text.lower()))


def veto(text: str, cfg: dict, tokens: frozenset[str] = frozenset(), *, shared: bool = False) -> str:
    """Why *text* must stay in its project, or "" when it may be pooled.

    A fact the user shared (`shared`) is checked for secrets only: they chose
    to share it. Anything else must also be short and free of paths, mail
    addresses, IP addresses, URLs and project names.
    """
    if scrub(text) != text:
        return "secret"
    if shared:
        return ""
    if len(text) > int(cfg.get("cross_project", {}).get("veto_max_chars", 400)):
        return "length"
    if _HOME_PATH.search(text):
        return "path"
    if _EMAIL.search(text):
        return "email"
    if _IPV4.search(text):
        return "address"
    if _URL.search(text):
        return "url"
    if _token_hit(text, tokens):
        return "project-name"
    return ""


# ── dedup ────────────────────────────────────────────────────────────────

def find_duplicate(tbl, vec, cfg: dict) -> str | None:
    """Id of a live row of pool table *tbl* at least dedup_cosine close to *vec*, else None."""
    if tbl is None or vec is None:
        return None
    try:
        hits = vector_search(tbl, list(vec), F.LIVE, 1)
    except Exception:  # noqa: BLE001
        return None
    gate = float(cfg.get("cross_project", {}).get("dedup_cosine", 0.94))
    return hits[0]["id"] if hits and hits[0]["_cosine"] >= gate else None


def join(gpdir: Path, gid: str, proj: str, src_id: str, src_pdir: Path) -> None:
    """Record that *proj* wrote the fact pool row *gid* (in pool dir *gpdir*) holds."""
    g = _meta(gpdir)
    try:
        row = g.execute("SELECT source_projects FROM chunks WHERE id = ? AND project = ?",
                        (gid, GLOBAL_PROJECT_HASH)).fetchone()
        if row is None:
            return
        have = _projects(row["source_projects"])
        if proj not in have:
            g.execute("UPDATE chunks SET source_projects = ? WHERE id = ? AND project = ?",
                      (json.dumps(have + [proj]), gid, GLOBAL_PROJECT_HASH))
            g.commit()
    finally:
        g.close()
    s = _meta(src_pdir)
    try:
        s.execute("UPDATE chunks SET cross_eligible = 1, global_id = ? WHERE id = ? AND project = ?",
                  (gid, src_id, proj))
        s.commit()
    finally:
        s.close()


def _projects(raw: str | None) -> list[str]:
    try:
        v = json.loads(raw or "[]")
        return [p for p in v if isinstance(p, str)] if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


# ── retraction ───────────────────────────────────────────────────────────

def _release(h: str, gid: str, veto_reason: str = "", keep_link: bool = False) -> None:
    """Clear the link from project *h*'s rows to pool row *gid*.

    The row may be promoted again (unforget, a re-share) unless *veto_reason*
    marks it as kept local. With *keep_link* global_id still names *gid*, so
    restore_pool can find the rows again.
    """
    pdir = _root() / h
    if not (pdir / "meta.sqlite").exists():
        return
    c = _meta(pdir)
    try:
        c.execute("UPDATE chunks SET cross_eligible = 0, global_id = ?, global_veto = ? "
                  "WHERE project = ? AND (global_id = ? OR (id = ? AND global_id = ''))",
                  (gid if keep_link else "", veto_reason, h, gid, gid))
        c.commit()
    finally:
        c.close()


def retract(gid: str, *, veto_reason: str = "", keep_link: bool = False) -> bool:
    """Tombstone pool row *gid* and release every project row that fed it.

    The pool copy is soft-deleted like any forget, and only ever in the pool
    (project = GLOBAL_PROJECT_HASH): no project's own row is superseded here.
    """
    pdir = pool_dir()
    if not (pdir / "meta.sqlite").exists():
        return False
    tomb = tombstone_id()
    g = _meta(pdir)
    try:
        row = g.execute("SELECT source_project, source_projects FROM chunks WHERE id = ? AND project = ?",
                        (gid, GLOBAL_PROJECT_HASH)).fetchone()
        if row is None:
            return False
        cur = g.execute("UPDATE chunks SET superseded_by = ? WHERE id = ? AND project = ? AND superseded_by = ''",
                        (tomb, gid, GLOBAL_PROJECT_HASH))
        g.commit()
    finally:
        g.close()
    if cur.rowcount:
        tbl = _pool_table()
        if tbl is not None:
            try:
                tbl.update(where=F.eq("id", gid), values={"superseded_by": tomb})
            except Exception as e:  # noqa: BLE001
                log.warning("pool: lance tombstone failed for %s: %s", gid[:8], e)
    for h in {row["source_project"], *_projects(row["source_projects"])} - {""}:
        _release(h, gid, veto_reason, keep_link)
    return bool(cur.rowcount)


def retract_sources(proj: str, ids: list[str]) -> int:
    """After a forget in *proj*: take the forgotten rows out of the pool now.

    A pool row copied from one of them is tombstoned (its other contributors are
    released, so they may promote their own copy next night); a pool row the
    project only joined loses *proj* from source_projects.
    """
    pdir = pool_dir()
    if not ids or not (pdir / "meta.sqlite").exists() or not (_root() / proj / "meta.sqlite").exists():
        return 0
    s = _meta(_root() / proj)
    try:
        q = ",".join("?" * len(ids))
        linked = {r["id"]: r["global_id"] for r in s.execute(
            f"SELECT id, global_id FROM chunks WHERE project = ? AND id IN ({q})", (proj, *ids))}
    finally:
        s.close()
    g = _meta(pdir)
    try:
        q = ",".join("?" * len(ids))
        # pool rows copied from these ids (older copies carry no source_project)
        owned = [r["id"] for r in g.execute(
            f"SELECT id FROM chunks WHERE project = ? AND superseded_by = '' AND "
            f"((source_project = ? AND source_id IN ({q})) OR (source_project = '' AND id IN ({q})))",
            (GLOBAL_PROJECT_HASH, proj, *ids, *ids))]
    finally:
        g.close()
    n = sum(retract(gid) for gid in owned)
    for sid, gid in linked.items():
        if gid and gid not in owned:
            _leave(gid, proj)
            _release_one(proj, sid)
            n += 1
    return n


def _leave(gid: str, proj: str) -> None:
    g = _meta(pool_dir())
    try:
        row = g.execute("SELECT source_projects FROM chunks WHERE id = ? AND project = ?",
                        (gid, GLOBAL_PROJECT_HASH)).fetchone()
        if row is not None:
            left = [p for p in _projects(row["source_projects"]) if p != proj]
            g.execute("UPDATE chunks SET source_projects = ? WHERE id = ? AND project = ?",
                      (json.dumps(left), gid, GLOBAL_PROJECT_HASH))
            g.commit()
    finally:
        g.close()


def _release_one(proj: str, sid: str) -> None:
    c = _meta(_root() / proj)
    try:
        c.execute("UPDATE chunks SET cross_eligible = 0, global_id = '' WHERE id = ? AND project = ?",
                  (sid, proj))
        c.commit()
    finally:
        c.close()


# ── user control ─────────────────────────────────────────────────────────

def share(proj: str, cid: str, cfg: dict) -> tuple[str, dict]:
    """The user shares memory *cid* of project *proj* with every project, now.

    It is marked pinned and scope=general, which makes it a shared fact (the
    veto checks it for secrets only), and promoted at once instead of at the
    next dream. Returns (status, prior pinned/scope, for unshare to put back):
      "shared"    in the pool (copied, revived or joined to an equal row)
      "saved"     marked, but sharing is off: the next dream with it on pools it
      "missing"   no live row; "untrusted" tool or web output; "kind" a kind
                  that never leaves its project; or the veto's reason ("secret")
    Anything but shared/saved leaves the row as it was.
    """
    pdir = _root() / proj
    if not (pdir / "meta.sqlite").exists():
        return "missing", {}
    c = _meta(pdir)
    try:
        row = c.execute(f"SELECT pinned, scope, kind, {provenance.trusted_sql()} AS trusted FROM chunks "
                        "WHERE id = ? AND project = ? AND superseded_by = ''", (cid, proj)).fetchone()
        if row is None:
            return "missing", {}
        prior = {"pinned": int(row["pinned"] or 0), "scope": row["scope"] or "project"}
        if row["kind"] in NEVER_POOLED:
            return "kind", prior
        if not row["trusted"]:
            return "untrusted", prior
        c.execute("UPDATE chunks SET pinned = 1, scope = 'general', global_veto = '' WHERE id = ? AND project = ?",
                  (cid, proj))
        c.commit()
    finally:
        c.close()
    xp = cfg.get("cross_project", {})
    if not xp.get("enabled", False) or xp.get("promote", "scope") == "off":
        return "saved", prior
    import dream  # noqa: PLC0415 — dream imports this module

    dream._stage_promote_to_global(pdir, cfg, only={cid})
    c = _meta(pdir)
    try:
        after = c.execute("SELECT cross_eligible, global_veto FROM chunks WHERE id = ? AND project = ?",
                          (cid, proj)).fetchone()
    finally:
        c.close()
    if after["cross_eligible"]:
        return "shared", prior
    _put_back(proj, cid, prior, after["global_veto"])
    return after["global_veto"] or "missing", prior


def unshare(proj: str, cid: str, prior: dict | None = None) -> bool:
    """Take memory *cid* of *proj* out of the pool and keep it local.

    Its pool copy is retracted (or it leaves a row another project also wrote)
    and global_veto = "user" stops the next dream pooling it again. *prior*
    (from share) puts pinned and scope back as they were before sharing.
    """
    if not (_root() / proj / "meta.sqlite").exists():
        return False
    retract_sources(proj, [cid])
    return _put_back(proj, cid, prior, "user")


def _put_back(proj: str, cid: str, prior: dict | None, veto_reason: str) -> bool:
    sets, args = ["cross_eligible = 0", "global_id = ''", "global_veto = ?"], [veto_reason]
    if prior and prior.get("scope") in ("project", "general"):
        sets += ["pinned = ?", "scope = ?"]
        args += [1 if prior.get("pinned") else 0, prior["scope"]]
    c = _meta(_root() / proj)
    try:
        cur = c.execute(f"UPDATE chunks SET {', '.join(sets)} WHERE id = ? AND project = ?", (*args, cid, proj))
        c.commit()
        return bool(cur.rowcount)
    finally:
        c.close()


def unshare_pool(gid: str) -> bool:
    """Take pool row *gid* out of the pool; every project that wrote it keeps it local.

    Like `forget --global`, and undone by restore_pool: the project rows keep
    their link to *gid*.
    """
    return retract(gid, veto_reason="user", keep_link=True)


def restore_pool(gid: str) -> bool:
    """Undo unshare_pool: the row is live again and its writers are linked to it.

    Only a tombstone is cleared, and only rows still vetoed by the user are
    linked; a writer forgotten since leaves at the next reconcile.
    """
    pdir = pool_dir()
    if not (pdir / "meta.sqlite").exists():
        return False
    g = _meta(pdir)
    try:
        row = g.execute("SELECT source_project, source_projects, superseded_by FROM chunks "
                        "WHERE id = ? AND project = ?", (gid, GLOBAL_PROJECT_HASH)).fetchone()
        if row is None or not row["superseded_by"] or g.execute(
                "SELECT 1 FROM chunks WHERE id = ?", (row["superseded_by"],)).fetchone():
            return False
        g.execute("UPDATE chunks SET superseded_by = '' WHERE id = ? AND project = ?", (gid, GLOBAL_PROJECT_HASH))
        g.commit()
    finally:
        g.close()
    tbl = _pool_table()
    if tbl is not None:
        try:
            tbl.update(where=F.eq("id", gid), values={"superseded_by": ""})
        except Exception as e:  # noqa: BLE001
            log.warning("pool: lance restore failed for %s: %s", gid[:8], e)
    for h in {row["source_project"], *_projects(row["source_projects"])} - {""}:
        if not (_root() / h / "meta.sqlite").exists():
            continue
        c = _meta(_root() / h)
        try:
            c.execute("UPDATE chunks SET cross_eligible = 1, global_veto = '' WHERE project = ? "
                      "AND global_id = ? AND global_veto = 'user' AND superseded_by = ''", (h, gid))
            c.commit()
        finally:
            c.close()
    return True


# ── nightly upkeep ───────────────────────────────────────────────────────

def reconcile(cfg: dict, dry_run: bool = False) -> dict:
    """Check every live pool copy against the row it came from.

    Source gone or no longer live (forgotten, replaced, decayed): the copy is
    retracted. Source reworded (reconsolidation edits the text in place): the
    copy takes the new text and vector, or is retracted when the new text fails
    the veto. A contributor whose row went away leaves source_projects.
    """
    out = {"checked": 0, "retracted": 0, "recopied": 0, "left": 0}
    pdir = pool_dir()
    if not (pdir / "meta.sqlite").exists():
        return out
    g = _meta(pdir)
    try:
        rows = g.execute("SELECT id, source_project, source_id, source_projects FROM chunks "
                         "WHERE project = ? AND superseded_by = '' AND source_id != ''",
                         (GLOBAL_PROJECT_HASH,)).fetchall()
    finally:
        g.close()
    if not rows:
        return out
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["source_project"], []).append(r)
    ptbl = _pool_table()
    tokens = project_tokens()
    for h, group in by_src.items():
        live, texts = _source_state(h, [r["source_id"] for r in group])
        for r in group:
            out["checked"] += 1
            sid, gid = r["source_id"], r["id"]
            if sid not in live:
                out["retracted"] += 1
                if not dry_run:
                    retract(gid)
                continue
            src = texts.get(sid)
            if src is None or ptbl is None:
                continue
            cur = fetch_by_ids(ptbl, [gid])
            if cur and cur[0].get("text") == src["text"]:
                continue
            reason = veto(src["text"], cfg, tokens, shared=live[sid])
            if reason:
                out["retracted"] += 1
                if not dry_run:
                    retract(gid, veto_reason=reason)
                continue
            out["recopied"] += 1
            if not dry_run:
                ptbl.update(where=F.eq("id", gid), values={"text": src["text"], "vector": list(src["vector"])})
                g = _meta(pdir)
                try:
                    g.execute("UPDATE chunks SET content_hash = ? WHERE id = ? AND project = ?",
                              (src.get("content_hash", ""), gid, GLOBAL_PROJECT_HASH))
                    g.commit()
                finally:
                    g.close()
    out["left"] = _prune_contributors(rows, dry_run)
    if out["recopied"] and ptbl is not None and not dry_run:
        refresh_fts(ptbl)
    return out


def _source_state(h: str, ids: list[str]) -> tuple[dict[str, bool], dict[str, dict]]:
    """Live ids among *ids* in project *h* (value: shared by the user) and their lance rows."""
    pdir = _root() / h
    if not ids or not (pdir / "meta.sqlite").exists():
        return {}, {}
    c = _meta(pdir)
    try:
        q = ",".join("?" * len(ids))
        live = {r["id"]: bool(r["pinned"]) and r["scope"] == "general" for r in c.execute(
            f"SELECT id, pinned, scope FROM chunks WHERE project = ? AND superseded_by = '' AND id IN ({q})",
            (h, *ids))}
    finally:
        c.close()
    tbl = _project_table(h)
    texts = {r["id"]: r for r in fetch_by_ids(tbl, list(live))} if tbl is not None and live else {}
    return live, texts


def _prune_contributors(rows: list, dry_run: bool) -> int:
    n = 0
    for r in rows:
        others = [p for p in _projects(r["source_projects"]) if p != r["source_project"]]
        for p in others:
            pdir = _root() / p
            alive = False
            if (pdir / "meta.sqlite").exists():
                c = _meta(pdir)
                try:
                    alive = c.execute("SELECT 1 FROM chunks WHERE project = ? AND global_id = ? "
                                      "AND superseded_by = ''", (p, r["id"])).fetchone() is not None
                finally:
                    c.close()
            if not alive:
                n += 1
                if not dry_run:
                    _leave(r["id"], p)
    return n


def sweep(cfg: dict, dry_run: bool = False, limit: int = 20) -> dict:
    """Judge new pool rows against their nearest pool neighbour from another project.

    Pool rows are the same fact seen from different projects, so a clash is a
    divergence (each project keeps its own truth), recorded as a `diverges`
    edge on both rows. A same-project clash is left to that project's own
    sweep, whose verdict reaches the pool through reconcile.
    """
    out = {"checked": 0, "diverges": 0}
    pdir = pool_dir()
    tbl = _pool_table()
    if tbl is None or not (pdir / "meta.sqlite").exists():
        return out
    sim = float(cfg.get("dream", {}).get("stage1", {}).get("conflict_sim_threshold", 0.85))
    g = _meta(pdir)
    try:
        fresh = g.execute("SELECT id, source_project FROM chunks WHERE project = ? AND superseded_by = '' "
                          "AND last_dream_ts = 0 ORDER BY ts LIMIT ?", (GLOBAL_PROJECT_HASH, limit)).fetchall()
        src = {r["id"]: r["source_project"] for r in g.execute(
            "SELECT id, source_project FROM chunks WHERE project = ?", (GLOBAL_PROJECT_HASH,))}
    finally:
        g.close()
    from backlinks import _add_typed_edge  # noqa: PLC0415
    from judge import judge_with_project_context  # noqa: PLC0415

    now = now_unix()
    for r in fresh:
        mine = fetch_by_ids(tbl, [r["id"]])
        verdict_ok = True
        if mine and mine[0].get("vector") is not None:
            try:
                near = vector_search(tbl, list(mine[0]["vector"]), F.and_(F.LIVE, F.ne("id", r["id"])), 3)
            except Exception:  # noqa: BLE001
                near = []
            near = [n for n in near if n["_cosine"] >= sim and src.get(n["id"], "") != r["source_project"]]
            if near:
                out["checked"] += 1
                if dry_run:
                    continue
                res = judge_with_project_context(
                    {"text": mine[0]["text"], "project": r["source_project"]},
                    {"text": near[0]["text"], "project": src.get(near[0]["id"], "")},
                    same_project=False, cfg=cfg)
                verdict_ok = res.get("verdict") is not None
                if res.get("verdict") == "DIVERGES":
                    out["diverges"] += 1
                    g = _meta(pdir)
                    try:
                        _add_typed_edge(g, r["id"], near[0]["id"], "diverges", 1.0, now, project=GLOBAL_PROJECT_HASH)
                        _add_typed_edge(g, near[0]["id"], r["id"], "diverges", 1.0, now, project=GLOBAL_PROJECT_HASH)
                        g.commit()
                    finally:
                        g.close()
        if verdict_ok and not dry_run:  # a failed judge call is retried next night
            g = _meta(pdir)
            try:
                g.execute("UPDATE chunks SET last_dream_ts = ? WHERE id = ? AND project = ?",
                          (now, r["id"], GLOBAL_PROJECT_HASH))
                g.commit()
            finally:
                g.close()
    return out


def maintain(cfg: dict, dry_run: bool = False) -> dict:
    """The pool's share of a dream: reconcile, sweep, compact."""
    if not cfg.get("cross_project", {}).get("enabled", False) or not (pool_dir() / "meta.sqlite").exists():
        return {}
    out = {"reconcile": reconcile(cfg, dry_run)}
    if cfg.get("dream", {}).get("stage1", {}).get("conflict_sweep", True):
        out["sweep"] = sweep(cfg, dry_run)
    tbl = _pool_table()
    if tbl is not None and not dry_run:
        compact_store(tbl)
    return out


# ── reading ──────────────────────────────────────────────────────────────

def widely_written(min_projects: int = 2) -> list[dict]:
    """Live, trusted pool rows that at least *min_projects* projects wrote
    (the source and every joined one), newest first: {id, kind, source,
    trust, taint, projects}. Evidence enough to suggest one for global core."""
    path = pool_dir() / "meta.sqlite"
    if not path.exists():
        return []
    c = _meta(pool_dir())
    try:
        rows = c.execute("SELECT id, kind, source, trust, taint, source_project, source_projects FROM chunks "
                         f"WHERE project = ? AND superseded_by = '' AND {provenance.trusted_sql()} "
                         "ORDER BY ts DESC", (GLOBAL_PROJECT_HASH,)).fetchall()
    finally:
        c.close()
    out = []
    for r in rows:
        projects = {r["source_project"], *_projects(r["source_projects"])} - {""}
        if len(projects) >= min_projects:
            out.append({**{k: r[k] for k in ("id", "kind", "source", "trust", "taint")}, "projects": projects})
    return out


def source_labels(ids: list[str]) -> dict[str, str]:
    """{pool id: path of the project it came from}; "__global__" when unknown."""
    path = pool_dir() / "meta.sqlite"
    out = {i: "__global__" for i in ids}
    if not ids or not path.exists():
        return out
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        q = ",".join("?" * len(ids))
        rows = c.execute(f"SELECT id, source_project FROM chunks WHERE id IN ({q})", ids).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        c.close()
    for cid, h in rows:
        if not h:
            continue
        label = _root() / h / "project_path.txt"
        try:
            out[cid] = (label.read_text().strip() if label.is_file() else "") or h
        except OSError:
            out[cid] = h
    return out


def ensure_pool_table(dim: int):
    """The pool's lance table, created on first write."""
    d = pool_dir() / "lancedb"
    d.mkdir(parents=True, exist_ok=True)
    return open_or_create_chunks(d, dim)
