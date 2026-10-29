"""Core memory: a small curated set of memories, injected in full every session.

Layers (decision D4, layered):
  global   ~/.claude/brain/core/_global.md         personal conventions, every project
  private  ~/.claude/brain/core/<repo>.md           this repo, never leaves the machine (default)
  shared   <repo>/.claude/brain/CORE.md             committed with the code; injected only after
                                                    the user opts in for that repo

`~/.claude/brain/core/` is its own git repo, auto-committed on every change.
The shared file is never committed by us — it lives in the user's repo.

File format (the file is the source of truth; hand edits win):

    ---
    repo: /abs/path/to/repo
    remote: https://host/owner/repo.git
    shared: false
    ---
    # Core memory: repo

    ## Decisions
    - Use uv, not pip. <!-- m:ab12cd34 2026-10-26 user -->

Sections are Decisions, Conventions, Lessons, Pinned (others are kept as-is).
A bullet without an id comment gets one derived from its text, written back
on the next programmatic write. Deleting a line retires the memory.

Writes: only user-typed paths write live core (`/remember --core`, `/core
accept`, hand edits). Everything the model or the dream pass suggests goes
to a proposal queue for `/core review` — core skips the injection taint
filter, so it must never be writable by text the model was shown.

Index (`~/.claude/brain/core.sqlite`, outside the git repo): one row per
version of each item (id, layer, text, source, valid_from, invalid_at), so an
item keeps its provenance, validity and history; plus the proposal queue.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import common
import provenance
from scrub import scrub

SECTIONS = ("Decisions", "Conventions", "Lessons", "Pinned")
LAYERS = ("shared", "private", "global")  # injection order: project layers first
SOURCES = (*provenance.SOURCES, "repo")
_ID = re.compile(r"<!--\s*m:([0-9a-f]{8})(?:\s+(\d{4}-\d{2}-\d{2}))?(?:\s+([a-z]+))?\s*-->")
_GIT_TIMEOUT = 5


# ── paths (resolved per call: tests swap common.BRAIN_ROOT) ──────────────

def core_dir() -> Path:
    return common.BRAIN_ROOT / "core"


def index_path() -> Path:
    return common.BRAIN_ROOT / "core.sqlite"


def global_file() -> Path:
    return core_dir() / "_global.md"


def shared_file(repo: Path) -> Path:
    return repo / ".claude" / "brain" / "CORE.md"


# ── repo identity ────────────────────────────────────────────────────────

def _git(args: list[str], cwd: Path | str, strip: bool = True) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=_GIT_TIMEOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() if strip else r.stdout


def clean_remote(url: str | None) -> str:
    """Remote URL without credentials (https://token@host/... → https://host/...)."""
    if not url:
        return ""
    if "://" in url:
        u = urlsplit(url)
        host = u.hostname or ""
        if u.port:
            host = f"{host}:{u.port}"
        return urlunsplit((u.scheme, host, u.path, "", ""))
    return url.split("@", 1)[-1] if re.match(r"^[^@/]+@[^:]+:", url) else url


def repo_root(cwd: str | os.PathLike) -> Path:
    """Main worktree root of the repo containing cwd (any worktree or subdir); realpath otherwise."""
    return common.repo_root(cwd)


def repo_remote(root: Path) -> str:
    return clean_remote(_git(["config", "--get", "remote.origin.url"], root))


# ── model and parsing ────────────────────────────────────────────────────

@dataclass
class Item:
    id: str
    section: str
    text: str
    date: str
    source: str


@dataclass
class CoreFile:
    path: Path
    meta: dict[str, str] = field(default_factory=dict)
    title: str = ""
    # (section, [lines]) in file order; "" holds lines before the first section
    blocks: list[tuple[str, list[str]]] = field(default_factory=list)

    def items(self) -> list[Item]:
        out = []
        for section, lines in self.blocks:
            if not section:
                continue
            for line in lines:
                it = parse_bullet(line, section, self._default_date())
                if it:
                    out.append(it)
        return out

    def _default_date(self) -> str:
        try:
            return time.strftime("%Y-%m-%d", time.localtime(self.path.stat().st_mtime))
        except OSError:
            return time.strftime("%Y-%m-%d")


def item_id(section: str, text: str) -> str:
    return hashlib.sha256(f"{section}\n{text}".encode()).hexdigest()[:8]


def parse_bullet(line: str, section: str, default_date: str) -> Item | None:
    # Only top-level bullets are items; indented lines stay the user's text.
    if not line.startswith(("- ", "* ")):
        return None
    body = line[2:].strip()
    m = _ID.search(body)
    text = _ID.sub("", body).strip()
    if not text:
        return None
    if m:
        # an unknown source tag never reads as the most trusted one
        src = m.group(3) if m.group(3) in SOURCES else ("user" if m.group(3) is None else "imported")
        return Item(m.group(1), section, text, m.group(2) or default_date, src)
    # A hand-written line: the user wrote it.
    return Item(item_id(section, text), section, text, default_date, "user")


def render_bullet(it: Item) -> str:
    return f"- {it.text} <!-- m:{it.id} {it.date} {it.source} -->"


def load(path: Path) -> CoreFile:
    cf = CoreFile(path)
    try:
        raw = path.read_text()
    except (OSError, UnicodeDecodeError):
        return cf
    lines = raw.splitlines()
    i = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                for kv in lines[1:j]:
                    k, _, v = kv.partition(":")
                    if k.strip():
                        cf.meta[k.strip()] = v.strip()
                i = j + 1
                break
    cur: tuple[str, list[str]] = ("", [])
    cf.blocks.append(cur)
    for line in lines[i:]:
        if line.startswith("# ") and not cf.title:
            cf.title = line[2:].strip()
            continue
        if line.startswith("## "):
            cur = (line[3:].strip(), [])
            cf.blocks.append(cur)
            continue
        cur[1].append(line)
    return cf


def dump(cf: CoreFile) -> str:
    out = []
    if cf.meta:
        out += ["---", *(f"{k}: {v}" for k, v in cf.meta.items()), "---"]
    if cf.title:
        out.append(f"# {cf.title}")
    for section, lines in cf.blocks:
        if section:
            if out and out[-1].strip():
                out.append("")
            out.append(f"## {section}")
        # every bullet carries its id once written by us
        for line in lines:
            it = parse_bullet(line, section, time.strftime("%Y-%m-%d")) if section else None
            out.append(render_bullet(it) if it else line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _new(path: Path, title: str, meta: dict[str, str]) -> CoreFile:
    cf = CoreFile(path, dict(meta), title, [("", [])])
    for s in SECTIONS:
        cf.blocks.append((s, []))
    return cf


# ── layer files ──────────────────────────────────────────────────────────

def _private_candidates() -> list[Path]:
    d = core_dir()
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.md") if not p.name.startswith("_") and not p.name.endswith(".proposed.md"))


def private_file(cwd: str | os.PathLike, create: bool = False) -> Path | None:
    """This repo's private core file, re-linked after a move by remote URL."""
    root = repo_root(cwd)
    files = [(p, load(p)) for p in _private_candidates()]
    for p, cf in files:
        if cf.meta.get("repo") == str(root):
            return p
    remote = repo_remote(root)
    if remote:
        for p, cf in files:
            old = cf.meta.get("repo", "")
            if cf.meta.get("remote") == remote and not (old and Path(old).exists()):
                cf.meta["repo"] = str(root)
                _write(p, dump(cf))
                commit(f"relink {p.name} to moved repo")
                return p
    if not create:
        return None
    name = re.sub(r"[^A-Za-z0-9._-]", "-", root.name).strip(".-") or "repo"
    p = core_dir() / f"{name}.md"
    if p.exists():
        p = core_dir() / f"{name}-{hashlib.sha256(str(root).encode()).hexdigest()[:6]}.md"
    cf = _new(p, f"Core memory: {root.name}",
              {"repo": str(root), "remote": remote, "project": common.project_key(cwd), "shared": "false"})
    _write(p, dump(cf))
    commit(f"create {p.name}")
    return p


def shared_enabled(cwd: str | os.PathLike) -> bool:
    p = private_file(cwd)
    return bool(p) and load(p).meta.get("shared", "false").lower() == "true"


def enable_shared(cwd: str | os.PathLike, on: bool = True) -> None:
    p = private_file(cwd, create=True)
    cf = load(p)
    cf.meta["shared"] = "true" if on else "false"
    _write(p, dump(cf))
    commit(f"{'enable' if on else 'disable'} shared layer for {p.name}")


def layer_file(cwd: str | os.PathLike, layer: str, create: bool = False) -> Path | None:
    if layer == "global":
        g = global_file()
        if create and not g.exists():
            _write(g, dump(_new(g, "Core memory: global", {})))
            commit("create _global.md")
        return g if g.exists() else None
    if layer == "private":
        return private_file(cwd, create=create)
    if layer == "shared":
        p = shared_file(repo_root(cwd))
        if create and not p.exists():
            _write(p, dump(_new(p, "Core memory (shared with this repo)", {})))
        return p if p.exists() else None
    raise ValueError(f"unknown layer {layer!r}")


# ── git ──────────────────────────────────────────────────────────────────

_GITIGNORE = "*.tmp\n*.proposed.md\n"


def _ensure_repo() -> bool:
    d = core_dir()
    d.mkdir(parents=True, exist_ok=True)
    if not (d / ".git").exists():
        if _git(["init", "-q"], d) is None:
            return False
        (d / ".gitignore").write_text(_GITIGNORE)
    return True


def _git_commit_args(message: str) -> list[str]:
    # Identity and signing per command: never read or change the user's git config.
    return ["-c", "user.name=biss-brain", "-c", "user.email=biss-brain@localhost",
            "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-m", message]


def _changed(d: Path) -> list[str]:
    """Paths with uncommitted changes in the core dir (porcelain v1, -z)."""
    raw = _git(["status", "--porcelain", "-z", "--untracked-files=all"], d, strip=False) or ""
    out, entries = [], raw.split("\0")
    i = 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if len(e) < 4:
            continue
        out.append(e[3:])
        if e[0] in "RC":
            i += 1  # the rename source follows
    return out


def withheld() -> list[str]:
    """Changed core files not committed because they hold a secret-shaped line."""
    d = core_dir()
    if not (d / ".git").exists():
        return []
    return [n for n in _changed(d) if (d / n).suffix == ".md" and (d / n).exists() and unscrubbed(d / n)]


def commit(message: str) -> bool:
    """Commit the core dir, leaving out any file with a secret-shaped line.

    Git history is permanent, so a hand-pasted secret is never staged, by
    this commit or a later one; doctor points at the file instead. Never
    touches the user's repos.
    """
    if not _ensure_repo():
        return False
    d = core_dir()
    held = set(withheld())
    paths = [n for n in _changed(d) if n not in held]
    if not paths:
        return False
    if _git(["add", "-A", "--", *paths], d) is None:
        return False
    if not _git(["diff", "--cached", "--name-only"], d):
        return False
    return _git(_git_commit_args(message), d) is not None


def unscrubbed(path: Path) -> list[int]:
    """1-based line numbers the secret scrubber would change.

    Every line but the frontmatter (repo path and remote are ours, and a long
    path trips the entropy pass). Id comments are ignored.
    """
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    start = 0
    if lines and lines[0].strip() == "---":
        start = next((j + 1 for j in range(1, len(lines)) if lines[j].strip() == "---"), 0)
    out = []
    for i, line in enumerate(lines[start:], start + 1):
        text = _ID.sub("", line)
        if scrub(text) != text:
            out.append(i)
    return out


def commit_hand_edits() -> str:
    """Commit hand edits in the core dir, except files holding a secret.

    Returns "committed", "clean" or "skipped:<file>" (doctor reports it; the
    user's line is never rewritten).
    """
    if not (core_dir() / ".git").exists():
        return "clean"
    held = withheld()
    done = commit("hand edit")
    if held:
        return f"skipped:{held[0]}"
    return "committed" if done else "clean"


# ── index (provenance, validity, history) ────────────────────────────────

def open_index() -> sqlite3.Connection:
    common.BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_path()), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS core_items (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            mid TEXT NOT NULL, file TEXT NOT NULL, layer TEXT NOT NULL,
            section TEXT NOT NULL, text TEXT NOT NULL, source TEXT NOT NULL,
            date TEXT, valid_from INTEGER NOT NULL, invalid_at INTEGER
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_core_items_live ON core_items(file, invalid_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_core_items_mid ON core_items(mid)")
    conn.execute("CREATE TABLE IF NOT EXISTS core_files (file TEXT PRIMARY KEY, mtime REAL NOT NULL)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS core_proposals (
            id TEXT PRIMARY KEY, project TEXT NOT NULL, layer TEXT NOT NULL,
            op TEXT NOT NULL DEFAULT 'add', section TEXT NOT NULL DEFAULT '', text TEXT NOT NULL DEFAULT '',
            target TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL, source TEXT NOT NULL,
            chunk_id TEXT NOT NULL DEFAULT '', ts INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'open'
        )""")
    return conn


def sync_file(conn: sqlite3.Connection, path: Path, layer: str, force: bool = False) -> bool:
    """Re-read *path* into the index if its mtime changed. Returns True if re-read."""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    row = conn.execute("SELECT mtime FROM core_files WHERE file = ?", (key,)).fetchone()
    if not force and row is not None and mtime is not None and row[0] == mtime:
        return False
    now = int(time.time())
    items = {it.id: it for it in load(path).items()} if mtime is not None else {}
    live = {r["mid"]: r for r in conn.execute(
        "SELECT rowid, mid, text, section FROM core_items WHERE file = ? AND invalid_at IS NULL", (key,))}
    for mid, r in live.items():
        it = items.get(mid)
        if it is None or it.text != r["text"] or it.section != r["section"]:
            conn.execute("UPDATE core_items SET invalid_at = ? WHERE rowid = ?", (now, r["rowid"]))
    for mid, it in items.items():
        r = live.get(mid)
        if r is None or it.text != r["text"] or it.section != r["section"]:
            conn.execute(
                "INSERT INTO core_items (mid, file, layer, section, text, source, date, valid_from) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, key, layer, it.section, it.text, it.source, it.date, now),
            )
    if mtime is None:
        conn.execute("DELETE FROM core_files WHERE file = ?", (key,))
    else:
        conn.execute("INSERT OR REPLACE INTO core_files (file, mtime) VALUES (?, ?)", (key, mtime))
    conn.commit()
    return True


def history(conn: sqlite3.Connection, mid: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM core_items WHERE mid = ? ORDER BY valid_from DESC, rowid DESC", (mid,)).fetchall()


# ── layers for a session ─────────────────────────────────────────────────

def layers_for(cwd: str | os.PathLike) -> list[tuple[str, Path]]:
    """(layer, file) to inject, in order. Shared only once the user opted in."""
    out = []
    root = repo_root(cwd)
    if shared_enabled(cwd) and shared_file(root).exists():
        out.append(("shared", shared_file(root)))
    p = private_file(cwd)
    if p:
        out.append(("private", p))
    if global_file().exists():
        out.append(("global", global_file()))
    return out


def sync(cwd: str | os.PathLike) -> None:
    """Best effort: commit hand edits, then re-index changed layer files. No network, no embed."""
    commit_hand_edits()
    conn = open_index()
    try:
        for layer, path in layers_for(cwd):
            sync_file(conn, path, layer)
    finally:
        conn.close()


def count(cwd: str | os.PathLike) -> int:
    return sum(len(load(p).items()) for _, p in layers_for(cwd))


# ── writes ───────────────────────────────────────────────────────────────

class CoreError(ValueError):
    pass


def add(cwd: str | os.PathLike, text: str, section: str = "Decisions", layer: str = "private",
        source: str = "user") -> Item:
    """Write an item to live core. Callers are user-typed paths only."""
    if section not in SECTIONS:
        raise CoreError(f"unknown section {section!r}; one of {', '.join(SECTIONS)}")
    if layer not in LAYERS:
        raise CoreError(f"unknown layer {layer!r}; one of {', '.join(LAYERS)}")
    text = " ".join(scrub(text).split())
    if not text:
        raise CoreError("text must not be empty")
    if layer == "shared" and source != "user":
        raise CoreError("only the user writes the shared layer")
    if layer == "shared":
        enable_shared(cwd)
    path = layer_file(cwd, layer, create=True)
    cf = load(path)
    it = Item(item_id(section, text), section, text, time.strftime("%Y-%m-%d"), source)
    if any(x.id == it.id for x in cf.items()):
        return it
    for s, lines in cf.blocks:
        if s == section:
            while lines and not lines[-1].strip():
                lines.pop()
            lines.append(render_bullet(it))
            break
    else:
        cf.blocks.append((section, [render_bullet(it)]))
    _write(path, dump(cf))
    if layer != "shared":
        commit(f"add {it.id} to {path.name} ({section})")
    _reindex(path, layer)
    return it


def remove(cwd: str | os.PathLike, mid: str) -> Path | None:
    """Delete the item with id *mid* from whichever layer holds it."""
    for layer, path in _all_layers(cwd):
        cf = load(path)
        hit = False
        for section, lines in cf.blocks:
            keep = []
            for line in lines:
                it = parse_bullet(line, section, "") if section else None
                if it and it.id == mid:
                    hit = True
                    continue
                keep.append(line)
            lines[:] = keep
        if hit:
            _write(path, dump(cf))
            if layer != "shared":
                commit(f"remove {mid} from {path.name}")
            _reindex(path, layer)
            return path
    return None


def find(cwd: str | os.PathLike, mid: str) -> tuple[str, Item] | None:
    for layer, path in _all_layers(cwd):
        for it in load(path).items():
            if it.id == mid:
                return layer, it
    return None


def _all_layers(cwd) -> list[tuple[str, Path]]:
    out = []
    for layer in LAYERS:
        p = layer_file(cwd, layer)
        if p:
            out.append((layer, p))
    return out


def _reindex(path: Path, layer: str) -> None:
    conn = open_index()
    try:
        sync_file(conn, path, layer, force=True)
    finally:
        conn.close()


# ── proposals ────────────────────────────────────────────────────────────

def propose(project: str, op: str, origin: str, source: str, *, layer: str = "private",
            section: str = "Decisions", text: str = "", target: str = "", chunk_id: str = "") -> str | None:
    """Queue a suggested change for /core review. Returns the proposal id, None if a duplicate."""
    if op not in ("add", "remove"):
        raise CoreError(f"unknown op {op!r}")
    if layer not in ("private", "global"):
        raise CoreError("proposals target the private or global layer; shared is user-written only")
    if op == "add" and section not in SECTIONS:
        raise CoreError(f"unknown section {section!r}; one of {', '.join(SECTIONS)}")
    text = " ".join(scrub(text).split())
    if op == "add" and not text:
        raise CoreError("text must not be empty")
    pid = hashlib.sha256(f"{project}\n{op}\n{layer}\n{section}\n{text}\n{target}".encode()).hexdigest()[:8]
    conn = open_index()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO core_proposals (id, project, layer, op, section, text, target, origin, "
            "source, chunk_id, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, project, layer, op, section, text, target, origin, source, chunk_id, int(time.time())),
        )
        conn.commit()
        return pid if cur.rowcount else None
    finally:
        conn.close()


def proposals(projects: list[str], status: str = "open") -> list[sqlite3.Row]:
    conn = open_index()
    try:
        marks = ",".join("?" * len(projects))
        return conn.execute(
            f"SELECT * FROM core_proposals WHERE status = ? AND project IN ({marks}) ORDER BY ts",
            (status, *projects),
        ).fetchall()
    finally:
        conn.close()


def proposal_projects(cwd: str | os.PathLike) -> list[str]:
    return [common.project_key(cwd), common.GLOBAL_PROJECT_HASH]


def _set_status(pid: str, status: str) -> None:
    conn = open_index()
    try:
        conn.execute("UPDATE core_proposals SET status = ? WHERE id = ?", (status, pid))
        conn.commit()
    finally:
        conn.close()


def accept(cwd: str | os.PathLike, pid: str, layer: str | None = None) -> str:
    """Apply an open proposal (user-typed /core accept). Returns a status line."""
    rows = [r for r in proposals(proposal_projects(cwd)) if r["id"] == pid]
    if not rows:
        raise CoreError(f"no open proposal {pid!r}")
    r = rows[0]
    if r["op"] == "remove":
        path = remove(cwd, r["target"])
        _set_status(pid, "accepted")
        return f"removed {r['target']} from {path.name}" if path else f"{r['target']} was already gone"
    it = add(cwd, r["text"], r["section"], layer or r["layer"], source=r["source"])
    _set_status(pid, "accepted")
    return f"added {it.id} to {layer or r['layer']} ({it.section})"


def reject(cwd: str | os.PathLike, pid: str) -> None:
    if not any(r["id"] == pid for r in proposals(proposal_projects(cwd))):
        raise CoreError(f"no open proposal {pid!r}")
    _set_status(pid, "rejected")


def describe(r: sqlite3.Row) -> str:
    """One line for review; instruction-shaped text is flagged."""
    if r["op"] == "remove":
        what = f"remove m:{r['target']}" + (f" — {r['text']}" if r["text"] else "")
    else:
        what = f"[{r['layer']}/{r['section']}] {r['text']}"
    flag = " [looks like an instruction]" if r["text"] and provenance.looks_imperative(r["text"]) else ""
    return f"{what} ({r['origin']}, {r['source']}){flag}"


def render_proposed(cwd: str | os.PathLike) -> Path | None:
    """Write <repo>.proposed.md (git-ignored) so proposals can be read as a file."""
    rows = proposals(proposal_projects(cwd))
    p = private_file(cwd)
    if p is None or not rows:
        return None
    out = [f"# Proposed core memory: {p.stem}", "",
           "Accept with `/biss-brain:core accept <id>`, reject with `/biss-brain:core reject <id>`.", ""]
    for r in rows:
        out.append(f"- `{r['id']}` {describe(r)}")
    path = p.with_name(p.stem + ".proposed.md")
    _write(path, "\n".join(out) + "\n")
    return path


# ── injection ────────────────────────────────────────────────────────────

_LAYER_SOURCE = {"shared": "repo", "private": None, "global": None}


def frame(cwd: str | os.PathLike, cfg: dict) -> tuple[str, int, int]:
    """Core memory as escaped <brain-memory> blocks: project layers first,
    then global, at most core.max_items bullets in total.

    Returns (text, shown, total). Dropped items are announced in the text,
    never cut silently.
    """
    limit = int(cfg.get("core", {}).get("max_items", 200))
    shown = total = 0
    blocks = []
    for layer, path in layers_for(cwd):
        by_section: dict[str, list[Item]] = {}
        for it in load(path).items():
            total += 1
            if shown < limit:
                by_section.setdefault(it.section, []).append(it)
                shown += 1
        for section, items in by_section.items():
            src = _LAYER_SOURCE[layer]
            blocks.append(f'<brain-memory kind="core" layer="{layer}" section="{_attr(section)}">')
            for it in items:
                label = src or it.source
                # scrub on read too: a hand-edited file was never scrubbed on write
                blocks.append(html.escape(f"- {scrub(it.text)} ({label}, {it.date}, m:{it.id})", quote=False))
            blocks.append("</brain-memory>")
    if not blocks:
        return "", 0, total
    lines = ["## Core memory (biss-brain)", "",
             "Curated notes for this project; shared-layer items come from the repo.", "", *blocks]
    if total > shown:
        lines += ["", f"({total - shown} more core items not shown: limit {limit}. "
                      "Run /biss-brain:core to review and trim.)"]
    return "\n".join(lines), shown, total


def _attr(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _-]", "", s)[:40]
