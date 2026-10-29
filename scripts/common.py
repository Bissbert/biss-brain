"""Shared helpers: config, paths, project keying, SQLite schema, LanceDB schema."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BRAIN_ROOT = (
    Path(os.environ["BISS_BRAIN_ROOT"]).expanduser()
    if os.environ.get("BISS_BRAIN_ROOT")
    else Path.home() / ".claude" / "brain"
)

# sha256("__global__")[:12] — pinned in code; never recomputed.
GLOBAL_PROJECT_HASH = "5da0e6b1ad8a"
CONFIG_PATH = BRAIN_ROOT / "config.toml"
DEFAULT_CONFIG: dict[str, Any] = {
    # hooks do nothing in these directories (scripts/scope.py); "~" and "/"
    # match exactly, other entries match the whole tree
    "scope": {
        "exclude": ["~", "~/Downloads", "~/Library", "~/.claude", "/tmp"],
        "include_only": [],
    },
    "distill": {
        "backend": "claude_code",
        "models": {
            "stop": "haiku",
            "subagent_stop": "haiku",
            "pre_compact": "sonnet",
            "conflict_judge": "haiku",
        },
        "ollama": {"model": "qwen2.5-coder:14b-instruct"},
        # backends distill() tries in order when the primary returns nothing (distill.chain)
        "fallback": [],
        # a failed segment is retried after base, 2*base, … up to max seconds (pending.py)
        "retry_base_seconds": 60,
        "retry_max_seconds": 3600,
        # a window whose reply is invalid this many times in a row is skipped (pending.skip)
        "skip_invalid_after": 5,
    },
    "debounce": {
        "stop_seconds": 30,
        "subagent_stop_seconds": 10,
        "max_events_per_batch": 50,
        # a segment closes at SessionEnd, PreCompact, this much idle, or this many events
        "segment_idle_seconds": 600,
        "segment_max_events": 20,
        # the daemon exits after this long with an empty queue
        "idle_exit_seconds": 300,
    },
    "embed": {
        "backend": "ollama",
        "model": "nomic-embed-text",
        "dim": 768,
        "endpoint": "http://127.0.0.1:11434",
        # how long ollama keeps the model loaded after a request; a cold load
        # misses the prompt hook's embed budget
        "keep_alive": "30m",
        # embedder down at store time: keep the distilled chunk without a vector
        # and embed it later (flush._backfill), at most once per interval
        "store_without_vector": True,
        "backfill_interval_seconds": 60,
    },
    "retrieve": {
        "session_start_top_k": 3,
        # inject a compaction summary that landed after the digest restore (inject_context)
        "compact_late_restore": True,
        "mcp_default_top_k": 5,
        "hybrid": True,
        "rerank": False,
        # final score = relevance / max relevance in pool × decay_score ** decay_weight
        "decay_weight": 0.05,  # tuned on the v0.7 eval
        "pool_factor": 5,
        "pool_min": 25,
        # UserPromptSubmit injection (retrieve.prompt_inject); gate fit on the v0.7 eval
        "user_prompt_inject": False,
        # cosine OR bm25 (calibrate_inject.py, v0.7, in-sample: precision 0.81 real / 0.75
        # synthetic, 0.60-0.68 cross-suite; off-topic abstention 0.94 / 1.0; poison 0);
        # bm25 alone when the embed times out
        "inject_min_cosine": 0.6,
        "inject_min_bm25": 7.0,
        "inject_top_k": 1,
        "inject_max_k": 3,
        "inject_max_chars": 6000,
        "inject_min_words": 3,
        "inject_embed_ms": 300,
        "inject_deadline_ms": 2500,
        "inject_p95_target_ms": 800,  # doctor warns above this
    },
    # v0.7 provenance policy (see provenance.py)
    "provenance": {
        "trust": {"user": 1.0, "assistant": 0.8, "file": 0.7, "tool": 0.5, "imported": 0.5, "web": 0.3},
        "no_inject_sources": ["tool", "web"],  # never auto-injected, by taint
        "rank_multiplier": {"tool": 0.6, "web": 0.6},
        "quarantine_imperatives": True,
    },
    "hygiene": {
        "dedup_cosine": 0.94,
        "decay_half_life_days": 30,
        # per-retrieval boost in the nightly decay recompute; 0.2 before v0.7.
        # Off: no gain on either suite (docs/eval/v0.7-ablation.md).
        "decay_usage_boost": 0.0,
        "pin_immune": True,
        "conflict_judge_enabled": True,
        "conflict_cosine": 0.85,  # min cosine before a new chunk is judged against an old one
    },
    "dream": {
        "enabled": True,
        "session_cooldown_hours": 6,
        "min_chunks_per_arc": 3,
        "daily_summary_enabled": False,
        "prefer_summaries_in_recall": False,
        "models": {
            "arc": "haiku",
            "daily": "sonnet",
        },
        "stage1": {
            "backlink_refresh": True,
            "conflict_sweep": True,
            "conflict_sim_threshold": 0.85,
            "candidate_window_hours": 24,
        },
    },
    "capture": {
        # raw JSONL read from the transcript tail before parsing (harness records
        # such as api-request-blob lines are large and are dropped)
        "raw_tail_bytes": 2_000_000,
        "transcript_window_bytes": 16000,
        # one distill call per window; a segment is packed into at most N windows
        "segment_window_chars": 48000,
        "segment_max_windows": 3,
        # conversation tail restored after compaction until the summary is distilled
        "compact_digest_chars": 6000,
        "scrub_secrets": True,
        # v0.5 salience keys — use_llm is PERMANENT default-off
        "salience": {
            "use_llm": False,
            "heuristic": {
                "length_weight": 0.4,
                "novelty_weight": 0.6,
            },
        },
    },
    # v0.5 keys below ─────────────────────────────────────────────────────
    "reconsolidation": {
        "enabled": True,
        "labile_window_seconds": 300,
        "merge_cosine_threshold": 0.82,
        # merge only when the new text adds this share of new content terms;
        # a restatement is folded into the used memory without an LLM call
        "revise_min_novelty": 0.2,
        "promote_to_lesson_after": 5,
    },
    "stdp": {
        "enabled": True,
        "co_retrieval_min_count": 3,
        "edge_drain_window_days": 7,
    },
    "rif": {
        # Off since v0.7: ΔMRR -0.105 on the real suite and ~170 ms per read,
        # and dream's decay pass erases it nightly (docs/eval/v0.7-ablation.md).
        "enabled": False,
        "shared_neighbour_threshold": 3,
        "decay_multiplier": 0.98,
    },
    # Core memory (core.py): curated files injected in full at SessionStart.
    "core": {
        "inject": True,
        "max_items": 200,
        "warn_ratio": 0.8,
        "proposals_per_dream": 5,
        "propose_min_retrievals": 3,
    },
    "dmn": {
        "enabled": True,
        "idle_seconds_before_run": 300,
        "max_runtime_seconds": 60,
    },
    # cross_project.enabled is PERMANENT default-off (product decision)
    "cross_project": {
        "enabled": False,
        "admission": {
            "min_consolidation_round": 1,
            "eligible_kinds": ["schema", "lesson", "hypothesis"],
            "frequency": {
                "min_retrievals": 5,
                "distinct_projects": 3,
                "window_days": 30,
            },
        },
        "recall": {
            "cosine_gate": 0.85,
            "label_with_source_project": True,
        },
        "backlinks": {
            "cosine_gate": 0.85,
        },
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict[str, Any]:
    config_path = Path(os.environ.get("BISS_BRAIN_CONFIG", "")) if os.environ.get("BISS_BRAIN_CONFIG") else CONFIG_PATH
    if not config_path.exists():
        return DEFAULT_CONFIG
    with config_path.open("rb") as f:
        user = tomllib.load(f)
    r = user.get("retrieve", {})
    for old, new in _RENAMED_RETRIEVE.items():
        if old in r and new not in r:
            r[new] = r[old]
    return _deep_merge(DEFAULT_CONFIG, user)


# pre-0.7 key names, still read from user config files
_RENAMED_RETRIEVE = {
    "user_prompt_inject_min_cosine": "inject_min_cosine",
    "user_prompt_inject_top_k": "inject_top_k",
}


def _normalize_worktree(path: Path) -> Path:
    """Return the canonical main-repo root for a worktree path.

    When Claude Code runs inside a git worktree at
    ``~/.claude/worktrees/<name>``, the directory is linked to the main
    repository via a ``.git`` *file* (not a directory) that contains a line
    of the form::

        gitdir: /path/to/main/.git/worktrees/<name>

    We parse that line and walk up from ``.git/worktrees/<name>`` to find the
    main repo root (three levels up: ``worktrees/<name>`` → ``worktrees`` →
    ``.git`` → repo root).

    Edge-cases handled gracefully (return *path* unchanged):
    - Path does not contain ``.claude/worktrees/`` — not a worktree at all.
    - ``.git`` file is absent, is a real directory (bare clone), or unreadable.
    - The resolved gitdir path does not exist on disk.
    - Any other IO or parse error.
    """
    parts = path.parts
    if ".claude" not in parts:
        return path

    # Find the segment sequence .claude / worktrees / <name>
    try:
        idx = parts.index(".claude")
    except ValueError:
        return path

    if idx + 2 >= len(parts) or parts[idx + 1] != "worktrees":
        return path

    # Candidate worktree root: .claude/worktrees/<name>
    worktree_root = Path(*parts[: idx + 3])

    git_file = worktree_root / ".git"
    if not git_file.is_file():
        # .git is a directory (regular clone) or missing — not a linked worktree
        return path

    try:
        content = git_file.read_text(encoding="utf-8").strip()
    except OSError:
        return path

    # Expected format: "gitdir: /abs/path/to/main/.git/worktrees/<name>"
    if not content.startswith("gitdir:"):
        return path

    gitdir_str = content[len("gitdir:"):].strip()
    gitdir = Path(gitdir_str)

    if not gitdir.exists():
        return path

    # Navigate: .git/worktrees/<name>  →  .git/worktrees  →  .git  →  repo root
    try:
        repo_root = gitdir.parent.parent.parent
    except Exception:
        return path

    if not repo_root.is_dir():
        return path

    return repo_root


def _gitdir_of(dotgit: Path) -> Path | None:
    """The git dir a `.git` file points to (relative paths resolve from its directory)."""
    try:
        line = dotgit.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    if not line.startswith("gitdir:"):
        return None
    gd = Path(line[len("gitdir:"):].strip())
    gd = Path(os.path.realpath(dotgit.parent / gd))
    return gd if gd.is_dir() else None


def repo_root(cwd: str | os.PathLike) -> Path:
    """Main worktree root of the repo holding cwd, without spawning git; realpath otherwise.

    Walks up to the nearest `.git`. A directory is a plain clone. A file is a
    linked worktree (its git dir has `commondir`; the root is the main
    checkout) or a submodule (no `commondir`; the root is the submodule
    itself). A bare common dir, or no repo below $HOME, keeps the realpath.
    $HOME and / are never a project root.
    """
    real = Path(os.path.realpath(str(cwd)))
    home = Path(os.path.realpath(os.path.expanduser("~")))
    for d in (real, *real.parents):
        if d in (home, Path("/")):
            break
        dotgit = d / ".git"
        if dotgit.is_dir():
            return d
        if not dotgit.is_file():
            continue
        gd = _gitdir_of(dotgit)
        if gd is None:
            return d
        try:
            common_dir = Path(os.path.realpath(gd / (gd / "commondir").read_text().strip()))
        except OSError:
            if gd.parent.name != "worktrees":
                return d  # a submodule is its own project
            common_dir = gd.parent.parent  # <repo>/.git/worktrees/<name> without commondir
        return common_dir.parent if common_dir.name == ".git" and common_dir.is_dir() else d
    return _normalize_worktree(real)


def project_key(cwd: str | os.PathLike) -> str:
    """One store per repo: every worktree and subdirectory shares the main checkout's key."""
    return hashlib.sha256(str(repo_root(cwd)).encode()).hexdigest()[:12]


def project_dir(cwd: str | os.PathLike) -> Path:
    p = BRAIN_ROOT / project_key(cwd)
    p.mkdir(parents=True, exist_ok=True)
    return p


def queue_dir(cwd: str | os.PathLike) -> Path:
    q = project_dir(cwd) / "queue"
    q.mkdir(exist_ok=True)
    return q


def lancedb_dir(cwd: str | os.PathLike) -> Path:
    return project_dir(cwd) / "lancedb"


def meta_db_path(cwd: str | os.PathLike) -> Path:
    return project_dir(cwd) / "meta.sqlite"


def open_meta(cwd: str | os.PathLike) -> sqlite3.Connection:
    conn = sqlite3.connect(str(meta_db_path(cwd)), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the meta tables and apply every additive migration (idempotent).

    The single schema definition: open_meta, dream and dmn all call it.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            session_id TEXT,
            agent_id TEXT,
            kind TEXT NOT NULL,
            ts INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            superseded_by TEXT NOT NULL DEFAULT '',
            pinned INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            last_accessed INTEGER,
            raw_ref TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_project_kind_ts ON chunks(project, kind, ts DESC)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_super ON chunks(superseded_by)")

    # Migration: add decay_score column if missing (added in Phase 3.4)
    existing_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
    }
    if "decay_score" not in existing_cols:
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN decay_score REAL NOT NULL DEFAULT 1.0"
        )

    # Migration: add related_ids column if missing (added in Phase 4.2)
    if "related_ids" not in existing_cols:
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN related_ids TEXT NOT NULL DEFAULT ''"
        )

    # Migration: dream pass columns (added in Phase 5)
    if "parent_id" not in existing_cols:
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN parent_id TEXT NOT NULL DEFAULT ''"
        )
    if "evidence_ids" not in existing_cols:
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN evidence_ids TEXT NOT NULL DEFAULT ''"
        )
    if "consolidation_round" not in existing_cols:
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN consolidation_round INTEGER NOT NULL DEFAULT 0"
        )
    if "last_dream_ts" not in existing_cols:
        conn.execute(
            "ALTER TABLE chunks ADD COLUMN last_dream_ts INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_round ON chunks(consolidation_round)"
    )

    # migration_v05 — additive, idempotent
    existing_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
    }
    _v05_chunk_cols: list[tuple[str, str]] = [
        ("salience", "INTEGER DEFAULT 0"),
        ("labile_until", "INTEGER DEFAULT 0"),
        ("reconsolidation_count", "INTEGER DEFAULT 0"),
        ("co_retrieval_count", "INTEGER DEFAULT 0"),
        ("cross_links", "TEXT DEFAULT '[]'"),
        ("cross_eligible", "INTEGER DEFAULT 0"),
        ("retrieval_count_local", "INTEGER DEFAULT 0"),
    ]
    for col_name, col_def in _v05_chunk_cols:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_def}")

    # v0.7 provenance (see provenance.py). Pre-v0.7 rows read as imported.
    for col_name, col_def in (
        ("source", "TEXT NOT NULL DEFAULT 'imported'"),
        ("trust", "REAL NOT NULL DEFAULT 0.5"),
        ("taint", "TEXT NOT NULL DEFAULT '[\"imported\"]'"),
    ):
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_def}")

    # v0.7 temporal validity (see temporal.py). valid_from NULL reads as ts.
    if "valid_from" not in existing_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN valid_from INTEGER")
    if "invalid_at" not in existing_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN invalid_at INTEGER")
        # Earlier judge verdicts pointed superseded_by at the newer chunk;
        # those rows become invalid from the newer chunk's time. Tombstones
        # (superseded_by naming no chunk) stay invalid_at NULL.
        conn.execute("""
            UPDATE chunks SET invalid_at = (
                SELECT MAX(COALESCE(s.valid_from, s.ts), COALESCE(chunks.valid_from, chunks.ts))
                FROM chunks s WHERE s.project = chunks.project
                AND s.id IN (chunks.superseded_by)
            )
            WHERE superseded_by != '' AND EXISTS (
                SELECT 1 FROM chunks s WHERE s.project = chunks.project
                AND s.id IN (chunks.superseded_by)
            )
        """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_bumps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project     TEXT NOT NULL,
            chunk_a     TEXT NOT NULL,
            chunk_b     TEXT NOT NULL,
            bumped_at   INTEGER NOT NULL,
            session_id  TEXT,
            drained     INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_edge_bumps_pair
        ON edge_bumps (project, chunk_a, chunk_b)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_edge_bumps_undrained
        ON edge_bumps (project, drained, bumped_at)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dream_state (
            project         TEXT PRIMARY KEY,
            last_stage      TEXT NOT NULL DEFAULT 'IDLE',
            last_complete_ts INTEGER NOT NULL DEFAULT 0,
            last_error      TEXT NOT NULL DEFAULT '',
            last_error_ts   INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_labile
        ON chunks (project, labile_until, superseded_by)
    """)

    conn.commit()


@dataclass
class Event:
    """One capture event written to the queue, drained by flush.py."""
    kind: str  # stop | subagent_stop | pre_compact | session_end
    session_id: str
    agent_id: str | None
    cwd: str
    transcript_path: str
    transcript_excerpt: str
    payload: dict[str, Any] = field(default_factory=dict)
    enqueued_ts: int = field(default_factory=lambda: int(time.time()))
    # v0.7: raw provenance sources present in the (tagged) excerpt. None marks a
    # pre-v0.7 event whose excerpt is untagged raw JSONL.
    sources: list[str] | None = None
    # v0.7 5.4: transcript byte offset at capture time. The daemon reads the
    # segment from the transcript; transcript_excerpt then only holds the
    # scrubbed last assistant message as a fallback. None: inline excerpt.
    offset: int | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Event":
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def ensure_project_label(cwd: str | os.PathLike) -> None:
    """Write project_path.txt into the project hash dir if it doesn't exist yet.

    Safe to call on every capture event – idempotent, low-overhead.
    """
    pdir = project_dir(cwd)
    label_file = pdir / "project_path.txt"
    if not label_file.exists():
        label_file.write_text(str(repo_root(cwd)))


def discover_projects() -> list[dict]:
    """Walk ~/.claude/brain/*/ and return metadata for each project.

    Returns a list of dicts with keys:
      hash          – 12-hex project key
      path          – str (from project_path.txt) or None if missing
      chunk_count   – int
      last_active_ts – int (unix seconds) or None
    """
    results: list[dict] = []
    if not BRAIN_ROOT.exists():
        return results

    for entry in sorted(BRAIN_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        # Two project shapes are admitted:
        #   - 12-char hex per-project hash dirs
        #   - the literal "_global" cross-project pool, surfaced under
        #     GLOBAL_PROJECT_HASH so callers can address it uniformly
        is_global = (name == "_global")
        if not is_global and (
            len(name) != 12 or not all(c in "0123456789abcdef" for c in name)
        ):
            continue

        emit_hash = GLOBAL_PROJECT_HASH if is_global else name

        # Read optional path label
        label_file = entry / "project_path.txt"
        path: str | None = label_file.read_text().strip() if label_file.exists() else None

        # Quick stats from SQLite
        chunk_count = 0
        last_active_ts: int | None = None
        db_path = entry / "meta.sqlite"
        if db_path.exists():
            try:
                conn = sqlite3.connect(
                    f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
                )
                row = conn.execute(
                    "SELECT COUNT(*), MAX(ts) FROM chunks"
                ).fetchone()
                conn.close()
                if row:
                    chunk_count = row[0] or 0
                    last_active_ts = row[1]  # may be None if no rows
            except Exception:
                pass

        # Skip truly empty orphan dirs: no chunks AND no path label.
        # _global is exempt: the index pin shows an empty-state guidance
        # row when the cross-project pool has not yet been populated.
        if chunk_count == 0 and path is None and not is_global:
            continue

        results.append(
            {
                "hash": emit_hash,
                "path": path,
                "chunk_count": chunk_count,
                "last_active_ts": last_active_ts,
            }
        )

    return results


def project_dir_for(project_hash: str) -> Path:
    """Return the on-disk directory for a project hash.

    Translates GLOBAL_PROJECT_HASH to the literal `_global` directory
    used by dream.py / recall.py / forget.py. All other hashes map
    directly. Centralised so UI and CLI tools agree on the lookup.
    """
    if project_hash == GLOBAL_PROJECT_HASH:
        return BRAIN_ROOT / "_global"
    return BRAIN_ROOT / project_hash


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_transcript_tail(path: str, max_bytes: int) -> str:
    """Return the last `max_bytes` of a transcript JSONL file as text."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # skip partial first line
        return f.read().decode("utf-8", errors="replace")


def now_unix() -> int:
    return int(time.time())


def scoped(sql: str, params: tuple, project: str | None) -> tuple[str, tuple]:
    """Append `AND project = ?` to a meta query when a project is given."""
    return (sql + " AND project = ?", (*params, project)) if project else (sql, params)


def tombstone_id() -> str:
    """Return a fresh opaque tombstone uuid for soft-delete (superseded_by)."""
    import uuid
    return uuid.uuid4().hex


def _fake_embed(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-random vector seeded on text hash.

    Used in tests and when BISS_BRAIN_FAKE_EMBED is set.
    Canonical implementation — all scripts import from here.
    """
    import hashlib as _hashlib
    import random as _rng
    seed = int(_hashlib.md5(text.encode()).hexdigest(), 16)
    rng = _rng.Random(seed)
    return [rng.random() for _ in range(dim)]


# ── embeddings and vector search ─────────────────────────────────────────
#
# Every vector the brain stores or queries with is unit-norm, and every vector
# query uses the cosine metric, so `_distance` = 1 - cosine and similarity gates
# are plain cosine thresholds. (v0.6 stored raw Ollama vectors with norm ~15 and
# gated on L2 thresholds below 1, so backlinks, conflict checks and DMN
# neighbours almost never fired.)

def normalize(vec: Any) -> list[float]:
    import math  # noqa: PLC0415

    v = [float(x) for x in vec]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def cosine(a: Any, b: Any) -> float:
    import math  # noqa: PLC0415

    num = sum(float(x) * float(y) for x, y in zip(a, b))
    den = math.sqrt(sum(float(x) ** 2 for x in a)) * math.sqrt(sum(float(y) ** 2 for y in b))
    return num / den if den > 0 else 0.0


def embed(text: str, cfg: dict, purpose: str = "document", timeout: float = 60) -> list[float]:
    """Embed one text with the configured model; returns a unit vector.

    `purpose` is "query" or "document". Models trained with task prefixes
    (nomic, snowflake-arctic, embeddinggemma, mxbai) get
    `embed.query_prefix` / `embed.document_prefix` prepended.
    """
    e = cfg["embed"]
    if os.environ.get("BISS_BRAIN_FAKE_EMBED"):
        return normalize(_fake_embed(text, e["dim"]))
    import httpx  # noqa: PLC0415

    prefix = e.get("query_prefix" if purpose == "query" else "document_prefix", "") or ""
    r = httpx.post(
        f"{e['endpoint']}/api/embed",
        json={"model": e["model"], "input": prefix + text, "keep_alive": e.get("keep_alive", "30m")},
        timeout=timeout,
    )
    r.raise_for_status()
    vecs = r.json().get("embeddings") or []
    if not vecs or not vecs[0]:
        raise RuntimeError("ollama returned empty embedding")
    return normalize(vecs[0])


def _with_cosine(rows: list[dict], qvec: list[float]) -> list[dict]:
    for r in rows:
        if "_cosine" in r:
            continue
        if "_distance" in r and r.get("_distance") is not None and "_relevance_score" not in r:
            r["_cosine"] = 1.0 - float(r["_distance"])
        elif r.get("vector") is not None:
            r["_cosine"] = cosine(qvec, r["vector"])
        else:
            r["_cosine"] = 0.0
    return rows


def vector_search(tbl: Any, qvec: list[float], where_sql: str | None, limit: int) -> list[dict]:
    """Dense search with the cosine metric; rows carry `_cosine`."""
    q = tbl.search(qvec).distance_type("cosine")
    if where_sql:
        q = q.where(where_sql)
    return _with_cosine(q.limit(limit).to_list(), qvec)


# ── store marker: schema version and embed fingerprint ───────────────────
# <brain>/store.json records the embed model the vectors were made with. A
# store without it (pre-v0.7, eval snapshots, test roots) is tolerated.
STORE_MARKER = "store.json"
SCHEMA_VERSION = 7
MISMATCH_HINT = ("restore the [embed] settings in config.toml, or start a new store with "
                 "scripts/fresh_start.py (the old one is archived, never deleted)")


def embed_fingerprint(cfg: dict) -> dict:
    """What decides the vector space: model, dim and the task prefixes."""
    e = cfg["embed"]
    return {"model": e["model"], "dim": int(e["dim"]),
            "query_prefix": e.get("query_prefix", "") or "",
            "document_prefix": e.get("document_prefix", "") or ""}


def read_store_marker(root: Path | None = None) -> dict | None:
    try:
        d = json.loads(((root or BRAIN_ROOT) / STORE_MARKER).read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def write_store_marker(cfg: dict, root: Path | None = None, schema_only: bool = False) -> Path:
    """Write the marker. schema_only keeps an existing fingerprint (or none):
    stored vectors cannot prove which model made them."""
    root = root or BRAIN_ROOT
    d = read_store_marker(root) or {}
    d["schema_version"] = SCHEMA_VERSION
    if not schema_only:
        d["embed"] = embed_fingerprint(cfg)
        d["created"] = d.get("created") or int(time.time())
    root.mkdir(parents=True, exist_ok=True)
    f = root / STORE_MARKER
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    os.replace(tmp, f)
    return f


def store_mismatch(cfg: dict, root: Path | None = None) -> str | None:
    """How the config's embed settings differ from the store's, or None."""
    have = (read_store_marker(root) or {}).get("embed")
    if not isinstance(have, dict):
        return None
    want = embed_fingerprint(cfg)
    diffs = [f"{k} {have.get(k)!r} in the store, {v!r} in config"
             for k, v in want.items() if have.get(k, "" if k.endswith("prefix") else None) != v]
    return "; ".join(diffs) or None


def claim_store(cfg: dict, root: Path | None = None) -> bool:
    """Fingerprint a store that has no marker and no vectors yet."""
    root = root or BRAIN_ROOT
    if read_store_marker(root) is not None or any(root.glob("*/lancedb")):
        return False
    write_store_marker(cfg, root)
    return True


# ── chunks table and FTS index ───────────────────────────────────────────

def chunks_schema(dim: int) -> Any:
    import pyarrow as pa  # noqa: PLC0415

    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("project", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("agent_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("ts", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("superseded_by", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


def open_or_create_chunks(lance_path: str | os.PathLike, dim: int) -> Any:
    import lancedb  # noqa: PLC0415

    db = lancedb.connect(str(lance_path))
    if "chunks" in db.list_tables().tables:
        return db.open_table("chunks")
    tbl = db.create_table("chunks", schema=chunks_schema(dim), mode="create")
    refresh_fts(tbl)
    return tbl


def refresh_fts(tbl: Any) -> bool:
    """Make sure the FTS index exists and covers every row.

    A LanceDB FTS index only covers the rows present when it was built or last
    optimized; later rows are flat-scanned, and some queries over unindexed
    rows crash the hybrid search. Call after every write batch (cheap: ~10 ms
    for small tables). Returns False on failure (logged, never raised).
    """
    import logging  # noqa: PLC0415

    try:
        if not any(i.index_type == "FTS" and "text" in i.columns for i in tbl.list_indices()):
            tbl.create_fts_index("text", replace=True)
        tbl.optimize()
        return True
    except Exception as e:  # noqa: BLE001
        logging.getLogger("biss-brain").warning("FTS refresh failed: %s", e)
        return False


def fts_unindexed_rows(tbl: Any) -> int | None:
    """Rows the FTS index does not cover yet (None if there is no index)."""
    try:
        for i in tbl.list_indices():
            if i.index_type == "FTS":
                return tbl.index_stats(i.name).num_unindexed_rows
    except Exception:  # noqa: BLE001
        pass
    return None


# Number of hybrid queries in this process that failed and fell back to dense.
HYBRID_FALLBACKS = 0


def hybrid_search(
    tbl: Any,
    qvec: list[float],
    qtext: str,
    where_sql: str | None,
    top_k: int,
) -> list[dict]:
    """Hybrid (BM25 + dense, RRF-fused) search; falls back to dense on failure.

    Rows carry `_relevance_score` (RRF, hybrid only) and `_cosine` (always).
    Fallbacks are counted in HYBRID_FALLBACKS so the eval and doctor see them.
    """
    try:
        q = tbl.search(query_type="hybrid").text(qtext).vector(qvec).distance_type("cosine")
        if where_sql:
            q = q.where(where_sql)
        return _with_cosine(q.limit(top_k).to_list(), qvec)
    except Exception:
        global HYBRID_FALLBACKS
        HYBRID_FALLBACKS += 1
    return vector_search(tbl, qvec, where_sql, top_k)


def fetch_by_ids(tbl: Any, ids: list[str], limit: int | None = None) -> list[dict]:
    """Fetch rows from a LanceDB table by id list (no vector query needed)."""
    if not ids:
        return []
    import lance_filter as F  # noqa: PLC0415

    return tbl.search().where(F.isin("id", ids)).limit(limit if limit is not None else len(ids)).to_list()
