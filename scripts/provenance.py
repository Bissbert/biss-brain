"""Memory provenance: where a memory came from and how far to trust it.

Every chunk carries three meta columns:
  source  what wrote it: user | assistant | file | tool | web | distilled | imported
  trust   0–1; a derived memory is never more trusted than its least-trusted input
  taint   JSON list of the raw sources that fed it (propagates through merges,
          arcs and promotion, so a summary of web content stays web-tainted)

Policy (read side) keys off taint, not source, so derived rows inherit it:
  - rows tainted by a `no_inject_sources` entry are never auto-injected
  - search scores are multiplied by the lowest `rank_multiplier` in the taint
  - unknown provenance (no meta row) is never injected, but ranks as `imported`
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

SOURCES = ("user", "assistant", "file", "tool", "web", "distilled", "imported")
RAW_SOURCES = ("user", "assistant", "file", "tool", "web", "imported")

DEFAULT_TRUST = {
    "user": 1.0,
    "assistant": 0.8,
    "file": 0.7,
    "tool": 0.5,
    "imported": 0.5,
    "web": 0.3,
}

# Tool names whose results are third-party content.
_WEB_TOOLS = {"WebFetch", "WebSearch"}
_FILE_TOOLS = {"Read", "Glob", "Grep", "NotebookRead", "LS"}


@dataclass(frozen=True)
class Provenance:
    source: str
    trust: float
    taint: tuple[str, ...]

    def row(self) -> tuple[str, float, str]:
        """(source, trust, taint_json) in meta column order."""
        return self.source, self.trust, json.dumps(list(self.taint))


def _cfg(cfg: dict | None) -> dict:
    return (cfg or {}).get("provenance", {})


def trust_of(source: str, cfg: dict | None = None) -> float:
    table = {**DEFAULT_TRUST, **_cfg(cfg).get("trust", {})}
    return float(table.get(source, table["imported"]))


def make(source: str, cfg: dict | None = None) -> Provenance:
    """Provenance for a memory written directly from one raw source."""
    if source not in RAW_SOURCES:
        raise ValueError(f"not a raw source: {source!r}")
    return Provenance(source, trust_of(source, cfg), (source,))


def derive(taint: Iterable[str], cfg: dict | None = None, source: str = "distilled") -> Provenance:
    """Provenance for a memory built from inputs with the given raw sources."""
    t = tuple(sorted({s for s in taint if s in RAW_SOURCES})) or ("imported",)
    return Provenance(source, min(trust_of(s, cfg) for s in t), t)


def combine(provs: Iterable[Provenance], cfg: dict | None = None, source: str = "distilled") -> Provenance:
    """Merge, arc or roll-up: taint is the union, trust the minimum."""
    provs = list(provs)
    if not provs:
        return derive((), cfg, source)
    p = derive((s for pv in provs for s in pv.taint), cfg, source)
    return Provenance(source, min([p.trust, *(pv.trust for pv in provs)]), p.taint)


def parse_taint(raw: Any) -> tuple[str, ...]:
    try:
        vals = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    except (json.JSONDecodeError, TypeError):
        return ("imported",)
    t = tuple(sorted({v for v in vals if v in RAW_SOURCES}))
    return t or ("imported",)


def from_row(source: Any, trust: Any, taint: Any) -> Provenance:
    src = source if source in SOURCES else "imported"
    try:
        tr = float(trust)
    except (TypeError, ValueError):
        tr = DEFAULT_TRUST["imported"]
    return Provenance(src, max(0.0, min(1.0, tr)), parse_taint(taint))


IMPORTED = Provenance("imported", DEFAULT_TRUST["imported"], ("imported",))

# Sources whose content is first-party (the conversation or the user's files).
TRUSTED_SOURCES = frozenset({"user", "assistant", "file", "imported"})


def trust_class(p: Provenance | None) -> str:
    """'trusted' when every taint source is first-party, else 'low'.

    Merges, arcs and roll-ups only combine memories of the same class, so
    third-party content never launders into a trusted summary.
    """
    return "trusted" if set((p or IMPORTED).taint) <= TRUSTED_SOURCES else "low"


def _taint_excludes(sources: Iterable[str], col: str) -> str:
    # Values come from RAW_SOURCES only, never from config text, so they are
    # safe to inline; anything else is dropped.
    parts = [f"{col} NOT LIKE '%\"{s}\"%'" for s in sorted(set(sources) & set(RAW_SOURCES))]
    return " AND ".join(parts) or "1=1"


def trusted_sql(col: str = "taint") -> str:
    """SQL predicate on a meta taint column: true for trusted-class rows."""
    return _taint_excludes(set(RAW_SOURCES) - TRUSTED_SOURCES, col)


def injectable_sql(cfg: dict | None, col: str = "taint") -> str:
    """SQL predicate on a meta taint column: true for auto-injectable rows."""
    return _taint_excludes(no_inject_sources(cfg), col)


def lookup(meta: Any, proj: str, ids: list[str]) -> dict[str, Provenance]:
    """Provenance per id from the project's meta db; missing ids are absent."""
    if meta is None or not ids:
        return {}
    marks = ",".join("?" * len(ids))
    where = f"FROM chunks WHERE project = ? AND id IN ({marks})"
    try:
        rows = meta.execute(f"SELECT id, source, trust, taint {where}", (proj, *ids)).fetchall()
        return {r[0]: from_row(r[1], r[2], r[3]) for r in rows}
    except Exception:  # noqa: BLE001 — pre-v0.7 meta (read-only) without the columns
        pass
    try:
        rows = meta.execute(f"SELECT id {where}", (proj, *ids)).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    return {r[0]: IMPORTED for r in rows}


def tool_source(tool_name: str | None) -> str:
    """Raw source of a tool result. Unknown names are `tool`, never `file`."""
    if not tool_name:
        return "tool"
    if tool_name in _WEB_TOOLS or tool_name.startswith("mcp__"):
        return "web"
    if tool_name in _FILE_TOOLS:
        return "file"
    return "tool"


# ── policy ───────────────────────────────────────────────────────────────

def no_inject_sources(cfg: dict | None) -> set[str]:
    return set(_cfg(cfg).get("no_inject_sources", ["tool", "web"]))


def injectable(p: Provenance | None, cfg: dict | None) -> bool:
    """Auto-injection is allowed only for known, untainted provenance."""
    return p is not None and not (set(p.taint) & no_inject_sources(cfg))


def rank_multiplier(p: Provenance | None, cfg: dict | None) -> float:
    mult = {"tool": 0.6, "web": 0.6, **_cfg(cfg).get("rank_multiplier", {})}
    taint = (p or IMPORTED).taint
    return min(float(mult.get(s, 1.0)) for s in taint)


def may_supersede(new: Provenance | None, old: Provenance | None) -> bool:
    """A memory may only supersede one it is at least as trusted as."""
    return (new or IMPORTED).trust >= (old or IMPORTED).trust


def label(p: Provenance | None) -> str:
    """Short source label for display: the least-trusted raw source."""
    p = p or IMPORTED
    return min(p.taint, key=lambda s: DEFAULT_TRUST.get(s, 0.5))


# Text that tries to instruct rather than inform. Only applied to low-trust
# memories, so a false positive costs little: the memory is still stored,
# just as kind=quarantine (never injected, hidden from model-facing search).
_IMPERATIVE = re.compile(
    r"""
    \b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|instructions?|rules?)\b
  | \b(you|claude|the\s+assistant|the\s+model|agents?)\s+(must|should|shall|need\s+to|are\s+(now\s+)?(required|instructed|allowed))\b
  | \b(always|never|from\s+now\s+on|going\s+forward)\b.{0,60}?\b(run|use|skip|disable|enable|send|commit|push|delete|store|include|trust|print|reveal)\b
  | \b(remember|record|save|store|treat)\b.{0,30}\b(this|that|it)\b.{0,30}\b(as|instruction|rule|preference)\b
  | \b(system\s+prompt|user\s+instruction|new\s+instructions?|developer\s+message)\b
  | </?\s*(system|brain-memory|instructions?|user|assistant)\b
  | [\[(]\s*(user|claude|assistant|system)\b[^\])\n]{0,20}[\])]
    """,
    re.I | re.X,
)


def looks_imperative(text: str) -> bool:
    return bool(_IMPERATIVE.search(text or ""))


def chunk_kind(kind: str, text: str, p: Provenance | None, cfg: dict | None) -> str:
    """*kind*, or 'quarantine' for low-trust text that reads like an instruction."""
    if not _cfg(cfg).get("quarantine_imperatives", True):
        return kind
    return "quarantine" if trust_class(p) == "low" and looks_imperative(text) else kind
